#!/usr/bin/env bash
# Simple Memory Settings: apply now and persist across reboots.
set -uo pipefail

# Fixed restore pair retained from the original script; not auto-detected.
readonly DEFAULT_SWAP=60 DEFAULT_COMPACT=20
readonly CONFIG_FILE=/etc/sysctl.d/simple-memory-set.conf
NOTICE=''

if [[ ! -t 0 || ! -t 1 ]]; then
    echo 'Run bash simple-memory-settings.sh in an interactive terminal (local or SSH).'
    exit 1
fi

if [[ ! -r /proc/sys/vm/swappiness || ! -r /proc/sys/vm/compaction_proactiveness ]]; then
    echo 'Run on a supported Linux system. Required kernel settings are missing.'
    exit 1
fi

GREEN=$'\033[1;32m'
RED=$'\033[1;31m'
CYAN=$'\033[1;36m'
RESET=$'\033[0m'
if [[ ${TERM:-dumb} == dumb || -n ${NO_COLOR:-} ]]; then
    GREEN='' RED='' CYAN='' RESET=''
fi

read_current() {
    CURRENT_SWAP=$(sysctl -n vm.swappiness) || return 1
    CURRENT_COMPACT=$(sysctl -n vm.compaction_proactiveness) || return 1
}

as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        sudo -- "$@"
    fi
}

save_values() {
    as_root bash -s -- "$CONFIG_FILE" "$1" "$2" <<'SAVE'
set -euo pipefail
config=$1
swap=$2
compact=$3
# Write beside the destination, then rename atomically.
[[ ! -L "$config" ]] || { echo 'Refusing to replace a symlink.' >&2; exit 1; }
mkdir -p -- "${config%/*}"
temp=$(mktemp "${config}.tmp.XXXXXX")
trap 'rm -f -- "$temp"' EXIT
printf '# Managed by simple-memory-settings.sh; loaded at boot.\nvm.swappiness=%s\nvm.compaction_proactiveness=%s\n' "$swap" "$compact" > "$temp"
chmod 0644 "$temp"
if [[ -e "$config" ]]; then
    [[ -f "$config" ]] || exit 1
fi
mv -f -- "$temp" "$config"
SAVE
}

show_saved() {
    local key value saved_swap='' saved_compact=''
    if [[ ! -r "$CONFIG_FILE" ]]; then
        printf '  Boot settings: no readable saved config from this tool.\n'
        return
    fi
    while IFS='=' read -r key value; do
        case "$key" in
            vm.swappiness) saved_swap=$value ;;
            vm.compaction_proactiveness) saved_compact=$value ;;
        esac
    done < "$CONFIG_FILE"
    printf '  Saved for reboot: %s / %s\n' "${saved_swap:-unknown}" "${saved_compact:-unknown}"
    if [[ $saved_swap != "$CURRENT_SWAP" || $saved_compact != "$CURRENT_COMPACT" ]]; then
        printf '  [NOTICE] Current values differ from the saved boot settings.\n'
    fi
}

apply_values() {
    local target_swap=$1 target_compact=$2 label=$3 action=${4:-save}
    local command_ok=0
    if ! valid_custom_value "$target_swap" 200 || ! valid_custom_value "$target_compact" 100; then
        NOTICE='[ERROR] Invalid values. Nothing applied or saved.'
        return
    fi
    printf '\nApplying %s now...\n' "$label"
    as_root sysctl -w "vm.swappiness=$target_swap" "vm.compaction_proactiveness=$target_compact" && command_ok=1
    if ! read_current; then
        NOTICE='[ERROR] Could not verify current values. Boot file unchanged; runtime changes may have occurred.'
        printf '%s%s%s\n' "$RED" "$NOTICE" "$RESET"
        return
    fi
    if (( command_ok )) && [[ $CURRENT_SWAP == "$target_swap" && $CURRENT_COMPACT == "$target_compact" ]]; then
        if [[ $action == restore ]]; then
            if as_root rm -f -- "$CONFIG_FILE"; then
                NOTICE='[RESTORED] 60 / 20 applied and verified. Saved config removed; existing boot settings resume next boot.'
                printf '\n%s%s%s\n' "$GREEN" "$NOTICE" "$RESET"
            else
                NOTICE='[DELETE FAILED] 60 / 20 is active now, but the saved config could not be removed. Retry option 3.'
                printf '\n%s%s%s\n' "$RED" "$NOTICE" "$RESET"
            fi
        elif save_values "$target_swap" "$target_compact"; then
            NOTICE="[SAVED] $label: $CURRENT_SWAP / $CURRENT_COMPACT. Applied now and saved for reboot."
            printf '\n%s%s%s\n' "$GREEN" "$NOTICE" "$RESET"
        else
            NOTICE='[SAVE FAILED] Runtime values changed, but boot settings were not updated. Retry to save.'
            printf '\n%s%s%s\n' "$RED" "$NOTICE" "$RESET"
        fi
    else
        NOTICE="[FAILED] Current: $CURRENT_SWAP / $CURRENT_COMPACT. Partial runtime change possible; boot file unchanged."
        printf '\n%s%s%s\n' "$RED" "$NOTICE" "$RESET"
    fi
}

pause_menu() {
    printf '\n'
    read -r -p 'Press Enter to return to the menu...' || exit 0
}

valid_custom_value() {
    [[ $1 =~ ^[0-9]{1,3}$ ]] && (( 10#$1 <= $2 ))
}

read_custom_value() {
    local name=$1 maximum=$2 input
    while true; do
        read -r -p "$name (0-$maximum, Enter: cancel) > " input || return 1
        [[ -n "$input" ]] || return 1
        if valid_custom_value "$input" "$maximum"; then
            REPLY=$((10#$input))
            return 0
        fi
        printf '%sEnter a whole number between 0 and %s.%s\n' "$RED" "$maximum" "$RESET"
    done
}

apply_custom() {
    local custom_swap custom_compact
    printf '\nCustom settings: both values are applied AND saved after the second entry.\n'
    if ! read_custom_value 'Swappiness' 200; then
        NOTICE='[CANCELED] No settings were changed.'
        return
    fi
    custom_swap=$REPLY
    if ! read_custom_value 'Compaction proactiveness' 100; then
        NOTICE='[CANCELED] No settings were changed.'
        return
    fi
    custom_compact=$REPLY
    apply_values "$custom_swap" "$custom_compact" 'custom settings'
    pause_menu
}


show_help() {
    cat <<'HELP'

====================== SETTINGS GUIDE ======================
SWAPPINESS (vm.swappiness, 0-200)
Balances swapping against reclaiming filesystem cache.
100 treats their I/O costs equally; higher assumes cheaper swap.
It is not a RAM percentage. 0 does not completely disable swap.
Values above 100 can suit zram or fast swap; workload matters.

COMPACTION (vm.compaction_proactiveness, 0-100)
Moves memory pages to assemble contiguous free blocks.
0 disables proactive work, not all on-demand compaction.
It does not compress memory or add RAM.

PRESETS AND MENU ACTIONS
1: 150 / 80 favors swapping and stronger background compaction.
   Benefit: may preserve cache and improve contiguous allocation.
   Cost: extra swap work and compaction latency; test your workload.
3: Restore 60 / 20 immediately and remove this tool's boot file.
   Benefit: less aggressive behavior than 150 / 80.
   Cost: less emphasis on cache retention and proactive compaction.
   These are not automatically detected device/boot defaults.
2: Custom values trade these benefits and costs; no universal best.
4: Refresh only. 5: Help only. 0: Exit without reverting settings.

PERSISTENCE
Options 1 and 2 apply now AND save for future boots.
Option 3 applies 60 / 20 now and deletes the boot file below.
After option 3, future boots use the system's existing configuration.
Deleting the file manually alone does not change live kernel values.
The last successfully saved pair replaces the previous saved pair.
No backup file is created.
Boot file: /etc/sysctl.d/simple-memory-set.conf
Other tuning tools or later boot services may override these values.
The menu shows current values separately from saved boot values.
This does not create swap or enable virtual GPU memory.

Reference: Linux kernel documentation
https://docs.kernel.org/admin-guide/sysctl/vm.html
HELP
    pause_menu
}

while true; do
    if [[ ${TERM:-dumb} != dumb ]]; then printf '\033[2J\033[H'; fi
    if ! read_current; then
        echo 'Could not read current kernel settings.'
        exit 1
    fi
    printf '%s==============================================%s\n' "$CYAN" "$RESET"
    printf '           Simple Memory Settings\n'
    printf '%s==============================================%s\n\n' "$CYAN" "$RESET"
    printf '  Swappiness            : %s%s%s\n' "$CYAN" "$CURRENT_SWAP" "$RESET"
    printf '  Compaction proactiveness: %s%s%s\n\n' "$CYAN" "$CURRENT_COMPACT" "$RESET"
    if [[ $CURRENT_SWAP == 150 && $CURRENT_COMPACT == 80 ]]; then
        printf '  %s[PRESET] 150 / 80 active%s\n' "$GREEN" "$RESET"
    elif [[ $CURRENT_SWAP == "$DEFAULT_SWAP" && $CURRENT_COMPACT == "$DEFAULT_COMPACT" ]]; then
        printf '  %s[RESTORE PAIR] 60 / 20 active%s\n' "$GREEN" "$RESET"
    else
        printf '  [CUSTOM] %s / %s active\n' "$CURRENT_SWAP" "$CURRENT_COMPACT"
    fi
    printf '  Fixed restore pair: 60 / 20 (from the original script)\n'
    show_saved
    cat <<'SUMMARY'

  SWAPPINESS: swap versus filesystem-cache reclaim preference.
    Higher: can retain cache; costs swap I/O or zram CPU time.
  COMPACTION: background preparation of contiguous free memory.
    Higher: helps large allocations; costs CPU and may cause stalls.
SUMMARY
    printf '\n----------------------------------------------\n'
    printf '  1) Apply and SAVE preset 150 / 80\n'
    printf '  2) Enter, apply and SAVE custom values\n'
    printf '  3) Restore 60 / 20 NOW and DELETE saved config\n'
    printf '  4) Refresh current status\n'
    printf '  5) Settings guide / benefits and trade-offs\n'
    printf '  0) Exit\n'
    printf '%s\n' '----------------------------------------------'
    printf '  1/2: Apply + save. 3: Restore now + delete config.\n'
    printf '  Changes take effect immediately; root/sudo required.\n'
    [[ -z "$NOTICE" ]] || printf '\n  %s\n' "$NOTICE"
    printf '\n'
    read -r -p 'Select > ' choice || exit 0
    case "$choice" in
        1)
            apply_values 150 80 'preset 150 / 80'
            pause_menu
            ;;
        2)
            apply_custom
            ;;
        3)
            apply_values "$DEFAULT_SWAP" "$DEFAULT_COMPACT" 'restore pair 60 / 20' restore
            pause_menu
            ;;
        4) ;;
        5) show_help ;;
        0) printf '\nMenu closed. Current settings remain active.\n'; exit 0 ;;
        *) NOTICE='Enter 0, 1, 2, 3, 4, or 5.' ;;
    esac
done
