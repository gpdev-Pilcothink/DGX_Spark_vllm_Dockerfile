# syntax=docker/dockerfile:1.7
# =============================================================================
# DGX Spark / ARM64 / SM121
# DeepSeek-V4-Flash-Vision-Exp + native DSpark + B12X Linear/MoE
# Consolidated 2026-09-08 from the user's working, hotfixed image configuration.
#
# Build: docker build -f Dockerfile_deepseek_v4_vision_dspark \
#          -t vllm_spark_dsv4:0.29-b12x-clean .
# B12X is ON by default; --build-arg ENABLE_B12X=0 keeps the optional non-B12X build.
# No model/tokenizer downloads, pip/uv caches, or BuildKit cache mounts.
# Docker's ordinary layer cache is NOT disabled.
#
# IMPORTANT: keep the Vision-capable VLLM_REF below. The 8284955 repair ref is
# only a donor for o_proj.py, NOT a replacement checkout for the entire vLLM tree.
# Its older tree lacks the Vision export and current graph-profiling safeguards.
# The Python-only repair is applied after wheel installation so existing heavy
# build layers can be reused. Both the wheel provenance and repair are recorded.
# No custom entrypoint: use the existing run_cluster_dual.sh --no-ray command.
# =============================================================================

# 1. Fixed versions and immutable source references
ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG BUILD_JOBS=8
ARG UV_VERSION=0.8.22
ARG RUSTUP_TOOLCHAIN=1.98.1
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG CUTLASS_DSL_VERSION=4.7.0
ARG TVM_FFI_VERSION=0.1.11
ARG TORCH_CUDA_ARCH_LIST=12.1a
ARG FLASHINFER_CUDA_ARCH_LIST=12.1a
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
ARG NCCL_REF=fd168324a3dc0c9080fd4881b6c7f4bb252a95a2
ARG EUGR_PATCH_REF=841fdcc4bde9f84c0abbed72c6df2d435401942e
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=6fbb00b18874e27ba7d7adc0a3b8e93fee763ab1
ARG VLLM_OPROJ_FIX_REF=8284955fe1d31f3aaddfc691161e348fb0fc39d4
ARG FLASHINFER_REF=27d5b029818e530a9fd0c6b3b356217b5bef1226
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ARG VLLM_SOURCE_MODE=remote
ARG VLLM_SOURCE_COMMIT=""
# Optional package only. This does NOT switch to a text-only B12X vLLM fork.
ARG ENABLE_B12X=1
ARG B12X_REF=06b4de7c723e6f166d65abf5909c5b7d0f8acc68
ARG PRE_TRANSFORMERS=0
# PRE_TRANSFORMERS=1 also requires an exact version, to avoid a moving pre-release.
ARG TRANSFORMERS_PRE_VERSION=""

# =============================================================================
# 2. Base system, fresh source helper, and SM121 NCCL
# =============================================================================
FROM scratch AS vllm_source

FROM ${BASE_IMAGE} AS os-base
ARG TARGETARCH
ARG BUILD_JOBS
ARG UV_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_CUDA_ARCH_LIST
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1 UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
    MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    CARGO_BUILD_JOBS=${BUILD_JOBS} NVCC_THREADS=1 \
    MAKEFLAGS=-j${BUILD_JOBS} NINJAFLAGS=-j${BUILD_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    DG_JIT_USE_NVRTC=0 USE_CUDNN=1 \
    BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 \
    VLLM_BASE_DIR=/workspace
WORKDIR /workspace
# CUDA devel, Python headers, C++/Ninja remain intentionally: runtime kernel JIT.
RUN test "${TARGETARCH}" = arm64 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-pip python3-dev \
        build-essential cmake ninja-build \
        libcudnn9-cuda-13 libibverbs1 ibverbs-providers rdma-core \
        libnuma1 libgomp1 libxcb1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install "uv==${UV_VERSION}"

FROM os-base AS build-tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git pkg-config libibverbs-dev libnuma-dev \
        devscripts debhelper fakeroot && \
    rm -rf /var/lib/apt/lists/*

COPY --chmod=0755 <<'PY' /workspace/build-tools/fresh_source.py
#!/usr/bin/env python3
"""Fresh, exact-commit source checkout. No mutable branches and no reused checkout."""
import argparse
import json
from pathlib import Path
import re
import subprocess

def run(*args, cwd=None, capture=False):
    if capture:
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    subprocess.run(args, cwd=cwd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('repo')
    p.add_argument('commit')
    p.add_argument('destination')
    a = p.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', a.commit):
        raise SystemExit('Source ref must be a full, lowercase 40-character commit SHA.')
    dest = Path(a.destination)
    if dest.exists():
        raise SystemExit(f'Refusing to reuse existing source checkout: {dest}')
    run('git', 'clone', '--no-checkout', '--no-hardlinks', a.repo, str(dest))
    run('git', 'checkout', '--detach', a.commit, cwd=dest)
    if run('git', 'rev-parse', 'HEAD', cwd=dest, capture=True) != a.commit:
        raise SystemExit('Source SHA mismatch')
    run('git', 'submodule', 'sync', '--recursive', cwd=dest)
    run('git', 'submodule', 'update', '--init', '--recursive', '--jobs', '4', cwd=dest)
    run('git', 'diff', '--exit-code', cwd=dest)
    print(json.dumps({'repo': a.repo, 'commit': a.commit, 'destination': str(dest)}))

if __name__ == '__main__':
    main()
PY

FROM build-tools AS nccl-builder
ARG NCCL_REF
ARG NCCL_NVCC_GENCODE
ARG BUILD_JOBS
RUN python3 /workspace/build-tools/fresh_source.py \
        https://github.com/NVIDIA/nccl.git "${NCCL_REF}" /workspace/nccl && \
    cd /workspace/nccl && \
    make -j "${BUILD_JOBS}" src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make -j "${BUILD_JOBS}" pkg.debian.build && \
    mkdir -p /workspace/nccl-pkg && \
    cp build/pkg/deb/*.deb /workspace/nccl-pkg/ && \
    git rev-parse HEAD > /workspace/nccl-pkg/source-commit && \
    printf '%s\n' "${NCCL_NVCC_GENCODE}" > /workspace/nccl-pkg/nvcc-gencode

# No source/compiler caches in this ancestor of the final image.
# =============================================================================
# 3. Shared PyTorch / CUDA / CUTLASS build environment
# =============================================================================
FROM os-base AS torch-base
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUTLASS_DSL_VERSION
ARG TVM_FFI_VERSION
ENV CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION} TVM_FFI_VERSION=${TVM_FFI_VERSION} \
    VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/libnccl.so.2
RUN --mount=type=bind,from=nccl-builder,source=/workspace/nccl-pkg,target=/workspace/install/nccl <<'SH'
apt-get update
apt-get install -y --no-install-recommends \
    --allow-downgrades --allow-change-held-packages \
    /workspace/install/nccl/libnccl2_*.deb /workspace/install/nccl/libnccl-dev_*.deb
rm -rf /var/lib/apt/lists/*
uv pip install --python /usr/bin/python3 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu130
python3 - <<'PY'
import importlib.metadata as m
import os
from pathlib import Path
import site
import torch
if torch.version.cuda is None or not torch.version.cuda.startswith('13.'):
    raise SystemExit('CPU/CUDA-12 PyTorch was resolved instead of CUDA-13 PyTorch')
for name, env in [('torch', 'TORCH_VERSION'), ('torchvision', 'TORCHVISION_VERSION'),
                  ('torchaudio', 'TORCHAUDIO_VERSION')]:
    if m.version(name).split('+')[0] != os.environ[env]:
        raise SystemExit(f'Unexpected {name}: {m.version(name)}')
system = Path('/usr/lib/aarch64-linux-gnu/libnccl.so.2')
if not system.is_file():
    raise SystemExit('Source-built SM121 NCCL was not installed')
libs = {Path(p) / 'nvidia/nccl/lib/libnccl.so.2' for p in site.getsitepackages()}
libs = [p for p in libs if p.exists()]
if len(libs) != 1:
    raise SystemExit(f'Unexpected Python NCCL layout: {libs}')
# Remove the redundant wheel-provided binary IN ITS INSTALL LAYER.
libs[0].unlink()
libs[0].symlink_to(system)
constraints = [f'{n}=={m.version(n)}' for n in ('torch','torchvision','torchaudio')]
constraints += [f'nvidia-cutlass-dsl[cu13]=={os.environ["CUTLASS_DSL_VERSION"]}',
                f'apache-tvm-ffi=={os.environ["TVM_FFI_VERSION"]}',
                'setuptools==80.9.0']
Path('/workspace/build-constraints.txt').write_text('\n'.join(constraints) + '\n')
PY
uv pip install --python /usr/bin/python3 \
    --constraint /workspace/build-constraints.txt \
    'setuptools==80.9.0' 'wheel==0.45.1' packaging build \
    "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
    "apache-tvm-ffi==${TVM_FFI_VERSION}" \
    nvidia-nvshmem-cu13 filelock requests tqdm
cp /workspace/install/nccl/source-commit /workspace/nccl-source-commit
cp /workspace/install/nccl/nvcc-gencode /workspace/nccl-nvcc-gencode
ldconfig
SH

FROM torch-base AS base
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git pkg-config libcudnn9-dev-cuda-13 libibverbs-dev libnuma-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build-tools /workspace/build-tools /workspace/build-tools

# =============================================================================
# 4. FlashInfer core + cubin + JIT-cache wheels
# =============================================================================
FROM base AS flashinfer-builder
ARG FLASHINFER_REF
RUN python3 /workspace/build-tools/fresh_source.py \
    https://github.com/flashinfer-ai/flashinfer.git "${FLASHINFER_REF}" /workspace/flashinfer
WORKDIR /workspace/flashinfer
# Build all three packages from the SAME source; no prebuilt vLLM/FlashInfer overlay.
# Setuptools 80 supports PEP-639; the old license-string rewrite is unnecessary.
RUN <<'SH'
# Source capability gate: both C4 and C128 secondary-cache dispatch arms.
# This is not a GPU execution test and does not patch upstream CUDA code.
python3 - <<'PY'
from pathlib import Path
src = Path('csrc/sparse_mla_sm120_prefill.cu').read_text()
required = (
    'inline bool dispatch_dsv4_dual(',
    'if (page_block_size != 64) return false;',
    'extra_page_block_size == 64', 'extra_page_block_size == 2',
    'DISPATCH_BY_NH_PBSX(64)', 'DISPATCH_BY_NH_PBSX(2)',
    'DISPATCH_DUAL_MG_CM(BF16, 32, PBSX, 2)',
)
missing = [needle for needle in required if needle not in src]
if missing:
    raise SystemExit(f'Missing audited DSV4 dual-cache dispatch source: {missing}')
print('Source gate only: DSV4 TP2 dual-cache PBS 64/64 and 64/2 present')
PY
mkdir -p /workspace/wheels
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
uv pip install --python /usr/bin/python3 \
    --constraint /workspace/build-constraints.txt \
    --override /workspace/build-constraints.txt -r requirements.txt
uv build --python /usr/bin/python3 --no-build-isolation --wheel \
    --out-dir /workspace/wheels .
uv build --python /usr/bin/python3 --no-build-isolation --wheel \
    --out-dir /workspace/wheels ./flashinfer-cubin
uv pip install --python /usr/bin/python3 \
    --constraint /workspace/build-constraints.txt \
    /workspace/wheels/flashinfer_python-*.whl /workspace/wheels/flashinfer_cubin-*.whl
uv build --python /usr/bin/python3 --no-build-isolation --wheel \
    --out-dir /workspace/wheels ./flashinfer-jit-cache
python3 - <<'PY'
import hashlib, json, os, subprocess, zipfile
from pathlib import Path
def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
w = Path('/workspace/wheels')
for prefix in ('flashinfer_python-', 'flashinfer_cubin-', 'flashinfer_jit_cache-'):
    paths = list(w.glob(prefix + '*.whl'))
    if len(paths) != 1:
        raise SystemExit(f'Expected one {prefix} wheel: {paths}')
    with zipfile.ZipFile(paths[0]) as z:
        if prefix == 'flashinfer_python-':
            names = z.namelist()
            required = ['flashinfer/mla/_sparse_mla_sm120.py',
                        'flashinfer/mla/_sparse_mla_sm120_plan.py',
                        'flashinfer/jit/mla.py']
            for name in required:
                if name not in names:
                    raise SystemExit(f'Missing recent SM12x source in wheel: {name}')
        if prefix == 'flashinfer_jit_cache-' and not any(n.endswith('.so') for n in z.namelist()):
            raise SystemExit('Empty FlashInfer JIT wheel')
manifest = {
    'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    'arch': os.environ['FLASHINFER_CUDA_ARCH_LIST'],
    'source_ref': os.environ['FLASHINFER_REF'],
    'dual_cache_dispatch_source_checked': ['heads32 pbs64 extra_pbs64', 'heads32 pbs64 extra_pbs2'],
    'dual_cache_numerical_execution_tested': False,
    'wheels': {p.name: sha256_file(p) for p in w.glob('*.whl')},
}
(w/'flashinfer-build.json').write_text(json.dumps(manifest, indent=2))
PY
# Temporary generated cache is not carried to any export/runner stage.
rm -rf /root/.cache /workspace/flashinfer/build
SH
FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /

# =============================================================================
# 5. B12X wheel and all five CUTLASS dependency pins
# =============================================================================
FROM base AS b12x-builder
ARG ENABLE_B12X
ARG B12X_REF
ARG CUTLASS_DSL_VERSION
RUN <<'SH'
mkdir -p /workspace/wheels
case "${ENABLE_B12X}" in
  0) printf '{"enabled": false}\n' > /workspace/wheels/b12x-build.json ;;
  1)
    python3 /workspace/build-tools/fresh_source.py \
        https://github.com/local-inference-lab/b12x.git "${B12X_REF}" /workspace/b12x
    cd /workspace/b12x
    python3 - <<'PY'
import os
from pathlib import Path
p = Path("pyproject.toml")
s = p.read_text()
old = "==4.6.2"
count = s.count(old)
if count != 5:
    raise SystemExit(f"Expected exactly 5 B12X CUTLASS 4.6.2 pins, found {count}")
s = s.replace(old, f"=={os.environ['CUTLASS_DSL_VERSION']}")
p.write_text(s)
PY
    export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
    uv build --python /usr/bin/python3 --no-build-isolation --wheel \
        --out-dir /workspace/wheels .
    printf '{"enabled": true, "commit": "%s"}\n' "${B12X_REF}" > /workspace/wheels/b12x-build.json
    ;;
  *) echo 'ENABLE_B12X must be 0 or 1' >&2; exit 1 ;;
esac
SH
FROM scratch AS b12x-export
COPY --from=b12x-builder /workspace/wheels /

# =============================================================================
# 6. Vision-capable vLLM + native DSpark + Spark-only patches
# =============================================================================
FROM base AS vllm-builder
ARG RUSTUP_TOOLCHAIN
ARG VLLM_REPO
ARG VLLM_REF
ARG VLLM_SOURCE_MODE
ARG VLLM_SOURCE_COMMIT
ARG DEEPGEMM_REF
ARG EUGR_PATCH_REF
ENV RUSTUP_HOME=/workspace/.rustup CARGO_HOME=/workspace/.cargo \
    CARGO_TARGET_DIR=/workspace/vllm/target \
    PATH=/workspace/.cargo/bin:${PATH} \
    PROTOC_INCLUDE=/usr/include \
    DEEPGEMM_SRC_DIR=/workspace/DeepGEMM \
    VLLM_PRESERVE_SM12X_TARGET=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 https://sh.rustup.rs \
        -o /workspace/rustup-init.sh && \
    sh /workspace/rustup-init.sh -y --profile minimal \
        --default-toolchain "${RUSTUP_TOOLCHAIN}" --no-modify-path && \
    rm /workspace/rustup-init.sh && rustc --version && cargo --version
RUN --mount=type=bind,from=vllm_source,target=/workspace/local-source <<'SH'
case "${VLLM_SOURCE_MODE}" in
  remote)
    python3 /workspace/build-tools/fresh_source.py "${VLLM_REPO}" "${VLLM_REF}" /workspace/vllm
    ;;
  local)
    test -d /workspace/local-source/.git
    test -n "${VLLM_SOURCE_COMMIT}"
    test "${VLLM_SOURCE_COMMIT}" = "${VLLM_REF}"
    git -c safe.directory=/workspace/local-source -C /workspace/local-source diff --exit-code
    test "$(git -c safe.directory=/workspace/local-source -C /workspace/local-source rev-parse HEAD)" = "${VLLM_REF}"
    python3 /workspace/build-tools/fresh_source.py /workspace/local-source "${VLLM_REF}" /workspace/vllm
    ;;
  *) echo 'VLLM_SOURCE_MODE must be remote or local' >&2; exit 1 ;;
esac
python3 /workspace/build-tools/fresh_source.py \
    https://github.com/deepseek-ai/DeepGEMM.git "${DEEPGEMM_REF}" /workspace/DeepGEMM
SH

COPY --chmod=0755 <<'PY' /workspace/build-tools/vision_dspark_source_gate.py
#!/usr/bin/env python3
"""Fail before compilation unless native Vision + DSpark + SM12x paths are present."""
import ast
from pathlib import Path

root = Path('/workspace/vllm')
required = {
    'vllm/models/deepseek_v4/nvidia/vl_model.py': [
        'DeepseekV4ForConditionalGeneration', 'SupportsMultiModal'
    ],
    'vllm/models/deepseek_v4/common/mm_preprocess.py': [
        'DeepseekV4VLMultiModalProcessor'
    ],
    'vllm/models/deepseek_v4/common/vision.py': [],
    'vllm/models/deepseek_v4/__init__.py': [
        'DSparkDeepseekV4ForCausalLM',
        'DeepseekV4ForConditionalGeneration',
        '.nvidia.dspark',
        '.nvidia.vl_model',
    ],
    'vllm/models/deepseek_v4/nvidia/model.py': [
        'DeepseekV4FlashInferSM120Attention', 'bias_vl'
    ],
    'vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py': [
        'DeepseekV4FlashInferSM120Attention', 'prefill_swa_indices'
    ],
    'vllm/models/deepseek_v4/nvidia/dspark.py': [
        'class DSparkDeepseekV4Model',
        'class DSparkDeepseekV4ForCausalLM',
        'num_dspark_layers',
        'confidence_head',
        'mtp.',
    ],
    'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py': [
        'class DSparkSpeculator',
        'load_dspark_model',
        'num_speculative_steps',
        'enable_adaptive_verification',
    ],
    'vllm/v1/worker/gpu/spec_decode/dspark/utils.py': [
        'def load_dspark_model',
        '_resolve_dspark_attention_backend',
        'draft_vllm_config',
    ],
    'vllm/v1/attention/backends/mla/sparse_swa.py': [
        'prefill_left_visible', 'prefill_right_visible', 'vision_max_n_token'
    ],
    'vllm/tokenizers/deepseek_v4.py': [],
}
for name, needles in required.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f'Missing required DeepSeek V4 Vision/DSpark source: {name}')
    text = path.read_text()
    ast.parse(text, filename=str(path))
    for needle in needles:
        if needle not in text:
            raise SystemExit(f'Required feature missing: {name}: {needle}')

registry = root / 'vllm/model_executor/models/registry.py'
rtext = registry.read_text()
# DSpark draft models are selected by speculative decoding machinery rather than
# the normal ModelRegistry.  Require the normal target + Vision registrations here;
# DSpark itself is gated above by its concrete implementation/import paths.
for needle in ('DeepseekV4ForCausalLM', 'DeepseekV4ForConditionalGeneration'):
    if needle not in rtext:
        raise SystemExit(f'Missing DeepSeek V4 registry entry: {needle}')

# Native DSV4 DSpark must use the target checkpoint's own folded MTP weights.
dspark = (root / 'vllm/models/deepseek_v4/nvidia/dspark.py').read_text()
if 'load_weights' not in dspark or '_remap_dspark_name' not in dspark:
    raise SystemExit('Native folded DSpark weight loader is incomplete.')

print('Source Vision + native DSpark + SM12x sparse attention gates: PASS')
PY

COPY --chmod=0755 <<'PY' /workspace/build-tools/apply_spark_dspark_patches.py
#!/usr/bin/env python3
"""Apply only pinned eugr patches directly relevant to SM121/Spark/DSpark."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/workspace/vllm')
patch_dir = Path('/workspace/patches')
patches = [
    'patch_vllm_preserve_sm12x_target.py',
    'patch_vllm_spark_kv_cache_cleanup.py',
]

if subprocess.check_output(['git', 'diff', '--name-only'], cwd=root, text=True).strip():
    raise SystemExit('Spark patches must start from the clean pinned vLLM source.')

records = []
for name in patches:
    path = patch_dir / name
    data = path.read_bytes()
    if not data:
        raise SystemExit(f'Empty patch: {name}')
    compile(data, str(path), 'exec')
    subprocess.run([sys.executable, str(path), str(root)], check=True)
    records.append({'name': name, 'sha256': hashlib.sha256(data).hexdigest()})

# 1) SM121 build target must survive vLLM's CUDA allow-list.
cmake = (root / 'CMakeLists.txt').read_text()
if '12.1' not in cmake:
    raise SystemExit('SM121 is absent from vLLM CUDA_SUPPORTED_ARCHS after patching.')

# 2) MRV2 DSpark/DFlash graph managers must not leave profiling graphs in the
# persistent allocator pool on Spark UMA.
cg_path = root / 'vllm/v1/worker/gpu/cudagraph_utils.py'
cg = cg_path.read_text()
# This pinned vLLM already contains a broader upstream fix than eugr's older
# MRV2 patch: the *global* graph pool is redirected to a throwaway pool during
# profiling, and speculator CudaGraphManagers are discarded before real KV init.
# Fail closed if that native protection disappears in a future source ref.
cg_upstream = all(x in cg for x in (
    'throwaway_pool = current_platform.graph_pool_handle()',
    'platform_cls._global_graph_pool = throwaway_pool',
    'spec_manager_names: list[str] = []',
    'if isinstance(value, CudaGraphManager)',
    'setattr(speculator, name, None)',
    'platform_cls._global_graph_pool = saved_global_pool',
))
if not cg_upstream:
    raise SystemExit('Upstream MRV2/DSpark CUDA-graph profiling-pool isolation is missing.')
compile(cg, str(cg_path), 'exec')

# 3) Hopper cooperative top-k must never be selected merely because SM121 >= 90.
topk_path = root / 'vllm/model_executor/layers/sparse_attn_indexer.py'
topk = topk_path.read_text()
if 'use_cooperative_topk' in topk:
    # Upstream now explicitly excludes the full SM12x family.  Accept either
    # that guard or eugr's older exact-SM90 selector, but never a bare >=SM90 gate.
    safe_sm12x_exclusion = 'not current_platform.is_device_capability_family(120)' in topk
    safe_exact_sm90 = 'device_capability.to_int() == 90' in topk
    if not (safe_sm12x_exclusion or safe_exact_sm90):
        raise SystemExit('Unsafe Hopper cooperative sparse top-k selector remains on SM121.')
compile(topk, str(topk_path), 'exec')

# 4) Spark UMA allocations must be reclaimed before KV sizing and allocation.
worker_path = root / 'vllm/v1/worker/gpu_worker.py'
worker = worker_path.read_text()
post_ok = (
    'profile_result.after_profile.measure()' in worker
    and 'diff_from_create.non_torch_memory' in worker
)
pre_ok = all(x in worker for x in (
    'memory_reserved(self.device)',
    'memory_allocated(self.device)',
    'torch.cuda.empty_cache()',
))
if not (post_ok and pre_ok):
    raise SystemExit('Spark KV-cache cleanup/accounting postconditions are incomplete.')
compile(worker, str(worker_path), 'exec')

allowed = {
    'CMakeLists.txt',
    'vllm/v1/worker/gpu_worker.py',
}
changed = set(subprocess.check_output(
    ['git', 'diff', '--name-only'], cwd=root, text=True
).splitlines())
unexpected = changed - allowed
if unexpected:
    raise SystemExit(f'Unexpected eugr patch scope: {sorted(unexpected)}')

# Both remaining eugr scripts must be idempotent against the selected vLLM ref.
first = subprocess.check_output(['git', 'diff', '--binary'], cwd=root)
for rec in records:
    subprocess.run([sys.executable, str(patch_dir / rec['name']), str(root)], check=True)
second = subprocess.check_output(['git', 'diff', '--binary'], cwd=root)
if first != second:
    raise SystemExit('Spark/DSpark patch reapplication is not idempotent.')

(root.parent / 'spark-dspark-patches.json').write_text(json.dumps(records, indent=2))
print('Pinned SM121 + Spark + MRV2/DSpark patch gates: PASS')
PY

COPY --chmod=0755 <<'PY' /workspace/build-tools/align_dependencies.py
#!/usr/bin/env python3
import email
import json
import os
from pathlib import Path
import re
import zipfile

root = Path('/workspace/vllm')
req = root / 'requirements/cuda.txt'
text = req.read_text()
records = []
def replace(pattern, replacement):
    global text
    if len(re.findall(pattern, text, re.MULTILINE)) != 1:
        raise SystemExit(f'Expected exactly one dependency declaration: {pattern}')
    text = re.sub(pattern, lambda _: replacement, text, flags=re.MULTILINE)
    records.append(replacement)

def wheel_version(prefix):
    wheels = list(Path('/workspace/install/flashinfer').glob(prefix + '-*.whl'))
    if len(wheels) != 1:
        raise SystemExit(f'Expected exactly one wheel for {prefix}, got {wheels}')
    with zipfile.ZipFile(wheels[0]) as z:
        name = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
        return email.message_from_bytes(z.read(name))['Version']

replace(r'^flashinfer-python\s*==\s*0\.6\.18\s*$', 'flashinfer-python==' + wheel_version('flashinfer_python'))
replace(r'^flashinfer-cubin\s*==\s*0\.6\.18\s*$', 'flashinfer-cubin==' + wheel_version('flashinfer_cubin'))
replace(r'^apache-tvm-ffi\s*==\s*0\.1\.11\s*$', 'apache-tvm-ffi==' + os.environ['TVM_FFI_VERSION'])
replace(r'^nvidia-cutlass-dsl\[cu13\]\s*==\s*4\.6\.2\s*$', 'nvidia-cutlass-dsl[cu13]==' + os.environ['CUTLASS_DSL_VERSION'])
req.write_text(text)
(root.parent / 'dependency-alignment.json').write_text(json.dumps(records, indent=2))
PY

WORKDIR /workspace/vllm
# Only eugr patches still missing from the pinned upstream and tied directly to DGX Spark/SM121.
# Native DSpark CUDAGraph isolation and SM12x cooperative-topk exclusion are already upstream;
# they are fail-closed gated above instead of being redundantly patched.
# Qwen/Gemma/MiniMax/diffusion/AutoGPTQ/B12X-fork-only patches are absent.
# vLLM PR #54788 is intentionally NOT carried: it fixes the EAGLE/MTP
# load_eagle_model path, while native DSV4 DSpark uses load_dspark_model.
RUN <<'SH'
python3 /workspace/build-tools/vision_dspark_source_gate.py
[[ "${EUGR_PATCH_REF}" =~ ^[0-9a-f]{40}$ ]]
mkdir -p /workspace/patches
for name in \
    patch_vllm_preserve_sm12x_target.py \
    patch_vllm_spark_kv_cache_cleanup.py
do
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-all-errors \
        "https://raw.githubusercontent.com/eugr/spark-vllm-docker/${EUGR_PATCH_REF}/docker/${name}" \
        -o "/workspace/patches/${name}"
    test -s "/workspace/patches/${name}"
done
python3 /workspace/build-tools/apply_spark_dspark_patches.py
SH
RUN --mount=type=bind,from=flashinfer-export,source=/,target=/workspace/install/flashinfer <<'SH'
python3 /workspace/build-tools/align_dependencies.py
python3 use_existing_torch.py
uv pip install --python /usr/bin/python3 \
    --constraint /workspace/build-constraints.txt \
    --override /workspace/build-constraints.txt \
    -r requirements/build/cuda.txt 'setuptools-rust>=1.9.0' \
    /workspace/install/flashinfer/*.whl
SH
# Cargo caches and targets are removed IN THE BUILD RUN, not in a later layer.
RUN <<'SH'
mkdir -p /workspace/wheels
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
VLLM_REQUIRE_RUST_FRONTEND=1 uv build --python /usr/bin/python3 \
    --no-build-isolation --wheel --out-dir /workspace/wheels .
python3 - <<'PY'
import email, hashlib, json, os, subprocess, zipfile
from pathlib import Path
def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
root = Path('/workspace/vllm')
wheels = list(Path('/workspace/wheels').glob('vllm-*.whl'))
if len(wheels) != 1:
    raise SystemExit(f'Expected one vLLM wheel: {wheels}')
with zipfile.ZipFile(wheels[0]) as z:
    for rel in ['vllm/models/deepseek_v4/nvidia/vl_model.py',
                'vllm/models/deepseek_v4/common/vision.py',
                'vllm/models/deepseek_v4/common/mm_preprocess.py',
                'vllm/models/deepseek_v4/nvidia/dspark.py',
                'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py',
                'vllm/v1/worker/gpu/spec_decode/dspark/utils.py',
                'vllm/tokenizers/deepseek_v4.py']:
        if rel not in z.namelist():
            raise SystemExit(f'Wheel omitted required Vision file: {rel}')
        compile(z.read(rel), rel, 'exec')
    meta = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
    version = email.message_from_bytes(z.read(meta))['Version']
commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
if commit != os.environ['VLLM_REF']:
    raise SystemExit('Requested vLLM source HEAD was replaced')
subprocess.run(['git','merge-base','--is-ancestor',os.environ['VLLM_REF'],'HEAD'], check=True)
manifest = {
    'repo': os.environ['VLLM_REPO'], 'source_ref': os.environ['VLLM_REF'],
    'commit': commit, 'version_from_wheel': version, 'version_overridden': False,
    'deepgemm_commit': subprocess.check_output(['git','-C','/workspace/DeepGEMM','rev-parse','HEAD'], text=True).strip(),
    'arch': os.environ['TORCH_CUDA_ARCH_LIST'], 'eugr_patch_ref': os.environ['EUGR_PATCH_REF'],
    'runtime_patches': json.loads(Path('/workspace/spark-dspark-patches.json').read_text()),
    'dependency_alignment': json.loads(Path('/workspace/dependency-alignment.json').read_text()),
    'additional_prs': [],
    'native_dspark_required': True,
    'dspark_default_num_speculative_tokens': 3,
    'dspark_default_adaptive_verification': False,
    'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff','--binary'])).hexdigest(),
    'wheel_sha256': sha256_file(wheels[0]),
    'rust_toolchain': subprocess.check_output(['rustc','--version'], text=True).strip(),
    'vision_dspark_source_gate': 'passed', 'vision_dspark_wheel_files': 'passed',
    'docker_runtime_and_model_inference': 'not_tested_during_build',
}
Path('/workspace/wheels/vllm-build.json').write_text(json.dumps(manifest, indent=2))
Path('/workspace/wheels/vllm-source-changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary']))
PY
rm -rf /workspace/.cargo/registry /workspace/.cargo/git /workspace/vllm/target \
       /workspace/vllm/rust/target /root/.cache
SH
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =============================================================================
# 7. Runtime wheel installation and dependency/provenance checks
# =============================================================================
FROM torch-base AS runner
# Container defaults shared by both nodes; no Ray-only launch wrapper is required.
ENV VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    DG_JIT_USE_NVRTC=0 \
    CUTE_DSL_ARCH=sm_121a
ARG BASE_IMAGE
ARG BUILD_JOBS
ARG EUGR_PATCH_REF
ARG PRE_TRANSFORMERS
ARG TRANSFORMERS_PRE_VERSION
ARG ENABLE_B12X
ARG NCCL_REF
# No repository, Rust toolchain, local wheel cache, or model snapshot copied here.
RUN --mount=type=bind,from=flashinfer-export,source=/,target=/workspace/install/flashinfer \
    --mount=type=bind,from=vllm-export,source=/,target=/workspace/install/vllm \
    --mount=type=bind,from=b12x-export,source=/,target=/workspace/install/b12x <<'SH'
cp /workspace/build-constraints.txt /workspace/runtime-overrides.txt
# quack-kernels 0.6.4 pins CUTLASS 4.6.2 while this eugr/FlashInfer stack uses 4.7.0.
# This narrow override is recorded, not a blanket --no-deps installation.
case "${PRE_TRANSFORMERS}" in
  0) ;;
  1)
    test -n "${TRANSFORMERS_PRE_VERSION}"
    python3 - <<'PY'
import os
from packaging.version import Version
v = Version(os.environ['TRANSFORMERS_PRE_VERSION'])
if not v.is_prerelease or v < Version('5.10.4'):
    raise SystemExit('An exact Transformers pre-release >= 5.10.4 is required')
PY
    printf 'transformers==%s\n' "${TRANSFORMERS_PRE_VERSION}" >> /workspace/runtime-overrides.txt
    ;;
  *) echo 'PRE_TRANSFORMERS must be 0 or 1' >&2; exit 1 ;;
esac
shopt -s nullglob
wheels=(/workspace/install/flashinfer/*.whl /workspace/install/vllm/*.whl)
b12x_wheels=(/workspace/install/b12x/*.whl)
if [ "${ENABLE_B12X}" = 1 ]; then
    test "${#b12x_wheels[@]}" = 1
    wheels+=("${b12x_wheels[@]}")
fi
uv pip install --python /usr/bin/python3 \
    --constraint /workspace/build-constraints.txt \
    --override /workspace/runtime-overrides.txt \
    "${wheels[@]}" 'ray[default]'
mkdir -p /workspace/provenance
cp /workspace/install/flashinfer/flashinfer-build.json /workspace/provenance/
cp /workspace/install/vllm/vllm-build.json /workspace/provenance/
cp /workspace/install/vllm/vllm-source-changes.patch /workspace/provenance/
cp /workspace/install/b12x/b12x-build.json /workspace/provenance/
# Verify only the acknowledged CUTLASS pin exception, not arbitrary conflicts.
python3 - <<'PY'
import importlib.metadata as m
import json, os, platform, site, subprocess
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import torch
import yaml
errors, allowed = [], []
# Check the distribution Python actually selects, not shadowed Ubuntu metadata.
names = {canonicalize_name(d.metadata['Name']) for d in m.distributions() if d.metadata.get('Name')}
for parent in sorted(names):
    dist = m.distribution(parent)
    for value in dist.requires or []:
        r = Requirement(value)
        if r.marker and not r.marker.evaluate({'extra': ''}):
            continue
        try:
            installed = m.version(r.name)
        except m.PackageNotFoundError:
            errors.append(f'{parent}: missing {r}')
            continue
        if r.specifier and Version(installed) not in r.specifier:
            row = f'{parent}: {r}; installed {installed}'
            if canonicalize_name(r.name) == 'nvidia-cutlass-dsl' and parent in {'quack-kernels','b12x'}:
                allowed.append(row)
            else:
                errors.append(row)
if errors:
    raise SystemExit('Unexpected dependency conflicts:\n' + '\n'.join(errors))
lib = Path('/usr/lib/aarch64-linux-gnu/libnccl.so.2')
links = [Path(p)/'nvidia/nccl/lib/libnccl.so.2' for p in site.getsitepackages()]
links = [p for p in links if p.exists()]
if len(links) != 1 or not links[0].is_symlink() or links[0].resolve() != lib.resolve():
    raise SystemExit('Python NCCL symlink changed during dependency resolution')
if not torch.version.cuda.startswith('13.'):
    raise SystemExit('CUDA PyTorch was replaced during dependency resolution')
meta = {
    'target_model': 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
    'recommended_model_revision': '6821d6ad3681a4b137b066b76094fa82ebd0a380',
    'bundle_revision': 'v3-audit-20260907',
    'base_image': os.environ['BASE_IMAGE'], 'machine': platform.machine(),
    'eugr_policy_baseline': '574408a81cf1c9ed61434b687b195c75f108f81d',
    'eugr_reference': os.environ['EUGR_PATCH_REF'],
    'nccl_commit': Path('/workspace/nccl-source-commit').read_text().strip(),
    'nccl_gencode': Path('/workspace/nccl-nvcc-gencode').read_text().strip(),
    'build_jobs': int(os.environ['BUILD_JOBS']),
    'torch_arch': os.environ['TORCH_CUDA_ARCH_LIST'],
    'flashinfer_arch': os.environ['FLASHINFER_CUDA_ARCH_LIST'],
    'vllm': json.loads(Path('/workspace/provenance/vllm-build.json').read_text()),
    'flashinfer': json.loads(Path('/workspace/provenance/flashinfer-build.json').read_text()),
    'b12x': json.loads(Path('/workspace/provenance/b12x-build.json').read_text()),
    'packages': {n: m.version(n) for n in ('vllm','torch','torchvision','torchaudio','transformers',
                 'flashinfer-python','flashinfer-cubin','flashinfer-jit-cache',
                 'nvidia-cutlass-dsl','apache-tvm-ffi','triton','ray')},
    'acknowledged_dependency_exceptions': allowed,
    'unexpected_dependency_conflicts': [],
    'model_or_tokenizer_download_during_build': False,
    'gpu_kernel_execution_during_build': False,
    'model_inference_during_build': False,
    'source_pins_are_not_a_complete_transitive_lock': True,
}
Path('/workspace/build-metadata.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
Path('/workspace/provenance/python-packages.txt').write_text(
    subprocess.check_output(['uv','pip','freeze','--python','/usr/bin/python3'], text=True))
Path('/workspace/provenance/system-packages.txt').write_text(
    subprocess.check_output(['dpkg-query','-W'], text=True))
print(yaml.safe_dump(meta, sort_keys=False))
PY
# BuildKit bind mounts disappear. There are no stored wheel copies to delete.
SH

# =============================================================================
# 8. Consolidate the working runtime defaults and Python-only o_proj repair
# =============================================================================
# Added after the expensive wheel install to preserve its existing cache key.
# docker run -e still overrides ENV: keep AOT=0 / BREAKABLE=1 in the host command.
ENV VLLM_USE_AOT_COMPILE=0
ARG VLLM_REF
ARG VLLM_OPROJ_FIX_REF
RUN <<'SH'
[[ "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]]
[[ "${VLLM_OPROJ_FIX_REF}" =~ ^[0-9a-f]{40}$ ]]
mkdir -p /workspace/.oproj-fix
trap 'rm -rf /workspace/.oproj-fix' EXIT
curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-all-errors \
    "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_REF}/vllm/models/deepseek_v4/nvidia/ops/o_proj.py" \
    -o /workspace/.oproj-fix/base.py
curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-all-errors \
    "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_OPROJ_FIX_REF}/vllm/models/deepseek_v4/nvidia/ops/o_proj.py" \
    -o /workspace/.oproj-fix/fixed.py
test -s /workspace/.oproj-fix/base.py
test -s /workspace/.oproj-fix/fixed.py
python3 - <<'PY'
import ast
import base64
import csv
import difflib
import hashlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import yaml

rel = "vllm/models/deepseek_v4/nvidia/ops/o_proj.py"
dist = md.distribution("vllm")
path = Path(dist.locate_file(rel))
original = path.read_bytes()
base = Path("/workspace/.oproj-fix/base.py").read_bytes()
fixed = Path("/workspace/.oproj-fix/fixed.py").read_bytes()

# Check exact program structure, allowing only comments/docstrings to differ.
class WithoutDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

def tree(data):
    compile(data, rel, "exec")
    return WithoutDocstrings().visit(ast.parse(data))

def signature(node):
    return ast.dump(node, include_attributes=False)

expected = tree(base)
recipe = [n for n in expected.body if isinstance(n, ast.FunctionDef)
          and n.name == "compute_fp8_einsum_recipe"]
project = [n for n in expected.body if isinstance(n, ast.FunctionDef)
           and n.name == "deep_gemm_fp8_o_proj"]
if len(recipe) != 1 or len(project) != 1:
    raise SystemExit("Unexpected o_proj function layout; refusing repair")
recipe = recipe[0]
spots = [i for i, n in enumerate(recipe.body) if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "einsum_recipe" for t in n.targets)]
if len(spots) != 1:
    raise SystemExit("Unexpected FP8 recipe declaration; refusing repair")
recipe.body[spots[0]:spots[0]] = ast.parse(
    "if cap.major == 12:\n    return (1, 128, 128), True\n").body
project = project[0]
spots = [i for i, n in enumerate(project.body) if isinstance(n, ast.Expr)
         and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
         and n.value.func.id == "fp8_einsum"]
if len(spots) != 1:
    raise SystemExit("Unexpected fp8_einsum call; refusing repair")
pos = spots[0]
call = project.body[pos].value
if (len(call.args) != 4 or signature(call.args[2]) !=
        signature(ast.parse("(wo_a.weight, weight_scale)", mode="eval").body)):
    raise SystemExit("Unexpected FP8 weight operand; refusing repair")
call.args[2] = ast.parse("(weight, weight_scale)", mode="eval").body
project.body[pos:pos] = ast.parse(
    "weight = wo_a.weight\n"
    "if weight.ndim == 2:\n"
    "    weight = weight.view(n_groups, o_lora_rank, -1)\n"
    "    weight_scale = weight_scale.view(n_groups, o_lora_rank // 128, -1)\n"
).body
if signature(expected) != signature(tree(fixed)):
    raise SystemExit("Donor source is not exactly the two audited o_proj fixes")
if signature(tree(original)) not in {signature(tree(base)), signature(tree(fixed))}:
    raise SystemExit("Installed o_proj has unrelated changes; refusing overwrite")

metadata_path = Path("/workspace/build-metadata.yaml")
metadata = yaml.safe_load(metadata_path.read_text())
if metadata["vllm"]["commit"] != os.environ["VLLM_REF"]:
    raise SystemExit("Installed vLLM provenance does not match VLLM_REF")
record_files = [f for f in (dist.files or []) if str(f).endswith(".dist-info/RECORD")]
if len(record_files) != 1:
    raise SystemExit("Cannot identify the installed vLLM RECORD")
record_path = Path(dist.locate_file(record_files[0]))
rows = list(csv.reader(io.StringIO(record_path.read_text())))
entries = [row for row in rows if row and row[0] == rel]
if len(entries) != 1:
    raise SystemExit("Cannot identify o_proj.py in the installed vLLM RECORD")
digest = hashlib.sha256(fixed).digest()
entries[0][1:] = ["sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="), str(len(fixed))]
record_stream = io.StringIO(newline="")
csv.writer(record_stream).writerows(rows)
repair = {
    "kind": "installed_python_source_only",
    "file": rel,
    "base_ref": os.environ["VLLM_REF"],
    "donor_ref": os.environ["VLLM_OPROJ_FIX_REF"],
    "source_repository": "https://github.com/vllm-project/vllm",
    "base_source_sha256": hashlib.sha256(base).hexdigest(),
    "installed_before_sha256": hashlib.sha256(original).hexdigest(),
    "installed_after_sha256": digest.hex(),
    "scope": ["SM12x recipe + packed scales", "flat wo_a weight/scale 3D views"],
    "compiled_extensions_changed": False,
    "gpu_inference_tested_during_build": False,
}
# Reapplying the same repair must not rewrite its original before-hash.
for previous in metadata.get("installed_python_source_patches", []):
    if (original == fixed and previous.get("file") == rel
            and previous.get("base_ref") == repair["base_ref"]
            and previous.get("donor_ref") == repair["donor_ref"]
            and previous.get("installed_after_sha256") == repair["installed_after_sha256"]):
        repair["installed_before_sha256"] = previous["installed_before_sha256"]
metadata["bundle_revision"] = "2026-09-08-consolidated"
metadata["installed_python_source_patches"] = [repair]
metadata["runtime_defaults"] = {key: os.environ[key] for key in (
    "VLLM_USE_AOT_COMPILE", "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "VLLM_USE_V2_MODEL_RUNNER", "DG_JIT_USE_NVRTC", "CUTE_DSL_ARCH")}
# The original wheel hashes continue to describe the builder wheels. The
# installed-file change is tracked separately, not misrepresented as a new wheel.
path.write_bytes(fixed)
record_path.write_text(record_stream.getvalue())
for pyc in (path.parent / "__pycache__").glob("o_proj.*.pyc"):
    pyc.unlink()
provenance = Path("/workspace/provenance")
(provenance / "dsv4-oproj-repair.json").write_text(json.dumps(repair, indent=2) + "\n")
(provenance / "dsv4-oproj-repair.patch").write_text("".join(difflib.unified_diff(
    base.decode().splitlines(keepends=True), fixed.decode().splitlines(keepends=True),
    fromfile="a/" + rel, tofile="b/" + rel)))
metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
print("PASS: Vision-capable vLLM retained; audited SM121 o_proj repair installed")
PY
SH

# =============================================================================
# 9. Lightweight installed-source/GPU preflight (not an automatic model launch)
# =============================================================================
COPY --chmod=0755 <<'VERIFYPY' /workspace/verify_deepseek_vision_dspark.py
#!/usr/bin/env python3
"""Verify installed Vision + native DSpark + SM121/FlashInfer paths; no model download."""
import argparse
import hashlib
import os
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys

REQUIRED_FILES = (
    'vllm/models/deepseek_v4/nvidia/vl_model.py',
    'vllm/models/deepseek_v4/common/mm_preprocess.py',
    'vllm/models/deepseek_v4/common/vision.py',
    'vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py',
    'vllm/models/deepseek_v4/nvidia/dspark.py',
    'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py',
    'vllm/v1/worker/gpu/spec_decode/dspark/utils.py',
    'vllm/v1/attention/backends/mla/sparse_swa.py',
    'vllm/tokenizers/deepseek_v4.py',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files-only', action='store_true')
    p.add_argument('--compile-sparse', action='store_true')
    a = p.parse_args()

    dist = md.distribution('vllm')
    for rel in REQUIRED_FILES:
        path = Path(dist.locate_file(rel))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing installed implementation: {path}')
        compile(path.read_text(), str(path), 'exec')

    # Verify the installed source, not merely presence of an unpatched wheel.
    if os.environ.get('VLLM_USE_AOT_COMPILE') != '0':
        raise RuntimeError('This working Vision profile requires VLLM_USE_AOT_COMPILE=0.')
    if os.environ.get('VLLM_USE_BREAKABLE_CUDAGRAPH') != '1':
        raise RuntimeError('FULL_AND_PIECEWISE Vision needs VLLM_USE_BREAKABLE_CUDAGRAPH=1.')
    repair = json.loads(Path('/workspace/provenance/dsv4-oproj-repair.json').read_text())
    installed = Path(dist.locate_file(repair['file']))
    if hashlib.sha256(installed.read_bytes()).hexdigest() != repair['installed_after_sha256']:
        raise RuntimeError('Installed o_proj.py does not match the recorded repair.')
    b12x_meta = json.loads(Path('/workspace/provenance/b12x-build.json').read_text())
    if b12x_meta.get('enabled'):
        print('B12X package:', md.version('b12x'))

    versions = {}
    for name in (
        'vllm', 'torch', 'torchvision', 'torchaudio', 'transformers',
        'flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache',
        'nvidia-cutlass-dsl', 'apache-tvm-ffi', 'ray'
    ):
        versions[name] = md.version(name)
    print(json.dumps({
        'versions': versions,
        'machine': platform.machine(),
        'installed_vision_dspark_files': 'PASS',
    }, indent=2))
    if a.files_only:
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Run this inside the DGX Spark container with --gpus all.')
    cap = torch.cuda.get_device_capability(0)
    if cap != (12, 1):
        raise RuntimeError(f'Expected DGX Spark SM121, detected capability {cap}.')
    if torch.version.cuda is None or not torch.version.cuda.startswith('13.'):
        raise RuntimeError(f'Expected CUDA 13 PyTorch, got {torch.version.cuda}')

    vl = importlib.import_module('vllm.models.deepseek_v4.nvidia.vl_model')
    ds = importlib.import_module('vllm.models.deepseek_v4.nvidia.dspark')
    spec = importlib.import_module('vllm.v1.worker.gpu.spec_decode.dspark.speculator')
    utils = importlib.import_module('vllm.v1.worker.gpu.spec_decode.dspark.utils')
    for module, symbol in (
        (vl, 'DeepseekV4ForConditionalGeneration'),
        (ds, 'DSparkDeepseekV4ForCausalLM'),
        (spec, 'DSparkSpeculator'),
        (utils, 'load_dspark_model'),
    ):
        if not hasattr(module, symbol):
            raise RuntimeError(f'Missing runtime symbol {module.__name__}.{symbol}')

    # Target checkpoint: 64 attention heads -> TP2 gives 32 heads/rank.
    # DSpark non-causal SWA pads (window=128 + k=3/6) to 192.
    sparse_swa = importlib.import_module('vllm.v1.attention.backends.mla.sparse_swa')
    width_fn = getattr(sparse_swa, 'get_dspark_swa_index_width', None)
    if width_fn is None:
        raise RuntimeError('Missing get_dspark_swa_index_width in installed vLLM.')
    dspark_width_k3 = int(width_fn(128, 3))
    dspark_width_k6 = int(width_fn(128, 6))
    if (dspark_width_k3, dspark_width_k6) != (192, 192):
        raise RuntimeError(
            f'Unexpected DSpark SWA widths: k3={dspark_width_k3}, k6={dspark_width_k6}'
        )
    tp2_heads = 32
    vision_prefill_width = 128 + 384  # sliding_window + vision_max_n_token

    from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
    for width in (dspark_width_k3, vision_prefill_width):
        if not has_flashinfer_sparse_mla_sm120_config(tp2_heads, width):
            raise RuntimeError(
                f'Missing SM12x sparse MLA envelope: heads={tp2_heads}, topk={width}'
            )

    import flashinfer
    configs = flashinfer.mla.supported_sparse_mla_sm120_configs()
    dsv4 = configs.get('dsv4')
    if dsv4 is None:
        raise RuntimeError('FlashInfer does not expose the DSV4 SM120 sparse-MLA family.')
    for width in (dspark_width_k3, vision_prefill_width):
        if not dsv4.supports_decode(tp2_heads, width, page_block_size=64):
            raise RuntimeError(
                f'FlashInfer DSV4 decode envelope rejects heads={tp2_heads}, topk={width}'
            )

    # Vision uses the DSV4 dual-cache prefill route. Check that exact TP2 32x512
    # shape against the pinned FlashInfer planner, not just the decode envelope.
    plan = importlib.import_module('flashinfer.mla._sparse_mla_sm120_plan')
    dsv4_type = getattr(plan, '_MODEL_TYPE_DSV4')
    dual_ok = plan.prefill_mg_dual_eligible(
        dsv4_type, tp2_heads, vision_prefill_width, 64, True
    )
    if not dual_ok:
        raise RuntimeError(
            f'FlashInfer DSV4 dual-cache prefill rejects heads={tp2_heads}, topk={vision_prefill_width}'
        )

    nccl = Path('/usr/lib/aarch64-linux-gnu/libnccl.so.2')
    if not nccl.is_file():
        raise RuntimeError(f'Missing source-built NCCL: {nccl}')

    print('GPU:', torch.cuda.get_device_name(0), 'capability:', cap)
    print('Vision class: DeepseekV4ForConditionalGeneration')
    print('DSpark class: DSparkDeepseekV4ForCausalLM')
    print('DSpark speculator: DSparkSpeculator')
    print(f'DSpark TP2 sparse shape: heads={tp2_heads}, topk={dspark_width_k3} (k=3)')
    print(f'Vision TP2 dual-cache prefill shape: heads={tp2_heads}, topk={vision_prefill_width}')
    print('SM12x sparse MLA planner eligibility probes: PASS (not numerical kernel execution)')
    if a.compile_sparse:
        from flashinfer.jit.mla import gen_sparse_mla_sm120_module
        built = gen_sparse_mla_sm120_module().build_and_load()
        print('SM12x sparse MLA module compiled/loaded:', type(built).__name__)
        print('Compile/load does not execute or numerically validate dual-cache attention.')
    print('Preflight PASS. Full TP2 model load + text/image + DSpark acceptance still require two Sparks.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'PREFLIGHT FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
VERIFYPY

RUN python3 /workspace/verify_deepseek_vision_dspark.py --files-only && \
    test -x /usr/local/cuda/bin/ptxas && \
    test ! -d /workspace/vllm && test ! -d /workspace/flashinfer && \
    test ! -d /workspace/DeepGEMM && test ! -d /workspace/.cargo && \
    test ! -d /workspace/.rustup && test ! -d /workspace/wheels && \
    test ! -d /root/.cache/pip && test ! -d /root/.cache/uv
WORKDIR /workspace
EXPOSE 8000
# Keep the CUDA-base entrypoint. The host run_cluster_dual.sh controls serving.
CMD ["bash"]
