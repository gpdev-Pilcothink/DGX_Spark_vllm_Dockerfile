# syntax=docker/dockerfile:1.7
# DGX Spark / SM121 / DeepSeek-V4-Flash-Vision-Exp + native DSpark
# Prepared 2026-09-09. Static-validated only; DGX Spark runtime validation remains required.
# Source: official vLLM + official FlashInfer + DeepGEMM; eugr supplies SM121/Spark-only patches.
# DSpark is REQUIRED for this target image and is verified in both source and built wheel.
# Standalone: no model/tokenizer data is downloaded during build.

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
ARG FLASHINFER_REF=27d5b029818e530a9fd0c6b3b356217b5bef1226
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ARG VLLM_SOURCE_MODE=remote
ARG VLLM_SOURCE_COMMIT=""
# Optional package only. This does NOT switch to a text-only B12X vLLM fork.
ARG ENABLE_B12X=0
ARG B12X_REF=06b4de7c723e6f166d65abf5909c5b7d0f8acc68
ARG PRE_TRANSFORMERS=0
# PRE_TRANSFORMERS=1 also requires an exact version, to avoid a moving pre-release.
ARG TRANSFORMERS_PRE_VERSION=""

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

FROM torch-base AS runner
# Required on BOTH Ray nodes before Ray starts, not only the head shell.
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

COPY --chmod=0755 <<'SERVE' /workspace/serve_deepseek_vision_dspark.sh
#!/usr/bin/env bash
# Default: DSpark ON. --reference-baseline is exclusively for sequential A/B testing.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/launch_deepseek_vision_dspark.py" "$@"
SERVE

COPY --chmod=0755 <<'LAUNCHPY' /workspace/launch_deepseek_vision_dspark.py
#!/usr/bin/env python3
"""Launch the pinned Vision + DSpark stack; --dry-run never starts Ray or a model.

--reference-baseline is an explicit A/B diagnostic only. Normal serving always
uses native DSpark. This launcher does not change model weights or KV precision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Mapping

MODEL_ID = 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp'
# Official Hub main verified on 2026-09-07; no model data is fetched at image build.
DEFAULT_MODEL_REVISION = '6821d6ad3681a4b137b066b76094fa82ebd0a380'
REQUIRED_ENV = {
    'VLLM_USE_V2_MODEL_RUNNER': '1',
    'VLLM_USE_BREAKABLE_CUDAGRAPH': '1',
    'DG_JIT_USE_NVRTC': '0',
    'CUTE_DSL_ARCH': 'sm_121a',
}

def integer(env: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    value = env.get(key, str(default))
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'{key} must be an integer, not {value!r}')
    result = int(value)
    if not lo <= result <= hi:
        raise ValueError(f'{key} must be within [{lo}, {hi}]')
    return result

def flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key, '1' if default else '0')
    if value not in {'0', '1'}:
        raise ValueError(f'{key} must be 0 or 1')
    return value == '1'

def capture_sizes(seqs: int, k: int, override: str | None = None) -> list[int]:
    # Include every uniform target-verification batch, not just the largest.
    required = {(k + 1) * n for n in range(1, seqs + 1)}
    ceiling = max(required)
    if override:
        sizes = json.loads(override)
        if (not isinstance(sizes, list) or not sizes
                or any(type(x) is not int or x <= 0 for x in sizes)
                or sizes != sorted(set(sizes))):
            raise ValueError('CUDAGRAPH_CAPTURE_SIZES_JSON must be sorted positive unique integers')
        if not required.issubset(sizes):
            raise ValueError(f'Missing verification capture sizes: {sorted(required - set(sizes))}')
        return sizes
    dense = {1, 2, 4} | set(range(8, min(ceiling, 255) + 1, 8))
    if ceiling >= 256:
        dense.update(range(256, ceiling + 1, 16))
    return sorted((dense | required) & set(range(1, ceiling + 1)))

def local_model_manifest(path: Path) -> dict:
    path = path.resolve()
    config_path = path / 'config.json'
    index_path = path / 'model.safetensors.index.json'
    config = json.loads(config_path.read_text())
    if (config.get('model_type') != 'deepseek_v4'
            or config.get('vision_n_layers', 0) <= 0
            or config.get('num_nextn_predict_layers') != 3
            or config.get('vision_max_n_token') != 384):
        raise ValueError('The local config does not match the audited DeepSeek V4 Vision checkpoint')
    index = json.loads(index_path.read_text())
    weights = index.get('weight_map', {})
    if not all(any(n.startswith(f'mtp.{i}.') for n in weights) for i in range(3)):
        raise ValueError('The checkpoint index is missing one or more folded DSpark stages')
    shards = {}
    for name in sorted(set(weights.values())):
        # HF snapshot files normally symlink into the adjacent blobs directory.
        # Reject unsafe index paths, but do not reject those legitimate symlinks.
        part = PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or '\\' in name:
            raise ValueError(f'Unsafe checkpoint index path: {name}')
        shard = path / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f'Missing/empty/unsafe checkpoint shard: {name}')
        shards[name] = shard.stat().st_size
    if not shards:
        raise ValueError('Checkpoint contains no shards')
    return {
        'kind': 'local', 'path': str(path),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'shards_bytes': shards,
        'full_weight_content_hashed': False,
    }

def resolve_model(env: Mapping[str, str], dry_run: bool) -> tuple[str, str | None, dict]:
    model = env.get('MODEL_PATH', MODEL_ID)
    if Path(model).is_dir():
        manifest = local_model_manifest(Path(model))
        return manifest['path'], None, manifest
    if model != MODEL_ID:
        raise ValueError('MODEL_PATH must be the official target repo ID or an existing local directory')
    revision = env.get('MODEL_REVISION', DEFAULT_MODEL_REVISION)
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        if dry_run:
            raise ValueError('--dry-run with a remote model requires an exact MODEL_REVISION SHA')
        from huggingface_hub import HfApi
        revision = HfApi().model_info(repo_id=model, revision=revision or 'main').sha
    if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Failed to resolve the model to an immutable Hugging Face revision')
    return model, revision, {'kind': 'hub', 'repo_id': model, 'revision': revision}

def build_command(env: Mapping[str, str], model: str, revision: str | None,
                  reference: bool = False) -> tuple[list[str], dict]:
    for name, value in REQUIRED_ENV.items():
        if name in env and env[name] != value:
            raise ValueError(f'{name} must be {value} for this launcher')
    if env.get('DSPARK_ADAPTIVE_VERIFICATION', 'false') != 'false':
        raise ValueError('Adaptive verification is unsupported by the audited SM121 indexer path')
    tp = integer(env, 'TP_SIZE', 2, 2, 2)
    seqs = integer(env, 'MAX_NUM_SEQS', 4, 1, 64)
    k = integer(env, 'DSPARK_TOKENS', 3, 1, 6)
    if k not in (3, 6):
        raise ValueError('Use DSPARK_TOKENS=3; 6 is an explicit performance experiment')
    length = integer(env, 'MAX_MODEL_LEN', 16384, 1024, 1048576)
    batched = integer(env, 'MAX_BATCHED_TOKENS', 4096, 256, 1048576)
    if batched < seqs * (k + 1):
        raise ValueError('MAX_BATCHED_TOKENS is smaller than the largest verification batch')
    memory = float(env.get('GPU_MEMORY_UTILIZATION', '0.85'))
    if not 0.0 < memory < 1.0:
        raise ValueError('GPU_MEMORY_UTILIZATION must be between 0 and 1')
    images = integer(env, 'MAX_IMAGES', 1, 1, 64)
    port = integer(env, 'API_PORT', 8001, 1, 65535)
    prefix = flag(env, 'ENABLE_PREFIX_CACHING', False)
    autotune = flag(env, 'ENABLE_FLASHINFER_AUTOTUNE', True)
    ep = flag(env, 'ENABLE_EXPERT_PARALLEL', True)
    eager = flag(env, 'ENFORCE_EAGER', False)
    metrics = env.get('PER_REQUEST_SPEC_DECODE_METRICS', 'summary')
    if metrics not in ('none', 'summary', 'detailed'):
        raise ValueError('PER_REQUEST_SPEC_DECODE_METRICS must be none, summary, or detailed')
    sizes = capture_sizes(seqs, 0 if reference else k,
                          env.get('CUDAGRAPH_CAPTURE_SIZES_JSON'))
    compilation = {
        'cudagraph_mode': 'FULL_AND_PIECEWISE',
        'cudagraph_capture_sizes': sizes,
        'custom_ops': ['all'],
        # Upstream warns that this optimization can be incorrect with a MoE draft.
        'fast_moe_cold_start': False,
    }
    spec = {
        'method': 'dspark', 'model': model, 'num_speculative_tokens': k,
        'draft_sample_method': 'probabilistic',
        'rejection_sample_method': 'standard',
        'enable_adaptive_verification': False,
    }
    if revision:
        spec.update(revision=revision, code_revision=revision)
    compact = lambda x: json.dumps(x, separators=(',', ':'))
    args = ['vllm', 'serve', model,
            '--served-model-name', env.get('SERVED_MODEL_NAME', 'deepseek-v4-vision'),
            '--host', env.get('API_HOST', '0.0.0.0'), '--port', str(port),
            '--tensor-parallel-size', str(tp), '--distributed-executor-backend', 'ray',
            '--tokenizer-mode', 'deepseek_v4', '--trust-remote-code',
            '--dtype', 'bfloat16', '--attention-backend', 'FLASHINFER_MLA_SPARSE_DSV4',
            '--kv-cache-dtype', 'fp8', '--block-size', '256',
            '--max-model-len', str(length), '--max-num-seqs', str(seqs),
            '--max-num-batched-tokens', str(batched),
            '--gpu-memory-utilization', str(memory),
            '--limit-mm-per-prompt', compact({'image': images, 'video': 0}),
            '--per-request-spec-decode-metrics', metrics,
            '--reasoning-parser', 'deepseek_v4',
            '--reasoning-config', compact({'reasoning_parser': 'deepseek_v4',
                'reasoning_start_str': '', 'reasoning_end_str': ''}),
            '--tool-call-parser', 'deepseek_v4', '--enable-auto-tool-choice',
            '--enable-prefix-caching' if prefix else '--no-enable-prefix-caching',
            '--enable-flashinfer-autotune' if autotune else '--no-enable-flashinfer-autotune']
    if ep:
        args.append('--enable-expert-parallel')
    if eager:
        args.append('--enforce-eager')
    else:
        args += ['--compilation-config', compact(compilation)]
    if not reference:
        args += ['--speculative-config', compact(spec)]
    if revision:
        args += ['--revision', revision, '--tokenizer-revision', revision, '--code-revision', revision]
    settings = {
        'reference_baseline_only': reference, 'dspark': None if reference else spec,
        'tp_size': tp, 'expert_parallel': ep, 'max_model_len': length,
        'max_num_seqs': seqs, 'max_batched_tokens': batched,
        'gpu_memory_utilization': memory, 'max_images': images,
        'prefix_caching': prefix, 'flashinfer_autotune': autotune,
        'enforce_eager': eager, 'compilation_config': None if eager else compilation,
        'kv_cache_dtype': 'fp8', 'dtype': 'bfloat16',
    }
    return args, settings

def check_ray(env: Mapping[str, str], model: str, expected: dict) -> list[dict]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address=env.get('RAY_ADDRESS', 'auto'))
    try:
        nodes = [n for n in ray.nodes() if n.get('Alive') and n.get('Resources', {}).get('GPU', 0) >= 1]
        if len(nodes) != 2:
            raise RuntimeError(f'Expected exactly two GPU nodes in this dedicated cluster; found {len(nodes)}')
        # Send a function whose body has no imports from this local helper module.
        def inspect_node(model_path: str, local: bool) -> dict:
            import hashlib, importlib.metadata as md, json, os, platform, socket
            from pathlib import Path
            import torch
            meta_path = Path('/workspace/build-metadata.yaml')
            result = {
                'host': socket.gethostname(), 'machine': platform.machine(),
                'capability': list(torch.cuda.get_device_capability(0)),
                'gpu_name': torch.cuda.get_device_name(0),
                'versions': {x: md.version(x) for x in ('vllm', 'torch', 'flashinfer-python', 'ray')},
                'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                'required_env': {n: os.environ.get(n) for n in
                    ('VLLM_USE_V2_MODEL_RUNNER', 'VLLM_USE_BREAKABLE_CUDAGRAPH',
                     'DG_JIT_USE_NVRTC', 'CUTE_DSL_ARCH')},
            }
            if local:
                root = Path(model_path)
                index = root / 'model.safetensors.index.json'
                weights = json.loads(index.read_text())['weight_map']
                result['local_model'] = {
                    'config_sha256': hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
                    'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
                    'shards_bytes': {n: (root / n).stat().st_size for n in sorted(set(weights.values()))},
                }
            return result
        remote = ray.remote(num_gpus=1)(inspect_node)
        futures = [remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n['NodeID'], soft=False)).remote(model, expected['kind'] == 'local') for n in nodes]
        rows = ray.get(futures, timeout=180)
        for row in rows:
            if row['machine'] != 'aarch64' or row['capability'] != [12, 1]:
                raise RuntimeError(f'Not DGX Spark SM121: {row}')
            if row['required_env'] != REQUIRED_ENV:
                raise RuntimeError('Set the required runtime environment on BOTH Ray containers before Ray starts')
            if expected['kind'] == 'local':
                check = {key: expected[key] for key in ('config_sha256', 'index_sha256', 'shards_bytes')}
                if row['local_model'] != check:
                    raise RuntimeError(f'Local model inventory differs on {row["host"]}')
        if rows[0]['versions'] != rows[1]['versions'] or rows[0]['metadata_sha256'] != rows[1]['metadata_sha256']:
            raise RuntimeError('Node image provenance or package versions differ; copy one built image to both nodes')
        return rows
    finally:
        ray.shutdown()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reference-baseline', action='store_true',
                        help='DIAGNOSTIC ONLY: omit DSpark for a sequential A/B reference run')
    a = parser.parse_args()
    env = dict(os.environ)
    # Validate before any network or Ray call.
    build_command(env, env.get('MODEL_PATH', MODEL_ID), None, a.reference_baseline)
    model, revision, manifest = resolve_model(env, a.dry_run)
    command, settings = build_command(env, model, revision, a.reference_baseline)
    os.environ.update(REQUIRED_ENV)
    os.environ.setdefault('RAY_ADDRESS', 'auto')
    record = {'dry_run': a.dry_run, 'model': manifest, 'settings': settings,
              'required_env': REQUIRED_ENV, 'command': command}
    if a.reference_baseline:
        print('DIAGNOSTIC REFERENCE ONLY: DSpark disabled for quality/performance A/B; not the production profile.', file=sys.stderr)
    if not a.dry_run:
        record['ray_nodes'] = check_ray(os.environ, model, manifest)
        meta = Path('/workspace/build-metadata.yaml')
        record['build_metadata_sha256'] = hashlib.sha256(meta.read_bytes()).hexdigest()
    output = Path(env.get('LAUNCH_RECORD', '/workspace/deepseek-launch.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    print('COMMAND:', shlex.join(command), flush=True)
    if not a.dry_run:
        os.execvp(command[0], command)
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'LAUNCH FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
LAUNCHPY

COPY --chmod=0755 <<'VERIFYPY' /workspace/verify_deepseek_vision_dspark.py
#!/usr/bin/env python3
"""Verify installed Vision + native DSpark + SM121/FlashInfer paths; no model download."""
import argparse
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

COPY --chmod=0755 <<'SMOKEPY' /workspace/smoke_deepseek_vision_dspark.py
#!/usr/bin/env python3
"""Functional smoke test, not a benchmark or a proof of general model quality.

Uses local synthetic PNG data, checks counterfactual image answers, then requires
active AND accepted DSpark tokens on long image-conditioned requests.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def png(left_red: bool) -> bytes:
    w, h = 512, 256
    red, blue = b'\xff\x00\x00', b'\x00\x00\xff'
    left, right = (red, blue) if left_red else (blue, red)
    row = b'\x00' + left * (w // 2) + right * (w // 2)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(row * h)) + chunk(b'IEND', b''))


def image_content(red_left: bool, question: str) -> list[dict]:
    return [{'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(png(red_left)).decode()}},
        {'type': 'text', 'text': question}]


def request(base: str, path: str, key: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(base.rstrip('/') + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:6000]}') from exc


def check_spec(result: dict, k: int) -> dict:
    metrics = (result.get('metrics') or {}).get('speculative_decoding')
    active = bool(metrics and metrics.get('num_spec_tokens') == k
                  and metrics.get('num_spec_steps', 0) > 0
                  and metrics.get('num_draft_tokens', 0) > 0)
    accepted = bool(active and metrics.get('num_accepted_draft_tokens', 0) > 0)
    return {'active': active, 'accepted_tokens_observed': accepted, 'metrics': metrics}


def tests() -> list[dict]:
    rows = [{'name': 'text', 'content': 'Reply with exactly: TEXT_OK', 'expected': 'TEXT_OK'}]
    question = 'What color fills the LEFT HALF of this image? Reply with only RED or BLUE.'
    for label, red, expected in [('red-left', True, 'RED'), ('blue-left', False, 'BLUE')]:
        rows.append({'name': label, 'content': image_content(red, question), 'expected': expected})
    # 96 numbered image-conditioned lines exercise continued generation beyond
    # the 128-token sliding window. >=512 generated tokens is checked below.
    long_question = (
        'Look at the LEFT and RIGHT halves of the supplied image. '
        'Produce exactly 96 numbered lines, from 001 through 096. '
        'Each line must say LEFT=<left color>; RIGHT=<right color>. '
        'Use the actual colors you see, either RED or BLUE. '
        'Repeat the observations on all 96 lines. Do not add an introduction.'
    )
    for red, label in [(True, 'red-left-long-dspark'), (False, 'blue-left-long-dspark')]:
        rows.append({'name': label, 'content': image_content(red, long_question),
                     'expected_left': 'RED' if red else 'BLUE', 'long': True})
    return rows


def evaluate(case: dict, result: dict, expected_k: int) -> dict:
    choice = result['choices'][0]
    content = choice['message'].get('content') or ''
    complete = choice.get('finish_reason') == 'stop'
    if not case.get('long'):
        answer = re.sub(r'[^A-Z_]', '', content.upper()) == case['expected']
        checks = {'answer_correct': answer, 'finished_normally': complete}
    else:
        observed = re.findall(r'LEFT\s*[:=]\s*(RED|BLUE)', content.upper())
        checks = {
            'left_color_follows_image': len(observed) >= 90 and set(observed) == {case['expected_left']},
            'generated_at_least_512_tokens': result.get('usage', {}).get('completion_tokens', 0) >= 512,
            'finished_normally': complete,
        }
        if expected_k > 0:
            spec = check_spec(result, expected_k)
            checks.update(dspark_active_on_image=spec['active'],
                          dspark_accepted_on_image=spec['accepted_tokens_observed'])
    return {'name': case['name'], 'checks': checks, 'passed': all(checks.values()), 'response': result}



def compare_reports(reference: dict, candidate: dict) -> dict:
    """Small greedy regression check. A token difference requires investigation,
    not an automatic claim of quality loss. Never extrapolate to all tasks.
    """
    errors = []
    a, b = reference.get('launch_record') or {}, candidate.get('launch_record') or {}
    if not a or not b or a.get('dry_run') or b.get('dry_run'):
        errors.append('Both runs need real launcher provenance, not a dry-run record')
    if not a.get('settings', {}).get('reference_baseline_only'):
        errors.append('Reference must be an explicitly launched no-DSpark diagnostic baseline')
    if b.get('settings', {}).get('reference_baseline_only'):
        errors.append('Candidate must have DSpark enabled')
    for field in ('model', 'build_metadata_sha256', 'required_env'):
        if not a.get(field) or a.get(field) != b.get(field):
            errors.append(f'Inconsistent or missing launch {field}')
    for field in ('tp_size', 'expert_parallel', 'max_model_len', 'max_num_seqs',
                  'max_batched_tokens', 'gpu_memory_utilization', 'max_images',
                  'prefix_caching', 'flashinfer_autotune', 'enforce_eager', 'kv_cache_dtype', 'dtype'):
        if a.get('settings', {}).get(field) != b.get('settings', {}).get(field):
            errors.append(f'Changed non-DSpark setting: {field}')
    if reference.get('requests_sha256') != candidate.get('requests_sha256'):
        errors.append('Request suite or generation settings changed')
    if not reference.get('all_passed') or not candidate.get('all_passed'):
        errors.append('At least one functional smoke suite did not pass')
    reference_rows = {r['name']: r for r in reference.get('tests', [])}
    comparisons = []
    for row in candidate.get('tests', []):
        ref = reference_rows.get(row['name'], {})
        try:
            c1 = ref['response']['choices'][0]
            c2 = row['response']['choices'][0]
            ids1, ids2 = c1.get('token_ids'), c2.get('token_ids')
            ids_present = isinstance(ids1, list) and bool(ids1) and isinstance(ids2, list) and bool(ids2)
            comparisons.append({
                'name': row['name'], 'token_ids_available': ids_present,
                'identical_greedy_tokens': ids_present and ids1 == ids2,
                'identical_visible_answer': c1['message'].get('content') == c2['message'].get('content'),
                'reference_end_to_end_seconds': ref.get('elapsed_seconds'),
                'candidate_end_to_end_seconds': row.get('elapsed_seconds'),
                'reference_server_reported_tps': (ref['response'].get('metrics') or {}).get('tokens_per_second'),
                'candidate_server_reported_tps': (row['response'].get('metrics') or {}).get('tokens_per_second'),
            })
        except (KeyError, IndexError, TypeError):
            errors.append(f'Missing successful response for {row["name"]}')
    status = 'invalid_comparison' if errors else (
        'small_suite_greedy_match' if comparisons and all(r['identical_greedy_tokens'] for r in comparisons)
        else 'token_differences_need_review')
    return {'status': status, 'errors': errors, 'cases': comparisons,
            'certifies_losslessness_for_all_inputs': False, 'certifies_maximum_performance': False,
            'timing_note': 'Uncontrolled client end-to-end timings include prefill and overhead. '
                           'Server metrics are reported separately. Warm-up, repeated runs and workload benchmarks are still required.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--model', default='deepseek-v4-vision')
    parser.add_argument('--api-key', default=os.getenv('VLLM_API_KEY', ''))
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--expect-dspark-tokens', type=int, default=3,
                        help='0 is exclusively for a no-DSpark diagnostic reference server.')
    parser.add_argument('--output', default='vision-dspark-smoke-result.json')
    parser.add_argument('--launch-record', default='/workspace/deepseek-launch.json')
    parser.add_argument('--compare-reference', help='JSON from a sequential no-DSpark baseline smoke run')
    args = parser.parse_args()
    report = {'model': args.model, 'base_url': args.base_url,
              'scope': 'functional smoke only; not a lossless-quality or maximum-speed certification',
              'tests': []}
    record_path = Path(args.launch_record)
    report['launch_record'] = json.loads(record_path.read_text()) if record_path.is_file() else None
    requests = []
    try:
        models = request(args.base_url, '/models', args.api_key, timeout=args.timeout)
        if args.model not in [m['id'] for m in models.get('data', [])]:
            raise RuntimeError(f'Model {args.model!r} is not advertised by the server')
        for case in tests():
            started = time.monotonic()
            try:
                payload = {'model': args.model, 'messages': [{'role': 'user', 'content': case['content']}],
                           'max_tokens': 2048 if case.get('long') else 64,
                           'temperature': 0, 'top_p': 1, 'seed': 0, 'stream': False,
                           'return_token_ids': True, 'chat_template_kwargs': {'thinking': False}}
                requests.append(payload)
                result = request(args.base_url, '/chat/completions', args.api_key, payload, args.timeout)
                row = evaluate(case, result, args.expect_dspark_tokens)
            except Exception as exc:
                row = {'name': case['name'], 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
            row['elapsed_seconds'] = round(time.monotonic() - started, 3)
            report['tests'].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'response'}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['all_passed'] = not report.get('error') and len(report['tests']) == 5 and all(x['passed'] for x in report['tests'])
    report['requests_sha256'] = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    if args.compare_reference:
        try:
            reference = json.loads(Path(args.compare_reference).read_text())
            report['comparison'] = compare_reports(reference, report)
        except Exception as exc:
            report['comparison'] = {'status': 'invalid_comparison', 'errors': [str(exc)]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved:', args.output)
    if not report['all_passed']:
        print('FAIL/REVIEW: formatting, early EOS or no accepted drafts is not by itself proof of a kernel bug.', file=sys.stderr)
        return 1
    print('PASS: functional smoke checks completed. This is not a general quality certification.')
    if report.get('comparison'):
        print('A/B:', report['comparison']['status'])
        if report['comparison']['status'] != 'small_suite_greedy_match':
            return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
SMOKEPY

RUN python3 /workspace/verify_deepseek_vision_dspark.py --files-only && \
    test -x /usr/local/cuda/bin/ptxas && \
    test ! -d /workspace/vllm && test ! -d /workspace/flashinfer && \
    test ! -d /workspace/DeepGEMM && test ! -d /workspace/.cargo && \
    test ! -d /workspace/.rustup && test ! -d /workspace/wheels && \
    test ! -d /root/.cache/pip && test ! -d /root/.cache/uv
WORKDIR /workspace
EXPOSE 8001
# Keep the normal CUDA-base entrypoint; usable with eugr/Ray cluster launchers.
CMD ["bash"]


# ---------------------------------------------------------------------------
# DeepSeek V4 runtime overlays
# ---------------------------------------------------------------------------


# Consolidated from the supplied Dockerfile.oproj-fix.
# Keep the already-installed environment; replace only o_proj.py.
RUN curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/vllm/8284955fe1d31f3aaddfc691161e348fb0fc39d4/vllm/models/deepseek_v4/nvidia/ops/o_proj.py \
  -o /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py \
  && python3 -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py

# DeepSeek V4 tool-call parser fix.
RUN curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/vllm/d98c8c030b648f4178da16977424cdadef1fb596/vllm/parser/deepseek_v4.py \
  -o /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py \
  && python3 -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py
