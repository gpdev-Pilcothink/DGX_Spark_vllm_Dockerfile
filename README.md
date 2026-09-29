# DGX Spark vLLM Recipes

This repository contains Dockerfiles, launch scripts, and serving recipes that I use for running vLLM on a dual NVIDIA DGX Spark setup.

The `run_cluster_dual.sh` script used in this repository was inspired by the following projects:

https://github.com/eugr/spark-vllm-docker/blob/main/launch-cluster.sh

https://github.com/vllm-project/vllm/blob/main/examples/ray_serving/run_cluster.sh

It has been reorganized and extended specifically for a two-node DGX Spark environment.

Most recipes in this repository assume a setup with:

```text
DGX Spark × 2 || DGX Spark x 1
```

You can either build the Docker image directly from the Dockerfile included with each recipe, or use a prebuilt Docker image when one is provided.

Choose the model and version you want to run, build or download the corresponding Docker image, and then follow the recipe instructions for serving the model.

This repository primarily uses `run_cluster_dual.sh`, and the provided launch examples are generally optimized for a dual DGX Spark configuration.


## Simple Memory Settings (Optional)

Before running the serving recipes, you can use
[`simple-memory-settings.sh`](./simple-memory-settings.sh) to adjust
two Linux memory-management settings on your DGX Spark.

| Setting | Purpose | Trade-off |
| --- | --- | --- |
| `vm.swappiness` | Controls the relative preference for swapping anonymous memory versus reclaiming filesystem cache. Higher values favor swapping more. | May help retain useful filesystem cache, but additional swapping can increase storage I/O or zram CPU overhead. |
| `vm.compaction_proactiveness` | Controls how aggressively Linux compacts memory in the background to prepare contiguous free-memory blocks. | May help allocations that require contiguous memory, but more compaction consumes CPU time and can cause latency spikes. |

### Suggested Values

| Profile | Swappiness | Compaction proactiveness |
| --- | ---: | ---: |
| Recommended starting point for this repository | 150 | 40 |
| More aggressive compaction profile | 150 | 80 |

Start with **150 / 40** and compare performance against your existing
settings. Consider **150 / 80** only when stronger background compaction
benefits your workload.

These are repository tuning suggestions, not official NVIDIA defaults
or guaranteed performance improvements. Swappiness tuning requires
active swap to be useful; this script does not create or enable swap.

> [!WARNING]
> Higher values do not necessarily mean better performance.
> Excessive swapping and background compaction can cause substantial
> overhead, latency spikes, and **severe slowdowns**.
> Compare throughput and latency using the same model and workload,
> and reduce the values or restore the baseline if performance worsens.

### Usage

Run the script directly on each DGX Spark host you want to tune,
outside the Docker container:

```bash
bash simple-memory-settings.sh
```

Select **option 2** to enter `150` for swappiness and `40` for
compaction proactiveness. For the more aggressive profile, enter
`150` and `80`, or select the existing **option 1** preset.

Changes take effect immediately and are saved to:

```text
/etc/sysctl.d/simple-memory-set.conf
```

The saved settings are loaded on subsequent boots. The script does
not modify `/etc/sysctl.conf` or create a `.bak` file.

### Restore

Select **option 3** to immediately apply **60 / 20** and remove
`/etc/sysctl.d/simple-memory-set.conf`. Subsequent boots follow the
system's existing configuration.

The restore pair **60 / 20** is the baseline used by the maintainer.
It is fixed in the script, not a backup of each user's original values.
Record your current settings before making changes if your baseline differs:

```bash
sysctl vm.swappiness vm.compaction_proactiveness
```


## Multi-Node Cluster Setup

> [!IMPORTANT]
> If you are using **two or more DGX Spark systems**, you must complete the cluster setup using the official NVIDIA DGX Spark playbooks below before running any multi-node recipe in this repository.

The recipes maintained by **Pilcothink** are developed with the official NVIDIA DGX Spark playbooks as their reference and assume that the required cluster configuration is already complete.

Follow the instructions applicable to your node count and network topology:

1. [NCCL for Multiple Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks) — Configure the required network connectivity and verify communication between nodes with NCCL tests.
2. [vLLM Multi-Node Setup](https://build.nvidia.com/spark/vllm/multi-node) — Complete the prerequisites for running vLLM across multiple DGX Spark systems.

Once the cluster setup and communication tests are complete, return to the model-specific README and follow its Docker image and launch instructions.


## Docker Image and Dockerfile Sources

Most Docker images and Dockerfiles in this repository are built with reference to the following upstream projects:

https://github.com/eugr/spark-vllm-docker

https://github.com/vllm-project/vllm

https://github.com/flashinfer-ai/flashinfer

https://github.com/local-inference-lab/b12x

The goal is to keep modifications as close as possible to upstream implementations.

Whenever possible, patches and changes are based on identifiable upstream commits, pull requests, or official source changes rather than undocumented modifications.

Some recipes may include additional patches, configuration changes, or performance tuning specifically for DGX Spark.

## Notes

This repository is primarily intended for experimentation, benchmarking, and reproducible model serving on a dual DGX Spark system.

Configurations may need to be adjusted for different vLLM versions, model revisions, CUDA versions, networking environments, or upstream changes.

Always review the corresponding Dockerfile and recipe before using it in your own environment.


## Optimized Run Recipes by vLLM Version

The following optimized run recipes are currently available in this repository.

### vLLm 0.22
* Solar-Open2-250B(nota-ai/Solar-Open2-250B-Nota-INT4)

### vLLM 0.28

* DeepSeek-V4-Flash-Vision-Exp (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
* GLM-5.3-Flash (nvidia/GLM-5.3-Flash-NVFP4)
* Qwen3.8-Flash-Next (nvidia/Qwen3.8-Flash-Next-NVFP4)

### vLLM 0.29

* DeepSeek-V4-Flash-Vision-Exp (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
* GLM-5.3-Flash (nvidia/GLM-5.3-Flash-NVFP4)

### vLLM 0.30
* MiMo-V2.6-Flash-MOPD (XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
* GLM-5.3-Flash (nvidia/GLM-5.3-Flash-NVFP4)

Each recipe may include model-specific Dockerfiles, patches, backend configurations, and serving parameters optimized for DGX Spark.

The available recipes may change as vLLM, FlashInfer, B12X, and the corresponding model implementations are updated.
