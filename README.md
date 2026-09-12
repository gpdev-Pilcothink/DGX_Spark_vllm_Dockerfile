# DGX Spark vLLM Recipes

This repository contains Dockerfiles, launch scripts, and serving recipes that I use for running vLLM on a dual NVIDIA DGX Spark setup.

The `run_cluster_dual.sh` script used in this repository was inspired by the following projects:

https://github.com/eugr/spark-vllm-docker/blob/main/launch-cluster.sh

https://github.com/vllm-project/vllm/blob/main/examples/ray_serving/run_cluster.sh

It has been reorganized and extended specifically for a two-node DGX Spark environment.

Most recipes in this repository assume a setup with:

```text
DGX Spark × 2
```

You can either build the Docker image directly from the Dockerfile included with each recipe, or use a prebuilt Docker image when one is provided.

Choose the model and version you want to run, build or download the corresponding Docker image, and then follow the recipe instructions for serving the model.

This repository primarily uses `run_cluster_dual.sh`, and the provided launch examples are generally optimized for a dual DGX Spark configuration.

## Docker Image and Dockerfile Sources

Most Docker images and Dockerfiles in this repository are built with reference to the following upstream projects:

https://github.com/eugr/spark-vllm-docker

https://github.com/vllm-project/vllm

https://github.com/flashinfer-ai/flashinfer

https://github.com/local-inference-lab/b12x

The goal is to keep modifications as close as possible to upstream implementations.

Whenever possible, patches and changes are based on identifiable upstream commits, pull requests, or official source changes rather than undocumented modifications.

Some recipes may include additional patches, configuration changes, or performance tuning specifically for DGX Spark. These modifications are documented in the corresponding Dockerfile or recipe whenever applicable.

## Notes

This repository is primarily intended for experimentation, benchmarking, and reproducible model serving on a dual DGX Spark system.

Configurations may need to be adjusted for different vLLM versions, model revisions, CUDA versions, networking environments, or upstream changes.

Always review the corresponding Dockerfile and recipe before using it in your own environment.
