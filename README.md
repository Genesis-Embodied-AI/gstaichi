# GS-Taichi

[Taichi](https://github.com/taichi-dev/taichi) was forked in June 2025. This repository (or gs-taichi) is now a fully independent project with no intention of maintaining backward compatibility with the original taichi. Whilst the repo largely resembles upstream for now, we have made the following changes:
- revamped continuous integration, to run using recent python versions (up to 3.13), recent mac os x versions (up to 15), and to run reliably (at least 95% of runs with correct code succeed)
- added dataclasses.dataclass structs:
    - work with both ndarrays and fields (cf ti.struct (field only), ti.dataclass (field only), ti.data_oriented (field only), argpack (ndarray only))
    - can be passed into child `ti.func`tions (cf argpack)
    - does not affect kernel runtime speed (kernels see only the underlying arrays, no indirection is added within the kernel layer)
- removed GUI/GGUI
- upgraded supported OS and Python versions (eg added support for Python 3.13)

Planned features:
- reduce warm cache launch latency
- (maybe) add launch args caching, to reduce launch latency
- make dataclasses.dataclass nestable

Planned pruning:
- remove argpack
- remove ti CLI
- remove OpenGL (please use Vulkan)
- remove mobile device support (Android etc)
- remove support for NVidia GPUs earlier than sm_60/Pascal

# What is gs-taichi?

GS-Taichi is a high performance multi-platform compiler, targeted at physics simulations. It compiles Python code into parallelizable kernels that can run on:
- NVidia GPUs, using CUDA
- Vulkan-compatible GPUs, using SPIR-V
- Mac Metal GPUs
- x86 and arm64 CPUs

GS-Taichi supports automatic differentiation. GS-Taichi lets you build fully fused GPU kernels, using Python.

[Genesis simulator](https://genesis-world.readthedocs.io/en/latest/)'s best-in-class performance can be largely attributed to Taichi, its underlying GPU acceleration framework for Python. Given how critical is this component, we decided to fork Taichi and build our own very framework from there, so that from now on, we are free to drive its development in the direction that best supports the continuous improvement of Genesis simulator.

# Installation
## Prerequisites
- Python 3.10-3.13
- Mac OS 14, 15, Windows, or Ubuntu 22.04-24.04 or compatible

## Procedure
```
pip install gs-taichi
```

(For how to build from source, see our CI build scripts, e.g. [linux build scripts](.github/workflows/scripts_new/linux_x86/) )

# Documentation

- [docs](docs/lang/articles)
- [API reference](https://ideal-adventure-2n6lpyw.pages.github.io/taichi.html)

# Something is broken!

- [Create an issue](https://github.com/Genesis-Embodied-AI/taichi/issues/new/choose)

# Acknowledgements

- The original [Taichi](https://github.com/taichi-dev/taichi) was developed with love by many contributors over many years. For the full list of contributors and credits, see [Original taichi contributors](https://github.com/taichi-dev/taichi?tab=readme-ov-file#contributing)
