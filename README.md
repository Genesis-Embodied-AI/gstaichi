# GS-Taichi

[Taichi](https://github.com/taichi-dev/taichi) was forked in June 2025. It is being developed to target the [Genesis physics simulator](https://github.com/Genesis-Embodied-AI/Genesis). Whilst the repo largely resembles upstream for now, we have made the following changes:
- revamped continuous integration, to run using recent python versions (up to 3.13), recent mac os x versions (up to 15), and to run reliably (at least 95% of runs with correct code succeed)
- added dataclasses.dataclass structs:
    - work with both ndarrays and fields (cf ti.struct (field only), ti.dataclass (field only), ti.data_oriented (field only), argpack (ndarray only))
    - can be passed into child `ti.func`tions (cf argpack)
    - does not affect kernel runtime speed (kernels see only the underlying arrays, no indirection is added within the kernel layer)
- removed GUI/GGUI
    - not used by Genesis (we are using other, dedicated, rendering engines, see [Genesis physics simulator](https://github.com/Genesis-Embodied-AI/Genesis) for more details)
- removed support for:
    - older versions of Mac OS X (< 14)
    - older versions of Python (< 3.10)
    - older versions of NVidia GPUs (< sm_60, i.e. < Pascal)
    - OpenGL (please use Vulkan)
    - Mobile devices (Android etc)

We will aggressively prune code that is not used by [Genesis project](https://github.com/Genesis-Embodied-AI/Genesis) in order to stay focused on our mission to [build generalist robots to unlock unlimited physical labor, so humans can focus on creativity, curiosity, and what we love](https://genesis-ai.company/)

Other changes to gs-taichi are planned:
- reduce warm cache launch latency
- add launch arguments caching, to reduce kernel launch latency (only if we find Genesis needs this)
- make dataclasses.dataclass nestable
- remove argpack (Genesis doesn't use argpack, argpack functionality is a subset of dataclasses.dataclass and caching)

For the time being, the documentation is in flux. Please feel free to raise an issue with any inconsistencies or issues you find.

# Getting started

- See [hello_world.md](docs/lang/articles/get-started/hello_world.md)

# Documentation

- [docs](docs/lang/articles)
- [API reference](https://ideal-adventure-2n6lpyw.pages.github.io/taichi.html)

# Something is broken!

- [Create an issue](https://github.com/Genesis-Embodied-AI/taichi/issues/new/choose)

# Acknowledgements

- The original [Taichi](https://github.com/taichi-dev/taichi) was developed with love by many contributors over many years. For the full list of contributors and credits, see [Original taichi contributors](https://github.com/taichi-dev/taichi?tab=readme-ov-file#contributing)
