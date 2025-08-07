# Parallelization

Each top-level for-loop will be parallelized, within a kernel
- adding a non-static `if` over the top of a for-loop will lead to the for-loop not being parallelized
- a for-loop inside one or more inlined @ti.func, and/or static `if` will be still be parallelized however

Each top-level for-loop will be launched as a separate GPU kernel, under the hood.

The top-level for-loop can be encapsulate in one or more of the following, and still be parallelized:
- if statements where the conditional is `ti.static`
- inline functions (`@ti.func`)
