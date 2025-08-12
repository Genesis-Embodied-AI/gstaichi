from typing import Any

import gstaichi as ti


def ti_init_same_arch(options: dict[str, Any] | None = None) -> None:
    assert ti.cfg is not None
    if not options:
        options = {}
    ti.init(arch=getattr(ti, ti.cfg.arch.name), **options)
