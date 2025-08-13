from typing import Any

import gstaichi as ti


def ti_init_same_arch(**options) -> None:
    """
    Used in tests to call ti.init, passing in the same arch as currently
    configured. Since it's fairly fiddly to do that, extracting this out
    to this helper function.
    """
    assert ti.cfg is not None
    options["arch"] = getattr(ti, ti.cfg.arch.name)
    ti.init(**options)
