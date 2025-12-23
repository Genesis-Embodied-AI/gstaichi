from collections import defaultdict


class Pruning:
    """
    Note: this assumes when compiling a kernel that each function will only have
    one set of used parameters within the compiled kernel, even if called in multiple
    places.

    To be clear, there is no restriction that a function needs to have the same set of
    used parameters between kernels, or between calls to the same kernel.

    This assumption allows us to use the func id to uniquely identify each kernel, without
    some additional index based on used parameters or similar.

    Note that we unify handling of func and kernel by using func_id -1 to denote kernel.
    """

    def __init__(self, kernel_used_parameters: set[str] | None) -> None:
        self.enforcing: bool = False
        # func_id -1 means kernel
        self.used_parameters_by_func_id: dict[int, set[str]] = defaultdict(set)
        self.dotted_by_func_id: dict[int, tuple[str, ...]] | None = None
        if kernel_used_parameters is not None:
            self.used_parameters_by_func_id[-1].update(kernel_used_parameters)
        # self.caller_name_by_child_name_by_func_id: dict[int, dict[str, str]] = defaultdict(dict)
        self.child_name_by_caller_name_by_func_id: dict[int, dict[str, str]] = defaultdict(dict)

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        """
        func_id None means kernel
        """
        assert not self.enforcing
        self.used_parameters_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True
        self._calc_dotted()

    def is_used(self, func_id: int, parameter_flat_name: str) -> bool:
        return parameter_flat_name in self.used_parameters_by_func_id[func_id]

    def _calc_dotted(self):
        assert self.enforcing
        dotted_by_func_id = {}
        for func_id, used_parameters in self.used_parameters_by_func_id.items():
            dotted_by_func_id[func_id] = set([tuple(p.split("__ti_")[1:]) for p in used_parameters])
        self.dotted_by_func_id = dotted_by_func_id
