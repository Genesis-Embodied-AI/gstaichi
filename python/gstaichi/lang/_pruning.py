from collections import defaultdict
from typing import TYPE_CHECKING

from .kernel_arguments import ArgMetadata

if TYPE_CHECKING:

    from ._gstaichi_callable import GsTaichiCallable
    from .ast.ast_transformer_utils import ASTTransformerFuncContext


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
        self.child_name_by_caller_name_by_func_id: dict[int, dict[str, str]] = defaultdict(dict)

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        """
        func_id -1 means kernel
        """
        assert not self.enforcing
        self.used_parameters_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True
        self._calc_dotted()

    def is_used(self, func_id: int, parameter_flat_name: str) -> bool:
        return parameter_flat_name in self.used_parameters_by_func_id[func_id]

    def _calc_dotted(self) -> None:
        """
        There are two formats we need:
        - the internal variable name, like "__ti_struct__ti_some_member"
            - desigend to not conflict with customer variable names
            - cannot contain "."
        - tuple of str, like ("struct", "some_member")
            - named "dotted", for historical reasons. We should probably rename...

        The latter is called "dotted" because the format used to be "struct.some_member", but
        was changed to tuple notation. It's used in _recursive_set_args method, as parameter
        used_py_dataclass_parameters.

        For speed we pre-calculate dotted here.
        """
        assert self.enforcing
        dotted_by_func_id = {}
        for func_id, used_parameters in self.used_parameters_by_func_id.items():
            dotted_by_func_id[func_id] = set([tuple(p.split("__ti_")[1:]) for p in used_parameters])
        self.dotted_by_func_id = dotted_by_func_id

    def record_after_call(
        self,
        ctx: "ASTTransformerFuncContext",
        func: "GsTaichiCallable",
        node,
    ) -> None:
        """
        called from build_Call, after making the call, in pass 0
        """
        if not hasattr(func, "wrapper"):
            return

        _my_func_id = ctx.func.func_id
        _called_func_id = func.wrapper.func_id  # type: ignore
        func_id = func.wrapper.func_id  # type: ignore

        # Copy the used parameters from the child function into our own function.
        called_unpruned = self.used_parameters_by_func_id[_called_func_id]
        to_unprune: set[str] = set()
        arg_id = 0
        for arg in node.keywords:
            if hasattr(arg.value, "id"):
                calling_name = arg.value.id
                called_name = arg.arg
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1

        self.used_parameters_by_func_id[_my_func_id].update(to_unprune)

        # Store the mapping between parameter names in our namespace, and in the called function
        # namespace
        called_needed = self.used_parameters_by_func_id[_called_func_id]
        child_arg_id = 0
        child_name_by_our_name = self.child_name_by_caller_name_by_func_id[func_id]
        for arg in node.keywords:
            if hasattr(arg, "id"):
                calling_name = arg.value.id
                if calling_name.startswith("__ti_"):
                    called_name = arg.arg
                    if called_name in called_needed:
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        self.child_name_by_caller_name_by_func_id[func_id] = child_name_by_our_name
