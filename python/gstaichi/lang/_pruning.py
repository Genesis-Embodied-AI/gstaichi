from collections import defaultdict
from typing import TYPE_CHECKING

from ._gstaichi_callable import BoundGsTaichiCallable, GsTaichiCallable

if TYPE_CHECKING:
    import ast

    from .ast.ast_transformer_utils import ASTTransformerFuncContext


class Pruning:
    """
    Note: this assumes when compiling a kernel that each function will only have
    one set of used parameters within the compiled kernel, even if called in multiple
    places.

    In practice however, the implementation DOES work if a function
    is called multiple times within one compilation tree walk pass. What happens is that
    the union of used parameters, across all calls to that function, is stored. And that
    works fine for us, because all we need to ensure is 1. it compiles, 2. the list of
    parameters passed to the kernel is the minimum possible.

    To be clear, there is no restriction that a function needs to have the same set of
    used parameters between kernels, or between calls to the same kernel.

    This assumption allows us to use the func id to uniquely identify each kernel, without
    some additional index based on used parameters or similar.

    Note that we unify handling of func and kernel by using func_id KERNEL_FUNC_ID
    to denote the kernel.
    """

    KERNEL_FUNC_ID = 0

    def __init__(self, kernel_used_parameters: set[str] | None) -> None:
        self.enforcing: bool = False
        self.used_parameters_by_func_id: dict[int, set[str]] = defaultdict(set)
        if kernel_used_parameters is not None:
            self.used_parameters_by_func_id[Pruning.KERNEL_FUNC_ID].update(kernel_used_parameters)

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        assert not self.enforcing
        self.used_parameters_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True

    def is_used(self, func_id: int, parameter_flat_name: str) -> bool:
        return parameter_flat_name in self.used_parameters_by_func_id[func_id]

    def record_after_call(self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node: "ast.Call") -> None:
        """
        called from build_Call, after making the call, in pass 0
        """
        if type(func) not in {GsTaichiCallable, BoundGsTaichiCallable}:
            return

        _my_func_id = ctx.func.func_id
        _called_func_id = func.wrapper.func_id  # type: ignore

        # Copy the used parameters from the child function into our own function.
        called_unpruned = self.used_parameters_by_func_id[_called_func_id]
        to_unprune: set[str] = set()
        arg_id = 0
        for kwarg in node.keywords:
            if hasattr(kwarg.value, "id"):
                calling_name = kwarg.value.id  # type: ignore
                called_name = kwarg.arg
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1
        self.used_parameters_by_func_id[_my_func_id].update(to_unprune)
