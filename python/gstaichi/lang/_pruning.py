from collections import defaultdict
from typing import TYPE_CHECKING

from ._gstaichi_callable import BoundGsTaichiCallable, GsTaichiCallable

if TYPE_CHECKING:
    import ast

    from .ast.ast_transformer_utils import ASTTransformerFuncContext


class Pruning:
    """
    We use the func id to uniquely identify each function.

    Thus, each function has a single set of used parameters associated with it, within
    a single call to a single kernel. When the same function is called multiple times
    within the same call, to the same kernel, then the used parameters for that function
    will be the union over the parameters used by each call to that function.

    A function can have different used parameters parameters between kernels, and
    between different calls to the same kernel.

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

        my_func_id = ctx.func.func_id
        called_func_id = func.wrapper.func_id  # type: ignore

        # Copy the used parameters from the child function into our own function.
        called_unpruned = self.used_parameters_by_func_id[called_func_id]
        to_unprune: set[str] = set()
        arg_id = 0
        for kwarg in node.keywords:
            if hasattr(kwarg.value, "id"):
                calling_name = kwarg.value.id  # type: ignore
                called_name = kwarg.arg
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1
        self.used_parameters_by_func_id[my_func_id].update(to_unprune)
