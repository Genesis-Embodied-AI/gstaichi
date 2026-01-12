from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ._gstaichi_callable import BoundGsTaichiCallable, GsTaichiCallable
from .kernel_arguments import ArgMetadata

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
        # only needed for args, not kwargs
        self.child_name_by_caller_name_by_func_id: dict[int, dict[str, str]] = defaultdict(dict)

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

        note that this handles both args and kwargs
        """
        if type(func) not in {GsTaichiCallable, BoundGsTaichiCallable}:
            return

        my_func_id = ctx.func.func_id
        called_func_id = func.wrapper.func_id  # type: ignore
        # Copy the used parameters from the child function into our own function.
        called_unpruned = self.used_parameters_by_func_id[called_func_id]
        to_unprune: set[str] = set()
        arg_id = 0
        # node.args ordering will match that of the called function's metas_expanded,
        # because of the way calling with sequential args works.
        # We need to look at the child's declaration - via metas - in order to get the name they use.
        # We can't tell their name just by looking at our own metas.
        for i, arg in enumerate(node.args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                called_name = node.func.ptr.wrapper.arg_metas_expanded[arg_id].name
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1
        # Note that our own arg_metas ordering will in general NOT match that of the child's. That's
        # because our ordering is based on the order in which we pass arguments to the function, but the
        # child's ordering is based on the ordering of their declaration; and these orderings might not
        # match.
        # Luckily, for keywords, we don't need to look at the child's metas, because we can get the
        # child's name directly from our own keyword node.
        for kwarg in node.keywords:
            if hasattr(kwarg.value, "id"):
                calling_name = kwarg.value.id  # type: ignore
                called_name = kwarg.arg
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1
        self.used_parameters_by_func_id[my_func_id].update(to_unprune)

        called_needed = self.used_parameters_by_func_id[called_func_id]
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded
        child_name_by_our_name = self.child_name_by_caller_name_by_func_id[called_func_id]
        for i, arg in enumerate(node.args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                if calling_name.startswith("__ti_"):
                    called_name = child_metas[child_arg_id].name
                    if called_name in called_needed or not called_name.startswith("__ti_"):
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        self.child_name_by_caller_name_by_func_id[called_func_id] = child_name_by_our_name

    def filter_call_args(
        self,
        func: "GsTaichiCallable",
        node: "ast.Call",
        py_args: list[Any],
    ) -> list[Any]:
        """
        used in build_Call, before making the call, in pass 1

        note that this ONLY handles args, not kwargs
        """
        if not (hasattr(func, "wrapper") and hasattr(func.wrapper, "func_id")):
            return py_args

        called_func_id = func.wrapper.func_id  # type: ignore
        called_needed = self.used_parameters_by_func_id[called_func_id]
        new_args = []
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded  # type: ignore
        child_metas_pruned = []
        for _child in child_metas:
            if _child.name.startswith("__ti_"):
                if _child.name in called_needed:
                    child_metas_pruned.append(_child)
            else:
                child_metas_pruned.append(_child)
        child_metas = child_metas_pruned
        for i, arg in enumerate(node.args):
            if hasattr(arg, "id"):
                calling_name = arg.id  # type: ignore
                if calling_name.startswith("__ti_"):
                    called_name = self.child_name_by_caller_name_by_func_id[called_func_id].get(calling_name)
                    if called_name is not None and (
                        called_name in called_needed or not called_name.startswith("__ti_")
                    ):
                        new_args.append(py_args[i])
                else:
                    new_args.append(py_args[i])
            else:
                new_args.append(py_args[i])
            child_arg_id += 1
        py_args = new_args
        return py_args
