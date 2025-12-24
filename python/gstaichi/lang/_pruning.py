from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .kernel_arguments import ArgMetadata

if TYPE_CHECKING:
    import ast

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
        func_id None means kernel
        """
        assert not self.enforcing
        self.used_parameters_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True
        self._calc_dotted()

    def is_used(self, func_id: int, parameter_flat_name: str) -> bool:
        return parameter_flat_name in self.used_parameters_by_func_id[func_id]

    def _calc_dotted(self) -> None:
        assert self.enforcing
        dotted_by_func_id = {}
        for func_id, used_parameters in self.used_parameters_by_func_id.items():
            dotted_by_func_id[func_id] = set([tuple(p.split("__ti_")[1:]) for p in used_parameters])
        self.dotted_by_func_id = dotted_by_func_id

    def record_after_call(self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node) -> None:
        """
        called from build_Call, after making the call, in pass 0

        note that this ONLY handles args, not kwargs
        """
        if not hasattr(func, "wrapper"):
            return

        arg_id = 0
        _my_func_id = ctx.func.func_id
        _called_func_id = func.wrapper.func_id  # type: ignore
        func_id = func.wrapper.func_id  # type: ignore
        called_unpruned = self.used_parameters_by_func_id[_called_func_id]
        to_unprune: set[str] = set()
        for i, arg in enumerate(node.args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                called_name = node.func.ptr.wrapper.arg_metas_expanded[arg_id].name
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1
        for arg in node.keywords:
            if hasattr(arg.value, "id"):
                calling_name = arg.value.id
                called_name = node.func.ptr.wrapper.arg_metas_expanded[arg_id].name
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
            arg_id += 1

        self.used_parameters_by_func_id[_my_func_id].update(to_unprune)

        called_needed = self.used_parameters_by_func_id[_called_func_id]
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded
        child_name_by_our_name = self.child_name_by_caller_name_by_func_id[func_id]
        for i, arg in enumerate(node.args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                if calling_name.startswith("__ti_"):
                    called_name = child_metas[child_arg_id].name
                    if called_name in called_needed or not called_name.startswith("__ti_"):
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        for i, arg in enumerate(node.keywords):
            if hasattr(arg, "id"):
                calling_name = arg.value.id
                if calling_name.startswith("__ti_"):
                    called_name = arg.arg
                    if called_name in called_needed:
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        self.child_name_by_caller_name_by_func_id[func_id] = child_name_by_our_name

    def filter_call_args(
        self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node: "ast.Call", py_args: list[Any]
    ) -> list[Any]:
        """
        used in build_Call, before making the call, in pass 1

        note that this ONLY handles args, not kwargs
        """
        if not (hasattr(func, "wrapper") and hasattr(func.wrapper, "func_id")):
            return py_args

        _called_func_id = func.wrapper.func_id  # type: ignore
        func_id = func.wrapper.func_id  # type: ignore
        called_needed = self.used_parameters_by_func_id[_called_func_id]
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
                    called_name = self.child_name_by_caller_name_by_func_id[func_id].get(calling_name)
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
