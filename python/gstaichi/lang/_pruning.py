from collections import defaultdict
from typing import Any, TYPE_CHECKING

from .kernel_arguments import ArgMetadata

if TYPE_CHECKING:
    from .ast.ast_transformer_utils import ASTTransformerFuncContext
    from ._gstaichi_callable import GsTaichiCallable
    import ast
    from ast import keyword


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

    def record_after_call(self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node, node_args, node_keywords) -> None:
        """
        called from build_Call, after making the call, in pass 0

        note that this ONLY handles args, not kwargs
        """
        if not hasattr(func, "wrapper"):
            return

        func_name = func.fn.__name__
        ctx.debug('record_after_call()', func_name)

        _my_func_id = ctx.func.func_id
        _called_func_id = func.wrapper.func_id  # type: ignore
        func_id = func.wrapper.func_id  # type: ignore
        called_unpruned = self.used_parameters_by_func_id[_called_func_id]
        ctx.debug("called_unpruned:")
        for name in sorted(called_unpruned):
            if ctx.filter_name(name):
                ctx.debug("-", name)
        to_unprune: set[str] = set()
        ctx.debug("child arg metas expanded:")
        for i, arg_meta in enumerate(node.func.ptr.wrapper.arg_metas_expanded):
            # if ctx.filter_name(arg_meta.name):
            ctx.debug("- arg meta idx", i, "name", arg_meta.name)
        ctx.debug("(after child arg metas expanded)")
        ctx.debug("")
        # ctx.debug("_pruning record after call node.args")
        # for i, arg in enumerate(node.args):
        #     ctx.debug("   -", i, getattr(arg, "id", "<no id>"), "=>", node.func.ptr.wrapper.arg_metas_expanded[i].name)
        # ctx.debug("(after _pruning record after call node.args)")
        # ctx.debug("_pruning record after call node.keywords")
        # for i, arg in enumerate(node.keywords):
        #     ctx.debug("   -", i, getattr(arg.value, "id", "<no id>"), "=>", node.func.ptr.wrapper.arg_metas_expanded[i + len(node.args)].name)
        # ctx.debug("(after _pruning record after call node.keywords)")
        # ctx.debug("")
        ctx.debug("unpruning:")
        arg_id = 0
        # node.args ordering will match that of the called function's metas_expanded,
        # because of the way calling with sequential args works.
        # We need to look at the child's declaration - via metas - in order to get the name they use.
        # We can't tell their name just by looking at our own metas.
        #
        # One issue is when calling data-oriented methods, there will be a `self`. We'll detect this
        # by seeing if the childs arg_metas_expanded is exactly 1 longer than len(node.args) + len(node.kwargs)
        ctx.debug("len node.args", len(node_args), "len node.keywords", len(node_keywords), "len child metas", len(node.func.ptr.wrapper.arg_metas_expanded))
        has_self = len(node_args) + len(node_keywords) + 1 == len(node.func.ptr.wrapper.arg_metas_expanded)
        ctx.debug("has self", has_self)
        self_offset = 1 if has_self else 0
        for i, arg in enumerate(node_args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                called_name = node.func.ptr.wrapper.arg_metas_expanded[arg_id + self_offset].name
                if called_name in sorted(called_unpruned):
                    if ctx.filter_name(calling_name):
                        import ast
                        ctx.debug("- unpruning arg id", arg_id, calling_name, "=>", called_name, ast.dump(arg))
                    to_unprune.add(calling_name)
            arg_id += 1
        # Note that our own arg_metas ordering will in general NOT match that of the child's. That's
        # because our ordering is based on the order in which we pass arguments to the function, but the
        # child's ordering is based on the ordering of their declaration; and these orderings might not
        # match.
        # Luckily, for keywords, we don't need to look at the child's metas, because we can get the
        # child's name directly from our own keyword node.
        for arg in node_keywords:
            if hasattr(arg.value, "id"):
                calling_name = arg.value.id
                called_name = arg.arg
                if called_name in called_unpruned:
                    to_unprune.add(calling_name)
                    if ctx.filter_name(calling_name):
                        ctx.debug("- unpruning keyword arg_id", arg_id, calling_name, "=>", called_name)
            arg_id += 1
        ctx.debug("(after unpruning)")

        self.used_parameters_by_func_id[_my_func_id].update(to_unprune)
        ctx.debug("record after call, used_parameters:")
        for param in sorted(self.used_parameters_by_func_id[_my_func_id]):
            if ctx.filter_name(param):
                ctx.debug("-", param)

        called_needed = self.used_parameters_by_func_id[_called_func_id]
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded
        child_name_by_our_name = self.child_name_by_caller_name_by_func_id[func_id]
        for i, arg in enumerate(node_args):
            if hasattr(arg, "id"):
                calling_name = arg.id
                if calling_name.startswith("__ti_"):
                    called_name = child_metas[child_arg_id + self_offset].name
                    if called_name in called_needed or not called_name.startswith("__ti_"):
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        for i, arg in enumerate(node_keywords):
            if hasattr(arg, "id"):
                calling_name = arg.value.id
                if calling_name.startswith("__ti_"):
                    called_name = arg.arg
                    if called_name in called_needed:
                        child_name_by_our_name[calling_name] = called_name
            child_arg_id += 1
        # ctx.debug("child name by our name")
        # for our_name, child_name in child_name_by_our_name.items():
        #     if ctx.filter_name(our_name) or ctx.filter_name(child_name):
        #         ctx.debug("-", our_name, "=>", child_name)
        # ctx.debug("(after child name by our name)")
        self.child_name_by_caller_name_by_func_id[func_id] = child_name_by_our_name
        # ctx.debug("record after call", func.wrapper.func, "child_name_by_our_name", child_name_by_our_name)
        ctx.debug("record after call, child_name_by_our_name:")
        for our_name, child_name in sorted(child_name_by_our_name.items()):
            if ctx.filter_name(our_name):
                ctx.debug('- ', our_name, '=>', child_name)
        ctx.debug('(after record_after_call()', func_name, ")")

    def filter_call_args(self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node: "ast.Call", node_args, node_keywords, py_args: list[Any]) -> list[Any]:
        """
        used in build_Call, before making the call, in pass 1

        note that this ONLY handles args, not kwargs
        """
        if not (hasattr(func, "wrapper") and hasattr(func.wrapper, "func_id")):
            return py_args

        # _pruning = ctx.global_context.pruning
        _called_func_id = func.wrapper.func_id  # type: ignore
        func_id = func.wrapper.func_id  # type: ignore
        called_needed = self.used_parameters_by_func_id[_called_func_id]
        new_args = []
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded  # type: ignore
        child_metas_pruned = []
        # ctx.debug("filter call args", ctx.func.func, "called needed", sorted(list(called_needed)))
        func_name = func.fn.__name__
        ctx.debug('filter_call_args()', func_name)
        ctx.debug("filter call args called needed")
        for needed in sorted(called_needed):
            if ctx.filter_name(needed):
                ctx.debug("- ", needed)
        ctx.debug("filter call args, child_name_by_our_name:")
        for our_name, child_name in sorted(self.child_name_by_caller_name_by_func_id[func_id].items()):
            if ctx.filter_name(our_name):
                ctx.debug('- ', our_name, '=>', child_name)
        for _child in child_metas:
            if _child.name.startswith("__ti_"):
                if _child.name in called_needed:
                    child_metas_pruned.append(_child)
            else:
                child_metas_pruned.append(_child)
        child_metas = child_metas_pruned
        ctx.debug("enumerating node.args before call:")
        for i, arg in enumerate(node_args):
            import ast
            dumped_arg = ast.dump(arg)[:80]
            dump = ctx.filter_name(dumped_arg)
            is_starred = type(arg) is ast.Starred
            # Starred arguments are all just lumped together into the ptr for the Starred
            # node. We'll just pass them through.
            # we'll forbid py dataclasses in *args.
            # Also, let's require any *starred at the end of the parameters
            # (which is consistent with test_utils.test_utils_geom_taichi_vs_tensor_consistency)
            ctx.debug("is_starred", is_starred)
            if is_starred:
                assert i == len(node.args) - 1 and len(node_keywords) == 0
                # we'll just dump the rest of the py_args in:
                new_args.extend(py_args[i:])
                # new_args.append(py_args[i])
                child_arg_id += len(py_args[i:])
                break
            if dump:
                ctx.debug("-", i, ast.dump(arg)[:50])
            if hasattr(arg, "id"):
                # if dump:
                #     ctx.debug(".  => has id")
                calling_name = arg.id  # type: ignore
                if calling_name.startswith("__ti_"):
                    called_name = self.child_name_by_caller_name_by_func_id[func_id].get(calling_name)
                    if dump:
                        ctx.debug("    => ", called_name)
                    if called_name is not None and (
                        called_name in called_needed or not called_name.startswith("__ti_")
                    ):
                        new_args.append(py_args[i])
                else:
                    new_args.append(py_args[i])
            else:
                # if dump:
                #     ctx.debug("   => NO id")
                new_args.append(py_args[i])
            child_arg_id += 1
        py_args = new_args
        ctx.debug('(end filter_call_args()', func_name, ")")
        return py_args

    # def filter_keywords(self, ctx: "ASTTransformerFuncContext", func: "GsTaichiCallable", node: "ast.Call", added_keywords: "list[keyword]") -> "list[keyword]":
    #     """
    #     Filter results of expand dataclasses, in build_Call
    #     """
    #     ctx.debug("filter_call_kwargs")

    #     if not (hasattr(func, "wrapper") and hasattr(func.wrapper, "func_id")):
    #         ctx.debug("doesnt have wrapper or func_id")
    #         return keywords

    #     _called_func_id = func.wrapper.func_id  # type: ignore
    #     func_id = func.wrapper.func_id  # type: ignore
    #     called_needed = self.used_parameters_by_func_id[_called_func_id]

    #     ctx.debug("filter call args called needed")
    #     for needed in sorted(called_needed):
    #         ctx.debug("- ", needed)
    #     ctx.debug("filter call args, child_name_by_our_name:")
    #     for our_name, child_name in sorted(self.child_name_by_caller_name_by_func_id[func_id].items()):
    #         ctx.debug('- ', our_name, '=>', child_name)

    #     ctx.debug("keywords")
    #     indent = "  "
    #     pruned_keywords = []
    #     for keyword in keywords:
    #         import ast
    #         child_name = keyword.arg
    #         our_name = keyword.value.id
    #         ctx.debug(indent, "-", our_name, "->", child_name, ast.dump(keyword))
    #         if child_name in called_needed:
    #             pruned_keywords.append(keyword)
    #         # child_name = 

    #     return pruned_keywords

    #     # pruned_py_kwargs = {}
    #     # indent = "  "
    #     # for name, kwarg in py_kwargs.items():
    #     #     ctx.debug(indent, "-", name, kwarg)
    #     #     pruned_py_kwargs[name] = kwarg
    #     # return pruned_py_kwargs
