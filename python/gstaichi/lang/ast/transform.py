# type: ignore

from gstaichi.lang.ast.ast_transformer import ASTTransformer
from gstaichi.lang.ast.ast_transformer_utils import ASTTransformerContext


def transform_tree(tree, ctx: ASTTransformerContext):
    ASTTransformer()(ctx, tree)
    return ctx.return_data
