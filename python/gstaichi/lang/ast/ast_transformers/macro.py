import ast


class macro:
    def __init__(self, contents: str) -> None:
        self.contents_str = contents
        self.contents_ast = ast.parse(self.contents_str)
        print(ast.dump(self.contents_ast, indent=2))
    
    def __call__(self, *args, **kwargs) -> None:
        print("macro.call", self.contents_str, "******************", args, kwargs)
        return "ABCDEFG"
    

def macro2(contents: str):
    print("macro2 contents", contents)
    contents_ast = ast.parse(contents)
    return contents_ast
