# Unsupported python notations

## Argument unpacking

i.e. syntax like `some_function(*args)`

Only supported in a very narrow case:
- `*args` must be the last argument in the function call
- `*args` must not contain any `dataclasses.dataclass` objects
