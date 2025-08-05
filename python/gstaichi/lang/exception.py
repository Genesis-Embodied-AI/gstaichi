# type: ignore

from gstaichi._lib import core


class GsTaichiCompilationError(Exception):
    """Base class for all compilation exceptions."""

    pass


class GsTaichiSyntaxError(GsTaichiCompilationError, SyntaxError):
    """Thrown when a syntax error is found during compilation."""

    pass


class GsTaichiNameError(GsTaichiCompilationError, NameError):
    """Thrown when an undefine name is found during compilation."""

    pass


class GsTaichiIndexError(GsTaichiCompilationError, IndexError):
    """Thrown when an index error is found during compilation."""

    pass


class GsTaichiTypeError(GsTaichiCompilationError, TypeError):
    """Thrown when a type mismatch is found during compilation."""

    pass


class GsTaichiRuntimeError(RuntimeError):
    """Thrown when the compiled program cannot be executed due to unspecified reasons."""

    pass


class GsTaichiAssertionError(GsTaichiRuntimeError, AssertionError):
    """Thrown when assertion fails at runtime."""

    pass


class GsTaichiRuntimeTypeError(GsTaichiRuntimeError, TypeError):
    @staticmethod
    def get(pos, needed, provided):
        return GsTaichiRuntimeTypeError(
            f"Argument {pos} (type={provided}) cannot be converted into required type {needed}"
        )

    @staticmethod
    def get_ret(needed, provided):
        return GsTaichiRuntimeTypeError(f"Return (type={provided}) cannot be converted into required type {needed}")


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.GsTaichiTypeError):
        return GsTaichiTypeError(str(exc))
    if isinstance(exc, core.GsTaichiSyntaxError):
        return GsTaichiSyntaxError(str(exc))
    if isinstance(exc, core.GsTaichiIndexError):
        return GsTaichiIndexError(str(exc))
    if isinstance(exc, core.GsTaichiAssertionError):
        return GsTaichiAssertionError(str(exc))
    return exc


__all__ = [
    "GsTaichiSyntaxError",
    "GsTaichiTypeError",
    "GsTaichiCompilationError",
    "GsTaichiNameError",
    "GsTaichiRuntimeError",
    "GsTaichiRuntimeTypeError",
    "GsTaichiAssertionError",
]
