from typing import Type


def raise_exception(
    ExceptionClass: Type[Exception], msg: str, category: str, err_code: str, orig: Exception | None = None
):
    err = ExceptionClass(f"[{category}] {msg} (error code: {err_code}).")
    raise err from orig if orig else err
