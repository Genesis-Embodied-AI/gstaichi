def create_flat_name(basename: str, child_name: str) -> str:
    full_name = f"{basename}__ti_{child_name}"
    if not full_name.startswith("__ti_"):
        full_name = f"__ti_{full_name}"
    return full_name
