def create_flat_name(basename: str, child_name: str) -> str:
    """
    Appends child_name to basename, separated by __ti_.
    If basename does not start with __ti_ then prefix the resulting string
    with __ti_.

    Note that we want to avoid adding prefix __ti_ if already included in `basename`,
    to avoid duplicating said delimiter.

    We'll use this when expanding py dataclass members, e.g.

    @dataclasses.dataclass
    def Foo:
        a: int
        b: int

    foo = Foo(a=5, b=3)

    When we expand out foo, we'll replace foo with the following names instead:
    - __ti_foo__ti_a
    - __ti_foo__ti_b

    We use the __ti_ to ensure that it's easy to ensure no collision with existing user-defined
    names. We require the user to not create any fields or variables which themselves are prefixed
    with __ti_, and given this constraint, the names we create will not conflict with user-generated
    names.
    """
    full_name = f"{basename}__ti_{child_name}"
    if not full_name.startswith("__ti_"):
        full_name = f"__ti_{full_name}"
    return full_name
