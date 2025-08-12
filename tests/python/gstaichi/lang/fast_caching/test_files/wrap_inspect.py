# from 5 to 8, 4 lines
def my_func(a: int, b: int) -> int:
    a += 5
    b += a
    return a + b


class Foo:
    # from 13 to 16, 4 lines
    def my_func2(self, a: int, b: int) -> int:
        a += 5
        b += a
        return a + b
