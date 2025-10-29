import gstaichi as ti

from tests import test_utils


@test_utils.test()
def test_scalar():
    n = 16

    print('create x')
    x = ti.field(ti.i32, shape=n)
    print('create y')
    y = ti.field(ti.i32, shape=n)

    print('x[1] = 2')
    x[1] = 2
    x[10] = 123

    print('y[0] = 1')
    y[0] = 1
    print('y[2] = 3')
    y[2] = 3
    y[11] = 222

    print("x", x.to_numpy())
    print("y", y.to_numpy())

    print('launching copy from')
    x.copy_from(y)

    print("x", x.to_numpy())
    print("y", y.to_numpy())

    print('x[0]')
    assert x[0] == 1
    print('x[1]')
    assert x[1] == 0
    print('x[2]')
    assert x[2] == 3

    assert y[0] == 1
    assert y[1] == 0
    assert y[2] == 3


@test_utils.test()
def test_struct():
    @ti.dataclass
    class C:
        i: int
        f: float

    n = 16

    x = C.field(shape=n)
    y = C.field(shape=n)

    x[1].i = 2
    x[2].i = 4

    y[0].f = 1.0
    y[2].i = 3

    x.copy_from(y)

    assert x[0].f == 1.0
    assert x[1].i == 0
    assert x[2].i == 3

    assert y[0].f == 1.0
    assert y[1].i == 0
    assert y[2].i == 3
