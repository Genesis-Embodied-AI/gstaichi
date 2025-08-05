import subprocess
import sys

from tests import test_utils


@test_utils.test()
def test_pyi_stubs(tmpdir):
    test_code = """
import gstaichi._lib.core.gstaichi_python
reveal_type(gstaichi._lib.core.gstaichi_python)
"""
    test_file = tmpdir / "tmp_mypy_test.py"
    test_file.write(test_code)

    res = subprocess.check_output([sys.executable, "-m", "pyright", test_file]).decode("utf-8")
    assert "unknown" not in res.lower()
