from tests import test_utils
import tempfile
import sys
import os
import subprocess


@test_utils.test()
def test_pyi_stubs():
    test_code = """
import taichi._lib.core.taichi_python
reveal_type(taichi._lib.core.taichi_python)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "tmp_mypy_test.py")
        with open(test_file, "w") as f:
            f.write(test_code)

        # if no types, this will fail
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "mypy",
                test_file,
            ]
        )
