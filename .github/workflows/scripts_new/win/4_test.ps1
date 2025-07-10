$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

python -c 'import taichi as ti; ti.init();'
pip install -r requirements_test.txt
python .\tests\run_tests.py -v -r 3
