#!/bin/bash

echo hello from win.sh
systeminfo
systeminfo
Get-Location
Get-ChildItem
python -V
python --version

git submodule update --init --recursive

Start-Process -NoNewWindow -FilePath "python" -ArgumentList "build.py" -ErrorAction SilentlyContinue -Wait
# try {
#     python myscript.py
#     if ($LASTEXITCODE -ne 0) { throw "Python script failed with exit code $LASTEXITCODE" }
# } catch {
#     Write-Host "Python script failed, but we're continuing anyway: $_"
# }
python build.py
pip install .\dist\taichi-1.8.0-cp310-cp310-win_amd64.whl

python -c 'import taichi as ti; ti.init();'

pip install -r requirements_test.txt
python .\tests\run_tests.py -v
