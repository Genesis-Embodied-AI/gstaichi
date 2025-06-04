$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

pip install .\dist\taichi-1.8.0-cp310-cp310-win_amd64.whl
