$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

pip install .\dist\*.whl
