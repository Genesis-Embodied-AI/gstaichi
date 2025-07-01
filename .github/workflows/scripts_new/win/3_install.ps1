$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

$whl = Get-ChildItem .\dist\*.whl | Select-Object -First 1
pip install $whl
