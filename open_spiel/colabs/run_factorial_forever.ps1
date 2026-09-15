param([string]$cell)
$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$tag = "ALLMSE_" + $cell.ToUpper()
$log = Join-Path $colabs "fact_$cell.log"
$wrapperLog = Join-Path $colabs "fact_${cell}_wrap.log"
Set-Location $colabs
function Write-WLog($msg) { Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg" }
Write-WLog "wrapper started (pid $PID) cell=$cell"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "$tag DONE" -Quiet)) { Write-WLog "sentinel found -- exiting"; break }
    Write-WLog "launching run_factorial $cell"
    & $python run_factorial.py $cell *>> (Join-Path $colabs "fact_${cell}_stdout.log")
    Write-WLog "process exited ($LASTEXITCODE) -- re-checking after pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
