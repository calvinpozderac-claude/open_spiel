$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "az_match.log"
$wrapperLog = Join-Path $colabs "az_match_wrap.log"
Set-Location $colabs
function Write-WLog($msg) { Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg" }
Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "AZ_MATCH DONE" -Quiet)) { Write-WLog "sentinel found -- exiting"; break }
    Write-WLog "launching run_az"
    & $python run_az.py *>> (Join-Path $colabs "az_match_stdout.log")
    Write-WLog "process exited ($LASTEXITCODE) -- pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
