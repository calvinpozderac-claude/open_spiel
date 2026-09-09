$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "p1_cev.log"
$wrapperLog = Join-Path $colabs "p1_cev_wrap.log"
Set-Location $colabs
function Write-WLog($msg) {
    Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
}
Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "P1CEV DONE" -Quiet)) {
        Write-WLog "sentinel found -- run complete, exiting wrapper"; break
    }
    Write-WLog "launching run_p1_cev process"
    & $python run_p1_cev.py *>> (Join-Path $colabs "p1_cev_stdout.log")
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
