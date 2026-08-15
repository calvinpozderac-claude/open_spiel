$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "temp_sweep.log"
$wrapperLog = Join-Path $colabs "temp_sweep_wrapper.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "TEMP SWEEP DONE" -Quiet)) {
        Write-WLog "sentinel found -- sweep complete, exiting wrapper"
        break
    }
    Write-WLog "launching temp sweep"
    & $python run_temp_sweep.py
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
