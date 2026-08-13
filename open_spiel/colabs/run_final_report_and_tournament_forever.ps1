$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "final_report_tournament.log"
$wrapperLog = Join-Path $colabs "final_report_tournament_wrapper.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "ALL DONE" -Quiet)) {
        Write-WLog "sentinel found in log -- everything is complete, exiting wrapper"
        break
    }
    Write-WLog "launching report/tournament process"
    & $python run_final_report_and_tournament.py
    Write-WLog "process exited with code $LASTEXITCODE -- will re-check/relaunch after a short pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done -- experiment fully complete"
