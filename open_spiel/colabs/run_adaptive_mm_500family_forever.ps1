$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "adaptive_500family_sweep.log"
$wrapperLog = Join-Path $colabs "adaptive_500family_wrapper.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "500family sweep complete" -Quiet)) {
        Write-WLog "sentinel found in log -- sweep is complete, exiting wrapper"
        break
    }
    Write-WLog "launching training process"
    & $python run_adaptive_mm_500family.py
    Write-WLog "training process exited with code $LASTEXITCODE -- will re-check/relaunch after a short pause"
    Start-Sleep -Seconds 5
}

# Chain into the final report + round-robin tournament, same durable
# Scheduled Task mechanism as the previous hand-off.
Write-WLog "registering follow-up task OpenSpielFinalReportTournament"
$taskName = "OpenSpielFinalReportTournament"
$scriptPath = Join-Path $colabs "run_final_report_and_tournament_forever.ps1"
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-WLog "follow-up task started"

Write-WLog "wrapper done"
