$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$trainingLog = Join-Path $colabs "adaptive_500family_sweep.log"
$watchLog = Join-Path $colabs "watch_tournament.log"

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WATCHER: $msg"
    Add-Content -Path $watchLog -Value $line
}

Write-WLog "watcher started (pid $PID) -- polling for 500family completion, NOT touching the running training task"
while ($true) {
    if ((Test-Path $trainingLog) -and (Select-String -Path $trainingLog -Pattern "500family sweep complete" -Quiet)) {
        Write-WLog "500family sweep complete -- launching final report + tournament task"
        break
    }
    Start-Sleep -Seconds 60
}

$taskName = "OpenSpielFinalReportTournament"
$scriptPath = Join-Path $colabs "run_final_report_and_tournament_forever.ps1"
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-WLog "final report + tournament task started -- watcher done"
