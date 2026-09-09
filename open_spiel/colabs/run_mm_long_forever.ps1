$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "mm_long.log"
$wrapperLog = Join-Path $colabs "mm_long_wrap.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

Write-WLog "wrapper started (pid $PID), resuming MM_long"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "MM_LONG DONE" -Quiet)) {
        Write-WLog "sentinel found -- MM_long complete, exiting wrapper"
        break
    }
    Write-WLog "launching run_mm_long process"
    & $python run_mm_long.py
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
