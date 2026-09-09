$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$log = Join-Path $colabs "mixture_floor3.log"
$wrapperLog = Join-Path $colabs "mixture_floor3_wrap.log"
Set-Location $colabs
function Write-WLog($msg) {
    Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
}
Write-WLog "wrapper started (pid $PID)"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "MIXTURE_FLOOR3 DONE" -Quiet)) {
        Write-WLog "sentinel found -- run complete, exiting wrapper"; break
    }
    Write-WLog "launching run_mixture_floor3 process"
    & $python run_mixture_floor3.py *>> (Join-Path $colabs "mixture_floor3_stdout.log")
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
