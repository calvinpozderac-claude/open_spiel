param([string]$sched)
$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$tag = "ZKL_" + $sched.ToUpper()
$log = Join-Path $colabs "zkl_$sched.log"
$wrapperLog = Join-Path $colabs "zkl_${sched}_wrap.log"
Set-Location $colabs
function Write-WLog($msg) {
    Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
}
Write-WLog "wrapper started (pid $PID) sched=$sched"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "$tag DONE" -Quiet)) {
        Write-WLog "sentinel found -- run complete, exiting wrapper"; break
    }
    Write-WLog "launching run_zkl $sched process"
    & $python run_zkl.py $sched *>> (Join-Path $colabs "zkl_${sched}_stdout.log")
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
