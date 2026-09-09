$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$trainLog = Join-Path $colabs "mm_long.log"
$wrapperLog = Join-Path $colabs "mm_long_bench_wrap.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

# --once per launch, re-launched on a 300s cadence. A crash inside a match
# (which is what silently stopped benchmarking on Aug 21 at 18:56) costs one
# cycle instead of the rest of the run.
Write-WLog "bench wrapper started (pid $PID)"
while ($true) {
    & $python mm_long_bench.py --once *>> (Join-Path $colabs "mm_long_bench_stdout.log")
    if ($LASTEXITCODE -ne 0) {
        Write-WLog "mm_long_bench.py exited $LASTEXITCODE -- retrying next cycle"
    }
    if ((Test-Path $trainLog) -and (Select-String -Path $trainLog -Pattern "MM_LONG DONE" -Quiet)) {
        Write-WLog "training complete and all snapshots scored -- exiting wrapper"
        break
    }
    Start-Sleep -Seconds 300
}
Write-WLog "bench wrapper done"
