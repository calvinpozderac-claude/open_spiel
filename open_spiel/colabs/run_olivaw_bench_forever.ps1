$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$trainLog = Join-Path $colabs "olivaw.log"
$wrapperLog = Join-Path $colabs "olivaw_bench_wrap.log"
Set-Location $colabs

# mm_long_bench.py reads these; they must be in the ENVIRONMENT rather than
# patched module globals, because its worker pool re-imports the module under
# Windows 'spawn' and would otherwise score bench_MM_long's checkpoints.
$env:BENCH_CKPT_DIR = "othello_benchmark/bench_OLIVAW"
$env:BENCH_SNAP_DIR = "othello_benchmark/bench_OLIVAW_snaps"
$env:BENCH_EVERY    = "5000"
$env:BENCH_RESULTS  = "olivaw_bench.json"
$env:BENCH_LOG      = "olivaw_bench.log"

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

Write-WLog "bench wrapper started (pid $PID), every $($env:BENCH_EVERY) eps"
while ($true) {
    & $python mm_long_bench.py --once *>> (Join-Path $colabs "olivaw_bench_stdout.log")
    if ($LASTEXITCODE -ne 0) { Write-WLog "bench exited $LASTEXITCODE -- retrying next cycle" }
    if ((Test-Path $trainLog) -and (Select-String -Path $trainLog -Pattern "OLIVAW DONE" -Quiet)) {
        Write-WLog "training complete and all snapshots scored -- exiting"
        break
    }
    # 15 min, not 5: snapshots land every 5000 episodes (~3 h apart), and each
    # poll pays a full torch + OpenSpiel import before finding nothing to do.
    Start-Sleep -Seconds 900
}
Write-WLog "bench wrapper done"
