$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$trainLog = Join-Path $colabs "sample.log"
$wrapperLog = Join-Path $colabs "sample_bench_wrap.log"
Set-Location $colabs
$env:BENCH_CKPT_DIR = "othello_benchmark/bench_SAMPLE"
$env:BENCH_SNAP_DIR = "othello_benchmark/bench_SAMPLE_snaps"
$env:BENCH_EVERY    = "5000"
$env:BENCH_RESULTS  = "sample_bench.json"
$env:BENCH_LOG      = "sample_bench.log"
function Write-WLog($msg) {
    Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
}
Write-WLog "bench wrapper started (pid $PID), every $($env:BENCH_EVERY) eps"
while ($true) {
    & $python mm_long_bench.py --once *>> (Join-Path $colabs "sample_bench_stdout.log")
    if ($LASTEXITCODE -ne 0) { Write-WLog "bench exited $LASTEXITCODE -- retrying next cycle" }
    if ((Test-Path $trainLog) -and (Select-String -Path $trainLog -Pattern "SAMPLE DONE" -Quiet)) {
        Write-WLog "training complete and all snapshots scored -- exiting"; break
    }
    Start-Sleep -Seconds 900
}
Write-WLog "bench wrapper done"
