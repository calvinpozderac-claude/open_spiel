$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$trainLog = Join-Path $colabs "mm_long.log"
$wrapperLog = Join-Path $colabs "eval_set_wrap.log"
Set-Location $colabs

function Write-WLog($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] EVALSET: $msg"
    Add-Content -Path $wrapperLog -Value $line
}

# Wait for training to finish before touching the CPU -- the build is ~2 hours
# of Edax playouts and the run's last episodes should not be competing with it.
Write-WLog "waiting for MM_LONG DONE (pid $PID)"
while ($true) {
    if ((Test-Path $trainLog) -and (Select-String -Path $trainLog -Pattern "MM_LONG DONE" -Quiet)) {
        Write-WLog "training complete -- starting eval-set build"
        break
    }
    Start-Sleep -Seconds 120
}

& $python edax_eval_set.py --build 800 400 *>> (Join-Path $colabs "eval_set_build.log")
Write-WLog "build exited $LASTEXITCODE"
Write-WLog "EVALSET BUILD DONE"
