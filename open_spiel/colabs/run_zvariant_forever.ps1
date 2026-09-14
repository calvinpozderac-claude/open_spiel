param([string]$arm)
$ErrorActionPreference = 'Continue'
$colabs = "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
$tag = $arm.ToUpper()
$log = Join-Path $colabs "zvar_$arm.log"
$wrapperLog = Join-Path $colabs "zvar_${arm}_wrap.log"
Set-Location $colabs
function Write-WLog($msg) {
    Add-Content -Path $wrapperLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WRAPPER: $msg"
}
Write-WLog "wrapper started (pid $PID) arm=$arm"
while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "$tag DONE" -Quiet)) {
        Write-WLog "sentinel found -- run complete, exiting wrapper"; break
    }
    Write-WLog "launching run_zvariant $arm process"
    & $python run_zvariant.py $arm *>> (Join-Path $colabs "zvar_${arm}_stdout.log")
    Write-WLog "process exited with code $LASTEXITCODE -- re-checking after a pause"
    Start-Sleep -Seconds 5
}
Write-WLog "wrapper done"
