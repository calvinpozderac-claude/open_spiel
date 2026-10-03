Set-Location "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
$python = "C:\Users\Calvin Pozderac\Documents\open_spiel\.venv312\Scripts\python.exe"
"master_followup started $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\master_followup.log"
while (-not ((Test-Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\zkl_const.log") -and (Select-String -Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\zkl_const.log" -Pattern "ZKL_CONST DONE" -Quiet))) { Start-Sleep -Seconds 60 }
"const done, running duels $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\master_followup.log"
foreach ($a in @('ply','plies','const')) {
    & $python duel_nets.py "othello_benchmark/bench_ZKL_$a:10000" "othello_benchmark/bench_ZGROUND:10000" 400 121 *>> "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\duel_followup_stdout.log"
}
"duels done, launching variant arms $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\master_followup.log"
foreach ($arm in @('zkl2','cemse_double','cemse_shift','allmse')) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\run_zvariant_forever.ps1" $arm
}
"master_followup done $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\master_followup.log"
