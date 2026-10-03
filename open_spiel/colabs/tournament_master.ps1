Set-Location "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
"tournament_master waiting for factorial cells $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\tournament_master.log"
while (-not ((Test-Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\fact_admle_pathavg.log") -and (Select-String -Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\fact_admle_pathavg.log" -Pattern "ALLMSE_ADMLE_PATHAVG DONE" -Quiet))) { Start-Sleep -Seconds 120 }
"all cells done, running tournament $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\tournament_master.log"
& $python tournament.py 100 400 *>> "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\tournament_stdout.log"
"tournament done $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\tournament_master.log"
