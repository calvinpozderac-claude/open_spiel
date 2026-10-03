Set-Location "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
"master waiting for factorial cell 3 $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\az_tourn_master.log"
while (-not ((Test-Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\fact_admle_pathavg.log") -and (Select-String -Path "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\fact_admle_pathavg.log" -Pattern "ALLMSE_ADMLE_PATHAVG DONE" -Quiet))) { Start-Sleep -Seconds 120 }
"cells done, training AZ $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\az_tourn_master.log"
& powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\run_az_forever.ps1"
"AZ done, running tournament $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\az_tourn_master.log"
& $python tournament.py 100 400 *>> "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\tournament_stdout.log"
"tournament done $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\az_tourn_master.log"
