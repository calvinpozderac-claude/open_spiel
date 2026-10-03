Set-Location "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
"factorial chain started $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\factorial_chain.log"
foreach ($c in @('admle_avgconc','pathsum_pathavg','admle_pathavg')) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\run_factorial_forever.ps1" $c
}
"factorial chain done $(Get-Date)" | Out-File -Append "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\factorial_chain.log"
