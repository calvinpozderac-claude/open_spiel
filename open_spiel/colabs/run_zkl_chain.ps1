Set-Location "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs"
foreach ($s in @('ply','plies','const')) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Calvin Pozderac\Documents\open_spiel\open_spiel\colabs\run_zkl_forever.ps1" $s
}
