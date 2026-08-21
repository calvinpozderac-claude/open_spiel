#!/bin/bash
# Stop MM_long cleanly: wait until latest.pt advances to the next 1000-episode
# save, then tree-kill the trainer. Killing mid-interval would discard every
# episode since the last save (latest.pt is written on quick_eval_every=1000).
cd "C:/Users/Calvin Pozderac/Documents/open_spiel/open_spiel/colabs"
PY="C:/Users/Calvin Pozderac/Documents/open_spiel/.venv312/Scripts/python.exe"
TARGET=17000
while true; do
  EP=$("$PY" -c "
import torch
try:
    print(torch.load('othello_benchmark/bench_MM_long/latest.pt',map_location='cpu',weights_only=False)['ep'])
except Exception:
    print(0)" 2>/dev/null)
  echo "[pause] latest.pt ep=$EP target=$TARGET $(date)" >> pause_at_save.log
  if [ "$EP" -ge "$TARGET" ] 2>/dev/null; then
    echo "[pause] reached ep $EP -- stopping trainer" >> pause_at_save.log
    for P in $(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_mm_long.py*' }).ProcessId" 2>/dev/null); do
      powershell -NoProfile -Command "taskkill /PID $P /T /F" >> pause_at_save.log 2>&1
    done
    echo "[pause] PAUSED at ep $EP" >> pause_at_save.log
    break
  fi
  sleep 60
done
