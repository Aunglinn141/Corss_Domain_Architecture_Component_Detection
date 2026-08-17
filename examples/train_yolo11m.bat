@echo off
setlocal
cd /d "%~dp0.."

rem Edit these paths for your machine.
set DATA_YAML=configs\data.example.yaml
set WEIGHTS=weights\yolo11m-obb.pt

python scripts\train.py ^
  --model configs\yolo11m-p2-c2psa.yaml ^
  --data "%DATA_YAML%" ^
  --weights "%WEIGHTS%" ^
  --epochs 300 ^
  --imgsz 960 ^
  --batch 4 ^
  --device 0 ^
  --name yolo11m-p2-c2psa

exit /b %ERRORLEVEL%

