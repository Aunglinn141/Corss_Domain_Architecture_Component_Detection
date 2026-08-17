@echo off
setlocal
cd /d "%~dp0.."

rem Edit these paths for your machine.
set DATA_YAML=configs\data.example.yaml
set WEIGHTS=weights\yolo26m-obb.pt

python scripts\train.py ^
  --model configs\yolo26m-p2-c2psa.yaml ^
  --data "%DATA_YAML%" ^
  --weights "%WEIGHTS%" ^
  --remap-obb26-head ^
  --epochs 300 ^
  --imgsz 960 ^
  --batch 3 ^
  --device 0 ^
  --name yolo26m-p2-c2psa

exit /b %ERRORLEVEL%

