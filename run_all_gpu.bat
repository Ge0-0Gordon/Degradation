@echo off
setlocal

set PYTHON=C:/Users/Lenovo/.conda/envs/deeplearning/python.exe
set PROJECT_ROOT=%~dp0
cd /d %PROJECT_ROOT%

:: 1. 数据退化可视化
%PYTHON% scripts/generate_degradation_demo.py

:: 2. U-Net clean baseline 训练
%PYTHON% -m src.train --config configs/unet_baseline.yaml --run_name unet_baseline_gpu

:: 3. U-Net clean baseline 验证
%PYTHON% -m src.evaluate --config configs/unet_baseline.yaml --ckpt runs/unet_baseline_gpu/best.pt --split val

:: 4. Degradation-aware training
%PYTHON% -m src.train --config configs/deeplabv3_degaware.yaml --run_name deeplabv3_degaware_gpu

:: 5. JPEG severity=3 评估
%PYTHON% -m src.evaluate --config configs/deeplabv3_degaware.yaml --ckpt runs/deeplabv3_degaware_gpu/best.pt --eval_degradation jpeg --eval_severity 3

:: 6. Robustness 曲线
%PYTHON% scripts/run_ablation_grid.py --mode robustness --config configs/deeplabv3_degaware.yaml --ckpt runs/deeplabv3_degaware_gpu/best.pt --out_dir analysis_outputs

:: 7. 定性可视化
python -m scripts.visualize_predictions.py --config configs/deeplabv3_degaware.yaml --ckpt runs/deeplabv3_degaware_gpu/best.pt --indices 0 5 10 --eval_degradation motion_blur --eval_severity 4 --out_dir qual_vis

endlocal
