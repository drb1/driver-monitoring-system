## Classes
- Normal
- Aggressive
- Distracted
- Drowsy

## Environment
- Python 3.9 recommended
- TensorFlow **2.20.x** (uses `tensorflow.keras` imports)
- Works on macOS/Apple Silicon and Linux (CPU/GPU), as long as TF 2.20 is installed

## Quick Start
python -m venv .venv
source .venv/bin/activate 
python -m pip install --upgrade pip
pip install -r requirements.txt


### 1) Organize your data
data/raw/Normal/*.mp4
data/raw/Aggressive/*.mp4
data/raw/Distracted/*.mp4
data/raw/Drowsy/*.mp4

### 2) Convert videos to 16-frame clips (224×224 recommended)
python3 scripts/process_videos.py --input_root data/raw --output_root data/processed --seq_len 16 --img_size 224

### 3) Create **stratified** train/val/test splits
python3 scripts/make_splits.py --processed_root data/processed --train 0.8 --val 0.1 --test 0.1

### 4) Train
# LSTM (with class weights by default)
python3 src/training.py --model lstm --epochs 30 --batch 8 --seq_len 16 --img_size 224

# Transformer
python3 src/training.py --model transformer --epochs 30 --batch 8 --seq_len 16 --img_size 224

Options:
- `--fine_tune --warmup_epochs 3` to unfreeze MobileNetV2 after warmup
- `--loss focal` to use focal loss (default: cross-entropy)
- `--no_class_weight` to disable class weights

### 5) Evaluate (adds ECE + reliability plot, optional TTA)
# baseline eval
python3 src/evaluation.py --model_path outputs/models/best_lstm.keras --seq_len 16 --img_size 224

# with test-time augmentation (e.g., 5 temporal crops)
python3 src/evaluation.py --model_path outputs/models/best_lstm.keras --seq_len 16 --img_size 224 --tta 5

Outputs:
- Confusion matrix (raw + normalized)
- Classification report (CSV)
- ROC curves (per class) + macro/micro AUC
- Precision–Recall curves + mAP
- **ECE@10** and **reliability diagram** (`outputs/plots/reliability.png`)
- `outputs/metrics/*_preds.csv` with per-sample probabilities
- A **.keras copy** of best and final models is saved for portability.

### 6) Compare two models statistically
python3 src/compare_models.py --preds_a outputs/metrics/preds_lstm.csv --preds_b outputs/metrics/preds_transformer.csv

### 7) Real-time inference (webcam or video)
python3 src/realtime.py --model_path outputs/models/best_lstm.keras --seq_len 16 --img_size 224 --source 0
# or a file: --source path/to/video.mp4

### 8) inspect trainables
python3 src/inspect_trainables.py --model tcn
python3 src/inspect_trainables.py --model lstm
python3 src/inspect_trainables.py --model transformer

### next version
# evaluate
python src/evaluation_fusion.py \
  --model_path outputs/models/best_fusion_tcn.keras \
  --test_csv data/splits/test.csv \
  --seq_len 16 --img_size 224 --batch 8 --tta 1

# train
python src/build_audio_cache.py
python src/training.py --model tcn --multimodal --fusion_base tcn --epochs 30 --batch 8
# inspect prediction
python src/inspect_fusion_preds.py
# identify silenced audio clips
python src/find_silent_audio_clips.py
# Check if audio branch is overpowering video branch
python src/ablate_audio_on_mistakes.py
