import os
import subprocess
import optuna
import torch
import json
from roboflow import Roboflow
from ultralytics import YOLO

home = os.path.dirname(os.path.abspath(__file__))
print("Set home to:", home)

def download_dataset():
    print("Downloading dataset...")
    dataset_dir = os.path.join(home, "datasets")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        
        #TODO: replace with code snipplet
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key="<YOUR_API_KEY>") #TODO: replace with your Roboflow API key
        project = rf.workspace("first-sbpfm").project("assignment2-ukjhm")
        version = project.version(17)
        dataset = version.download("yolov8")

        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")
        
    return os.path.join(dataset_dir, "Assignment2-17") #TODO: replace with project version number

def get_latest_train_folder():
    print("Getting the latest training folder...")
    train_dir = os.path.join(home, "runs", "detect")
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    train_subdirs = [d for d in subdirs if d.startswith("train")]
    if train_subdirs:
        latest_subdir = max(train_subdirs, key=lambda d: int(d.replace("train", "") or "0"))
        print(f"Latest training folder is '{latest_subdir}'.")
        return os.path.join(train_dir, latest_subdir)
    print("No training folders found.")
    return None


def check_gpu_memory():
    print("Checking GPU memory usage...")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)

def clear_cuda_memory():
    print("Clearing CUDA memory...")
    torch.cuda.empty_cache()
    print("CUDA memory cleared.")


def train_model(dataset_location, params):
    print("Starting training...")
    os.chdir(home)

    # Initialize the YOLO model
    model = YOLO('yolov8l.pt')

    # Define training parameters
    training_params = {
        'data': f"{dataset_location}/data.yaml",
        'epochs': params['epochs'],
        'imgsz': 640,
        'plots': True,
        'lr0': params['lr0'],
        'lrf': params['lrf'],
        'momentum': params['momentum'],
        'weight_decay': params['weight_decay'],
        'warmup_epochs': params['warmup_epochs'],
        'warmup_momentum': params['warmup_momentum'],
        'box': params['box'],
        'cls': params['cls'],
        'dfl': params['dfl'],
        'hsv_h': params['hsv_h'],
        'hsv_s': params['hsv_s'],
        'hsv_v': params['hsv_v'],
        'degrees': params['degrees'],
        'translate': params['translate'],
        'scale': params['scale'],
        'shear': params['shear'],
        'perspective': params['perspective'],
        'flipud': params['flipud'],
        'fliplr': params['fliplr'],
        'mosaic': params['mosaic'],
        'mixup': params['mixup'],
        'copy_paste': params['copy_paste'],
        'optimizer': params['optimizer'],
        'batch': params['batch']
    }

    print("Training with parameters:")
    print(json.dumps(training_params, indent=4))

    try:
        model.train(**training_params)
        print("Training completed successfully.")
        return model
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

def validate_model(model, dataset_location):
    print("Starting validation...")
    os.chdir(home)
    
    latest_train_folder = get_latest_train_folder()
    if latest_train_folder is None:
        raise ValueError("No training folder found.")
    
    
    model_path = f"{latest_train_folder}/weights/best.pt"
    model = YOLO(model_path)
    
    print(f"Running validation")
    metrics = model.val()
    
    print("Validation complete")
    print("mAP50", metrics.box.map50)
    print("mAP50-95", metrics.box.map)
    
    score = metrics.box.map * 0.9 + metrics.box.map50 * 0.1
    print(f"Validation scored: {score}")
    return score


def objective(trial):
    print("Starting new trial...")
    
    params = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-1, log=True),
        'lrf': trial.suggest_float('lrf', 1e-5, 1e-1, log=True),
        'momentum': trial.suggest_float('momentum', 0.7, 0.99),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'warmup_epochs': trial.suggest_float('warmup_epochs', 0, 5),
        'warmup_momentum': trial.suggest_float('warmup_momentum', 0.0, 1.0),
        'box': trial.suggest_float('box', 0.02, 0.2),
        'cls': trial.suggest_float('cls', 0.2, 4.0),
        'dfl': trial.suggest_float('dfl', 0.5, 3.0),
        'hsv_h': trial.suggest_float('hsv_h', 0.0, 1.0),
        'hsv_s': trial.suggest_float('hsv_s', 0.0, 1.0),
        'hsv_v': trial.suggest_float('hsv_v', 0.0, 1.0),
        'degrees': trial.suggest_float('degrees', -180.0, 180.0),
        'translate': trial.suggest_float('translate', 0.0, 1.0),
        'scale': trial.suggest_float('scale', 0.0, 1.0),
        'shear': trial.suggest_float('shear', -180.0, 180.0),
        'perspective': trial.suggest_float('perspective', 0.0, 0.001),
        'flipud': trial.suggest_float('flipud', 0.0, 1.0),
        'fliplr': trial.suggest_float('fliplr', 0.0, 1.0),
        'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
        'mixup': trial.suggest_float('mixup', 0.0, 1.0),
        'copy_paste': trial.suggest_float('copy_paste', 0.0, 1.0),
        'optimizer': "AdamW",
        'batch': 8,
        'epochs': 30
    }

    check_gpu_memory()
    clear_cuda_memory()
    check_gpu_memory()
    
    dataset_location = download_dataset()
    model = train_model(dataset_location, params)
    score = validate_model(model, dataset_location)
    
    print(f"Trial completed with score: {score}")
    return -score





print("Loading study...")
study = optuna.load_study(
    study_name='yolov8_study_2',
    storage='mysql+pymysql://optuna_user:Pa$$word1234@192.168.25.182/optuna_db'
)

print("Starting optimization...")
study.optimize(objective, n_trials=100, n_jobs=1)
print("Optimization completed.")

