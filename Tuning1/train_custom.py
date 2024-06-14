import os
import subprocess
import torch
import json
from roboflow import Roboflow
from ultralytics import YOLO
from PIL import Image

home = os.path.dirname(os.path.abspath(__file__))
print("Set home to:", home)

def download_dataset():
    print("Downloading dataset...")
    dataset_dir = os.path.join(home, "datasets")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key="<YOUR_API_KEY>") #TODO: Replace with your Roboflow API key
        project = rf.workspace("first-sbpfm").project("assignment2-ukjhm")
        version = project.version(17)  # Update to the correct version number
        dataset = version.download("yolov8")
        
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")
    
    return os.path.join(dataset_dir, "Assignment2-17") # Update with the correct project version number

def get_latest_train_folder():
    print("Getting the latest training folder...")
    train_dir = os.path.join(home, "runs", "detect")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
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
        'batch': params['batch'],
        'patience': params['patience']
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

def validate_model(dataset_location):
    print("Starting validation...")
    os.chdir(home)
    
    latest_train_folder = get_latest_train_folder()
    if latest_train_folder is None:
        raise ValueError("No training folder found.")
    latest_train_folder="/home/labuser/optuna2/Tuning2/runs/detect/train11"
    model_path = os.path.join(latest_train_folder, "weights", "best.pt")
    model = YOLO(model_path)
    
    print("Running validation")
    metrics = model.val(data=f"{dataset_location}/data.yaml")
    
    print("Validation complete")
    print("mAP50", metrics.box.map50)
    print("mAP50-95", metrics.box.map)
    
    score = metrics.box.map * 0.9 + metrics.box.map50 * 0.1
    print(f"Validation scored: {score}")
    return score

def generate_results():
    latest_train_folder = get_latest_train_folder()
    if latest_train_folder is None:
        raise ValueError("No training folder found.")

    results_files = [
        "confusion_matrix.png",
        "results.png",
        "val_batch0_pred.jpg"
    ]

    for file in results_files:
        file_path = os.path.join(latest_train_folder, file)
        if os.path.exists(file_path):
            img = Image.open(file_path)
            img.show()

def inference_on_test_data(dataset_location):
    latest_train_folder = get_latest_train_folder()
    if latest_train_folder is None:
        raise ValueError("No training folder found.")

    model_path = os.path.join(latest_train_folder, "weights", "best.pt")
    model = YOLO(model_path)

    test_images_dir = os.path.join(dataset_location, "test", "images")
    predictions = model.predict(source=test_images_dir, conf=0.25, iou=0.3, save=True)
    

def upload_model_to_roboflow():
    latest_train_folder = get_latest_train_folder()
    if latest_train_folder is None:
        raise ValueError("No training folder found.")

    model_file_path = os.path.join(latest_train_folder)
    
    rf = Roboflow(api_key="EEIbIynr6aG4dsAFItbx")
    project = rf.workspace("first-sbpfm").project("assignment2-ukjhm")
    project.version(17).deploy(model_type="yolov8", model_path=model_file_path)  # Update with correct version


def objective():
    print("Starting new trial...")
    params = {
        'lr0': 5.340830203597959e-05,
        'lrf': 6.658719858230725e-05,
        'momentum': 0.7151563798083189,
        'weight_decay': 0.00740449704977633,
        'warmup_epochs': 3.0148126015578924,
        'warmup_momentum': 0.05083623912266233,
        'box': 0.18697729734167934,
        'cls': 0.47955025711954513,
        'dfl': 1.7555951297274137,
        'hsv_h': 0.5376538649114236,
        'hsv_s': 0.05797103582305906,
        'hsv_v': 0.9974153470339517,
        'degrees': -11.415975476467715,
        'translate': 0.6420969755042787,
        'scale': 0.34942184409581534,
        'shear': 2.1372345146670457,
        'perspective': 0.0002800787194384415,
        'flipud': 0.06407703549460325,
        'fliplr': 0.648331583552765,
        'mosaic': 0.3638940828543471,
        'mixup': 0.14315164347443593,
        'copy_paste': 0.004197438845051173,
        'optimizer': "AdamW",
        'batch': 8,
        'epochs': 300,
        'patience': 100
    }

    check_gpu_memory()
    clear_cuda_memory()
    check_gpu_memory()
    
    dataset_location = download_dataset()
    model = train_model(dataset_location, params)
    score = validate_model(dataset_location)
    
    generate_results()
    inference_on_test_data(dataset_location)
    upload_model_to_roboflow()
    
    if score:
        print(f"Trial completed with score: {score}")
        return -score

score = objective()
print(score)

