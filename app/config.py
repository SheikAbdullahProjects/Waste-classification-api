import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "garbage_classification"

MODEL_PATH_ENV = os.getenv("MODEL_PATH")
if MODEL_PATH_ENV:
	MODEL_PATH = Path(MODEL_PATH_ENV)
else:
	candidate_project = PROJECT_ROOT / "backend" / "model_artifacts" / "best_model.pt"
	candidate_backend = BACKEND_ROOT / "model_artifacts" / "best_model.pt"
	MODEL_PATH = candidate_project if candidate_project.exists() else candidate_backend

MODEL_DIR = MODEL_PATH.parent

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows-friendly default
NUM_EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
USE_WEIGHTED_SAMPLER = True

# Optional speed controls (set to None to use full dataset)
MAX_SAMPLES = None
MAX_TRAIN_STEPS = None
MAX_VAL_STEPS = None
