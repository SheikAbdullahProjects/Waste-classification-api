from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "garbage_classification"
MODEL_DIR = ROOT_DIR / "backend" / "model_artifacts"
MODEL_PATH = MODEL_DIR / "best_model.pt"

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
