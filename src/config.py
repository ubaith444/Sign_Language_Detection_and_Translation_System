import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "trained_models"
RECORDINGS_DIR = DATA_DIR / "raw_recordings"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Model Artifact Paths
STATIC_MODEL_PATH = MODELS_DIR / "isl_static_model.joblib"
DYNAMIC_MODEL_PATH = MODELS_DIR / "isl_dynamic_gru.pt"
DYNAMIC_METADATA_PATH = MODELS_DIR / "dynamic_metadata.json"
TRANSLATION_MODEL_PATH = MODELS_DIR / "isl_translation.json"
BENCHMARK_REPORT_PATH = DATA_DIR / "evaluation_report.json"

# Landmark Extraction Configuration
# Left Hand: 21 * 3 = 63
# Right Hand: 21 * 3 = 63
# Upper Body Pose (Shoulders, Elbows, Wrists, Hips): 12 * 3 = 36
# Face Non-Manual Cues (Eyebrows, Eyes, Lips): 20 * 3 = 60
# Total Holistic Feature Vector Dimension = 222
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 12
NUM_FACE_LANDMARKS = 20

FEATURE_DIM_HANDS_ONLY = NUM_HAND_LANDMARKS * 3 * 2  # 126
FEATURE_DIM_HOLISTIC = (NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS) * 3  # 222

# Temporal Sequence Configuration for Dynamic Signs
SEQUENCE_LENGTH = 30  # 30 frames (~1.0s at 30fps)
SEQUENCE_STRIDE = 2   # Downsampling/subsampling step if necessary

# Classification & Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.40       # Minimum posterior probability for 42-class classification (chance = 2.4%)
UNCERTAINTY_MARGIN = 0.08         # Margin between top-1 and top-2
UNKNOWN_LABEL = "UNKNOWN / UNCERTAIN"

# Continuous Sign Segmentation & Debounce
DEBOUNCE_STABLE_FRAMES = 8        # Number of consecutive identical predictions before appending to sentence
KINETIC_VELOCITY_THRESHOLD = 0.038 # Hand coordinate delta indicating intentional sign motion vs resting
TRANSITION_REST_FRAMES = 6        # Low-velocity frame count signaling completion of current sign

# Speech Synthesis Settings
TTS_RATE = 155
TTS_VOLUME = 1.0

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Web Application Settings
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
