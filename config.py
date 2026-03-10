"""
Configuration file for StrikeSense Boxing AI
"""

# Punch Classes
PUNCH_CLASSES = [
    "No Action",
    "Jab",
    "Cross",
    "Lead Hook",
    "Rear Hook",
    "Lead Uppercut",
    "Rear Uppercut"
]

# MediaPipe Pose Landmarks to track (upper body focus)
LANDMARK_INDICES = [
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips (for stability reference)
    0        # Nose (for head position)
]

# Temporal Window Settings
WINDOW_SIZE = 15  # Number of frames (1 second at 30 FPS)
STRIDE = 3  # How many frames to skip between windows (for overlap)

# Model Settings
INPUT_SHAPE = (WINDOW_SIZE, len(LANDMARK_INDICES) * 3)  # 30 frames x (9 landmarks x 3 coords)
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Real-time Detection Settings
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to classify a punch
DEBOUNCE_FRAMES = 5  # Prevent flickering between classes
FPS_TARGET = 30

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Data Collection Settings
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/boxing_classifier.h5"

# Visualization Colors (BGR format for OpenCV)
COLORS = {
    "No Action": (200, 200, 200),
    "Jab": (0, 255, 0),
    "Cross": (255, 0, 0),
    "Lead Hook": (0, 255, 255),
    "Rear Hook": (255, 255, 0),
    "Lead Uppercut": (255, 0, 255),
    "Rear Uppercut": (0, 165, 255)
}