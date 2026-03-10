"""
Utility functions for StrikeSense Boxing AI
Compatible with MediaPipe 0.10.32+ (Vision Tasks API)
"""

import numpy as np
import cv2
from config import LANDMARK_INDICES
import urllib.request
import os

class MediaPipePoseDetector:
    """
    Pose detector using MediaPipe Vision Tasks API (0.10.32+)
    """
    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download the pose landmarker model if it doesn't exist
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                print("Downloading MediaPipe Pose model (this only happens once)...")
                url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
                urllib.request.urlretrieve(url, model_path)
                print("✓ Model downloaded successfully")
            
            # Create pose landmarker
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.frame_timestamp_ms = 0
            self.has_mediapipe = True
            print("✓ MediaPipe Pose initialized successfully (Vision Tasks API)")
            
        except Exception as e:
            print(f"ERROR initializing MediaPipe: {e}")
            self.has_mediapipe = False
            self.detector = None
    
    def process(self, frame):
        """Process a BGR frame and return landmarks"""
        if not self.has_mediapipe:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Increment timestamp (in milliseconds)
        self.frame_timestamp_ms += 33  # ~30 FPS
        
        # Detect pose
        detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        return detection_result
    
    def draw_landmarks(self, frame, detection_result):
        """Draw pose landmarks on the frame"""
        if not detection_result or not detection_result.pose_landmarks:
            return frame
        
        # Get the first pose's landmarks
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Draw landmarks manually
        h, w, _ = frame.shape
        
        # Define connections (simplified skeleton)
        connections = [
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in pose_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        return frame

# Create global pose detector
pose_detector = MediaPipePoseDetector()

def extract_landmarks(frame):
    """
    Extract pose landmarks from a frame using MediaPipe.
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        landmarks: numpy array of shape (num_landmarks * 3,) or None if not detected
        results: MediaPipe pose results object for visualization
    """
    # Process the frame
    results = pose_detector.process(frame)
    
    if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
        # Extract only the landmarks we care about
        pose_landmarks = results.pose_landmarks[0]
        landmarks = []
        
        for idx in LANDMARK_INDICES:
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks), results
    
    return None, results

def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be position-invariant.
    Centers around the midpoint of shoulders.
    
    Args:
        landmarks: numpy array of shape (num_landmarks * 3,)
        
    Returns:
        normalized: numpy array of same shape
    """
    landmarks = landmarks.reshape(-1, 3)
    
    # Use shoulder midpoint as reference (indices 0 and 1 in our subset)
    shoulder_midpoint = (landmarks[0] + landmarks[1]) / 2
    
    # Center all landmarks around shoulder midpoint
    normalized = landmarks - shoulder_midpoint
    
    return normalized.flatten()

def draw_landmarks(frame, results):
    """
    Draw pose landmarks on the frame.
    
    Args:
        frame: BGR image
        results: MediaPipe pose results
        
    Returns:
        frame: annotated frame
    """
    return pose_detector.draw_landmarks(frame, results)

def create_sliding_window(data, window_size, stride):
    """
    Create sliding windows from sequential data.
    
    Args:
        data: numpy array of shape (num_frames, num_features)
        window_size: number of frames per window
        stride: step size between windows
        
    Returns:
        windows: numpy array of shape (num_windows, window_size, num_features)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)

class DebounceFilter:
    """
    Prevents rapid flickering between classifications.
    Only changes prediction if new class persists for debounce_frames.
    """
    def __init__(self, debounce_frames=10):
        self.debounce_frames = debounce_frames
        self.current_class = "No Action"
        self.candidate_class = "No Action"
        self.candidate_count = 0
    
    def update(self, predicted_class):
        """
        Update filter with new prediction.
        
        Args:
            predicted_class: string of predicted punch type
            
        Returns:
            stable_class: the debounced class prediction
        """
        if predicted_class == self.current_class:
            # Reset candidate if we return to current
            self.candidate_class = predicted_class
            self.candidate_count = 0
            return self.current_class
        
        if predicted_class == self.candidate_class:
            # Increment candidate count
            self.candidate_count += 1
            
            # If candidate has persisted long enough, switch
            if self.candidate_count >= self.debounce_frames:
                self.current_class = predicted_class
                self.candidate_count = 0
        else:
            # New candidate appeared
            self.candidate_class = predicted_class
            self.candidate_count = 1
        
        return self.current_class

def calculate_velocity(landmarks_sequence):
    """
    Calculate velocity of wrist movement (useful for motion triggers).
    
    Args:
        landmarks_sequence: numpy array of shape (num_frames, num_features)
        
    Returns:
        velocity: average velocity of both wrists
    """
    # Wrist indices in our subset are 4 and 5
    # Each landmark has 3 coords (x, y, z)
    left_wrist = landmarks_sequence[:, 12:15]  # Index 4 * 3
    right_wrist = landmarks_sequence[:, 15:18]  # Index 5 * 3
    
    # Calculate frame-to-frame differences
    left_velocity = np.linalg.norm(np.diff(left_wrist, axis=0), axis=1)
    right_velocity = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1)
    
    # Return average velocity
    return np.mean(left_velocity + right_velocity)

def put_text_with_background(frame, text, position, font_scale=1.0, thickness=2, 
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draw text with a background rectangle for better visibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - 5, y - text_height - 5),
                  (x + text_width + 5, y + baseline + 5),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    
    return frame