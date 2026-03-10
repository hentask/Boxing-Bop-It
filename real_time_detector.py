"""
Real-Time Boxing Punch Detector for StrikeSense
Main application that classifies punches in real-time from webcam.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import cv2
import numpy as np
import time
from collections import deque
from tensorflow import keras
from config import *
from utils import (
    extract_landmarks, 
    normalize_landmarks, 
    draw_landmarks,
    DebounceFilter,
    calculate_velocity,
    put_text_with_background
)

class RealtimeDetector:
    def __init__(self):
        print("=== StrikeSense Real-Time Detector ===")
        
        # Load model
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except:
            print(f"Error: Could not load model from {MODEL_PATH}")
            print("Please run train_model.py first to train a model.")
            exit(1)
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)
        self.debounce_filter = DebounceFilter(DEBOUNCE_FRAMES)
        
        # Stats
        self.punch_counts = {class_name: 0 for class_name in PUNCH_CLASSES}
        self.fps = 0
        self.current_prediction = "No Action"
        self.current_confidence = 0.0
        
        # Motion trigger
        self.motion_threshold = 0.02  # Minimum velocity to trigger detection
        self.is_moving = False
        
        print("\nControls:")
        print("  R - Reset punch counter")
        print("  Q - Quit")
        print("\nReady to detect punches!")
    
    def predict(self):
        """Run inference on current frame buffer"""
        if len(self.frame_buffer) < WINDOW_SIZE:
            return "No Action", 0.0
        
        # Convert buffer to numpy array
        window = np.array(self.frame_buffer)
        
        # Check if there's significant motion
        velocity = calculate_velocity(window)
        self.is_moving = velocity > self.motion_threshold
        
        # If not moving, return No Action
        if not self.is_moving:
            return "No Action", 1.0
        
        # Reshape for model input
        X = window.reshape(1, WINDOW_SIZE, -1)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return "No Action", confidence
        
        predicted_class = PUNCH_CLASSES[predicted_class_idx]
        
        return predicted_class, confidence
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        prev_time = time.time()
        last_punch = "No Action"
        frame_skip_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Extract landmarks
            landmarks, results = extract_landmarks(frame)
            
            if landmarks is not None:
                # Normalize and add to buffer
                normalized = normalize_landmarks(landmarks)
                self.frame_buffer.append(normalized)
                
                # Draw skeleton
                frame = draw_landmarks(frame, results)
                
                # Only run model inference every 2 frames (doubles FPS!)
                frame_skip_counter += 1
                if frame_skip_counter >= 2 and len(self.frame_buffer) == WINDOW_SIZE:
                    frame_skip_counter = 0
                    
                    predicted_class, confidence = self.predict()
                    
                    # Apply debouncing
                    stable_class = self.debounce_filter.update(predicted_class)
                    
                    self.current_prediction = stable_class
                    self.current_confidence = confidence
                    
                    # Count punches (when transitioning from No Action to a punch)
                    if stable_class != "No Action" and last_punch == "No Action":
                        self.punch_counts[stable_class] += 1
                    
                    last_punch = stable_class
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow("StrikeSense - Real-Time Detector", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset counters
                self.punch_counts = {class_name: 0 for class_name in PUNCH_CLASSES}
                print("Punch counter reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        print("\n=== Session Summary ===")
        total_punches = sum(self.punch_counts.values()) - self.punch_counts["No Action"]
        print(f"Total punches thrown: {total_punches}")
        for class_name, count in self.punch_counts.items():
            if class_name != "No Action" and count > 0:
                print(f"  {class_name}: {count}")
    
    def draw_ui(self, frame):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Ensure frame is valid
        if frame is None or h == 0 or w == 0:
            return frame
        
        # Draw solid black background bars
        cv2.rectangle(frame, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        
        # Current prediction - large and bold
        pred_color = COLORS.get(self.current_prediction, (255, 255, 255))
        cv2.putText(frame, f"PUNCH: {self.current_prediction}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, pred_color, 3, cv2.LINE_AA)
        
        # Confidence
        conf_text = f"Confidence: {self.current_confidence:.1%}"
        cv2.putText(frame, conf_text, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(frame, fps_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Buffer status
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{WINDOW_SIZE}"
        cv2.putText(frame, buffer_text, (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Motion indicator
        motion_text = "MOTION" if self.is_moving else "IDLE"
        motion_color = (0, 255, 0) if self.is_moving else (100, 100, 100)
        cv2.putText(frame, motion_text, (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, motion_color, 2, cv2.LINE_AA)
        
        # Punch counter on right side
        counter_x = max(20, w - 300)
        y_offset = 50
        cv2.putText(frame, "PUNCH COUNT", (counter_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        y_offset += 40
        for class_name in PUNCH_CLASSES:
            if class_name != "No Action":
                count = self.punch_counts[class_name]
                color = COLORS.get(class_name, (255, 255, 255))
                count_text = f"{class_name}: {count}"
                cv2.putText(frame, count_text, (counter_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                y_offset += 30
        
        # Instructions at bottom
        cv2.putText(frame, "Press 'R' to reset | 'Q' to quit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    detector = RealtimeDetector()
    detector.run()