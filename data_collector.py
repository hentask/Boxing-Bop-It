"""
Data Collection Tool for StrikeSense Boxing AI
Record yourself performing each punch type to build training data.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from config import *
from utils import extract_landmarks, normalize_landmarks, draw_landmarks

class DataCollector:
    def __init__(self):
        self.current_class = 0
        self.recording = False
        self.session_data = []
        self.frame_count = 0
        
        # Create directories
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        print("=== StrikeSense Data Collector ===")
        print(f"Classes: {PUNCH_CLASSES}")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  N - Next class")
        print("  P - Previous class")
        print("  S - Save session")
        print("  Q - Quit")
        
    def run(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        current_sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, results = extract_landmarks(frame)
            
            # Draw skeleton
            frame = draw_landmarks(frame, results)
            
            # Record data if recording
            if self.recording and landmarks is not None:
                normalized = normalize_landmarks(landmarks)
                current_sequence.append(normalized)
            
            # Update frame count to show current sequence length
            if self.recording:
                self.frame_count = len(current_sequence)
            else:
                self.frame_count = 0
            
            # Display UI
            status_color = (0, 0, 255) if self.recording else (0, 255, 0)
            status_text = "RECORDING" if self.recording else "READY"
            frame_text = f"Current: {self.frame_count}" if self.recording else "Ready to record"
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Class: {PUNCH_CLASSES[self.current_class]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status_text}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, frame_text, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Data Collector", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Toggle recording
                if not self.recording:
                    self.recording = True
                    current_sequence = []
                    print(f"Started recording: {PUNCH_CLASSES[self.current_class]}")
                else:
                    self.recording = False
                    if len(current_sequence) > 0:
                        self.session_data.append({
                            'class': self.current_class,
                            'class_name': PUNCH_CLASSES[self.current_class],
                            'sequence': current_sequence,
                            'length': len(current_sequence)
                        })
                        print(f"Saved sequence: {len(current_sequence)} frames")
                    current_sequence = []
            
            elif key == ord('n'):
                # Next class
                self.current_class = (self.current_class + 1) % len(PUNCH_CLASSES)
                print(f"Switched to: {PUNCH_CLASSES[self.current_class]}")
                self.frame_count = 0
            
            elif key == ord('p'):
                # Previous class
                self.current_class = (self.current_class - 1) % len(PUNCH_CLASSES)
                print(f"Switched to: {PUNCH_CLASSES[self.current_class]}")
                self.frame_count = 0
            
            elif key == ord('s'):
                # Save session
                self.save_session()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_session(self):
        """Save collected data to disk"""
        if len(self.session_data) == 0:
            print("No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RAW_DATA_DIR}/session_{timestamp}.npz"
        
        # Prepare data
        sequences = []
        labels = []
        
        for item in self.session_data:
            sequences.append(item['sequence'])
            labels.append(item['class'])
        
        # Save as compressed numpy file
        np.savez_compressed(
            filename,
            sequences=np.array(sequences, dtype=object),
            labels=np.array(labels),
            class_names=PUNCH_CLASSES
        )
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_sequences': len(sequences),
            'class_distribution': {}
        }
        
        for i, class_name in enumerate(PUNCH_CLASSES):
            count = sum(1 for label in labels if label == i)
            metadata['class_distribution'][class_name] = count
        
        with open(f"{RAW_DATA_DIR}/session_{timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n=== Session Saved ===")
        print(f"File: {filename}")
        print(f"Total sequences: {len(sequences)}")
        print("Class distribution:")
        for class_name, count in metadata['class_distribution'].items():
            print(f"  {class_name}: {count}")
        
        # Clear session data
        self.session_data = []
        self.frame_count = 0

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()