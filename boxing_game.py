"""
Boxing Bop It Game Mode for StrikeSense
Play along with voice commands and test your speed!
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import time
import random
from collections import deque
from tensorflow import keras
from config import *
from utils import (
    extract_landmarks, 
    normalize_landmarks, 
    draw_landmarks,
    DebounceFilter,
    calculate_velocity
)

# Audio playback
try:
    import pygame
    pygame.mixer.init()
    pygame.mixer.set_num_channels(1)  # Only one sound at a time
    AUDIO_AVAILABLE = True
    
    # Pre-load all sounds for faster playback
    SOUND_CACHE = {}
    print("✓ Audio system initialized (pygame)")
except ImportError:
    print("Warning: pygame not installed. Run: pip install pygame")
    AUDIO_AVAILABLE = False
    SOUND_CACHE = {}

class BoxingGame:
    def __init__(self):
        print("=== Boxing Bop It Game ===")
        
        # Load model
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded")
        except:
            print(f"ERROR: Could not load model from {MODEL_PATH}")
            exit(1)
        
        # Game punch types (exclude "No Action")
        self.game_punches = [p for p in PUNCH_CLASSES if p != "No Action"]
        
        # Pre-load all sound files
        self.load_sounds()
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)
        self.debounce_filter = DebounceFilter(DEBOUNCE_FRAMES)
        
        # Game state
        self.game_active = False
        self.current_target = None
        self.target_start_time = 0
        self.time_limit = 4.0  # seconds to respond (was 3.0, now 4.0)
        self.score = 0
        self.total_attempts = 0
        self.streak = 0
        self.best_streak = 0
        
        # Detection state
        self.current_prediction = "No Action"
        self.current_confidence = 0.0
        self.fps = 0
        self.motion_threshold = 0.01
        self.is_moving = False
        
        # Game settings
        self.min_time = 1.5  # Minimum time between commands (gets faster)
        self.max_time = 3.0  # Maximum time between commands
        
        print("\nControls:")
        print("  SPACE - Start/Pause game")
        print("  R - Reset score")
        print("  Q - Quit")
        print("\nPress SPACE to start!")
    
    def load_sounds(self):
        """Pre-load all sound files into memory"""
        if not AUDIO_AVAILABLE:
            return
        
        print("Loading sound files...")
        for punch in self.game_punches:
            filename = punch.lower().replace(" ", "_") + ".mp3"
            filepath = os.path.join("sounds", filename)
            
            if os.path.exists(filepath):
                try:
                    SOUND_CACHE[punch] = pygame.mixer.Sound(filepath)
                    print(f"  ✓ Loaded: {punch}")
                except Exception as e:
                    print(f"  ✗ Error loading {punch}: {e}")
            else:
                print(f"  ✗ Missing: {filepath}")
        
        print(f"✓ {len(SOUND_CACHE)} sounds loaded")
    
    def play_sound(self, punch_name):
        """Play audio command for a punch"""
        if not AUDIO_AVAILABLE:
            return
        
        # Stop any currently playing sound
        pygame.mixer.stop()
        
        # Play the sound from cache
        if punch_name in SOUND_CACHE:
            try:
                SOUND_CACHE[punch_name].play()
            except Exception as e:
                print(f"Error playing {punch_name}: {e}")
        else:
            print(f"Warning: Sound not loaded for {punch_name}")
    
    def new_target(self):
        """Generate a new random punch command"""
        self.current_target = random.choice(self.game_punches)
        self.target_start_time = time.time()
        
        # Play the voice command
        self.play_sound(self.current_target)
        
        print(f"Command: {self.current_target}")
    
    def check_response(self, detected_punch):
        """Check if the player threw the correct punch"""
        if detected_punch == "No Action":
            return  # Ignore no action
        
        if detected_punch == self.current_target:
            # Correct!
            self.score += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            print(f"✓ CORRECT! Streak: {self.streak}")
            
            # Speed up the game slightly (but starts at 4.0 now)
            self.time_limit = max(2.0, self.time_limit - 0.05)  # Minimum 2.0 seconds
            
            # Next target
            self.new_target()
        else:
            # Wrong punch!
            print(f"✗ Wrong! Expected {self.current_target}, got {detected_punch}")
            self.streak = 0
            # Don't count as attempt yet - give another chance
    
    def check_timeout(self):
        """Check if time ran out"""
        elapsed = time.time() - self.target_start_time
        
        if elapsed > self.time_limit:
            # Time's up!
            print(f"✗ Too slow! Expected {self.current_target}")
            self.total_attempts += 1
            self.streak = 0
            self.new_target()
            return True
        
        return False
    
    def predict(self):
        """Run inference on current frame buffer"""
        if len(self.frame_buffer) < WINDOW_SIZE:
            return "No Action", 0.0
        
        window = np.array(self.frame_buffer)
        velocity = calculate_velocity(window)
        self.is_moving = velocity > self.motion_threshold
        
        if not self.is_moving:
            return "No Action", 1.0
        
        X = window.reshape(1, WINDOW_SIZE, -1)
        predictions = self.model.predict(X, verbose=0)[0]
        
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        if confidence < CONFIDENCE_THRESHOLD:
            return "No Action", confidence
        
        predicted_class = PUNCH_CLASSES[predicted_class_idx]
        return predicted_class, confidence
    
    def run(self):
        """Main game loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Create fullscreen window
        cv2.namedWindow("Boxing Bop It", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Boxing Bop It", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        prev_time = time.time()
        last_punch = "No Action"
        frame_skip_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Extract landmarks
            landmarks, results = extract_landmarks(frame)
            
            if landmarks is not None:
                normalized = normalize_landmarks(landmarks)
                self.frame_buffer.append(normalized)
                frame = draw_landmarks(frame, results)
                
                # Run inference every 2 frames
                frame_skip_counter += 1
                if frame_skip_counter >= 2 and len(self.frame_buffer) == WINDOW_SIZE:
                    frame_skip_counter = 0
                    
                    predicted_class, confidence = self.predict()
                    stable_class = self.debounce_filter.update(predicted_class)
                    
                    self.current_prediction = stable_class
                    self.current_confidence = confidence
                    
                    # Game logic
                    if self.game_active and stable_class != "No Action" and stable_class != last_punch:
                        self.check_response(stable_class)
                        self.total_attempts += 1
                    
                    last_punch = stable_class
            
            # Check timeout
            if self.game_active:
                self.check_timeout()
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow("Boxing Bop It", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Toggle game
                if not self.game_active:
                    self.game_active = True
                    self.new_target()
                    print("Game started!")
                else:
                    self.game_active = False
                    print("Game paused")
            elif key == ord('r'):
                # Reset score
                self.score = 0
                self.total_attempts = 0
                self.streak = 0
                self.time_limit = 4.0  # Reset to 4.0 seconds
                print("Score reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n=== Game Over ===")
        accuracy = (self.score / self.total_attempts * 100) if self.total_attempts > 0 else 0
        print(f"Score: {self.score}/{self.total_attempts}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Best Streak: {self.best_streak}")
    
    def draw_ui(self, frame):
        """Draw game UI - professional, clean design"""
        h, w = frame.shape[:2]
        
        # Professional dark overlay bars
        overlay_color = (20, 20, 20)  # Dark gray instead of pure black
        cv2.rectangle(frame, (0, 0), (w, 140), overlay_color, -1)
        cv2.rectangle(frame, (0, h - 60), (w, h), overlay_color, -1)
        
        # Add subtle border lines
        cv2.line(frame, (0, 140), (w, 140), (60, 60, 60), 2)
        cv2.line(frame, (0, h - 60), (w, h - 60), (60, 60, 60), 2)
        
        if self.game_active:
            # === TOP SECTION - TWO COLUMN LAYOUT ===
            
            # LEFT COLUMN: Target Punch (large and clear)
            target_color = COLORS.get(self.current_target, (255, 255, 255))
            
            # "THROW:" label in white
            cv2.putText(frame, "THROW:", 
                       (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Punch name in color (below label)
            cv2.putText(frame, self.current_target.upper(), 
                       (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, target_color, 4, cv2.LINE_AA)
            
            # RIGHT COLUMN: Score & Streak (aligned right)
            accuracy = (self.score / self.total_attempts * 100) if self.total_attempts > 0 else 0
            
            # Score (top right)
            score_text = f"SCORE: {self.score}/{self.total_attempts}"
            cv2.putText(frame, score_text, 
                       (w - 350, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Accuracy percentage (below score)
            accuracy_text = f"{accuracy:.0f}% Accuracy"
            accuracy_color = (0, 255, 0) if accuracy >= 80 else (0, 165, 255) if accuracy >= 60 else (0, 100, 255)
            cv2.putText(frame, accuracy_text, 
                       (w - 350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accuracy_color, 2, cv2.LINE_AA)
            
            # Streak (below accuracy)
            streak_color = (0, 255, 0) if self.streak >= 3 else (200, 200, 200)
            streak_text = f"Streak: {self.streak}"
            if self.streak >= 5:
                streak_text += " 🔥"  # Fire emoji for hot streak
            cv2.putText(frame, streak_text, 
                       (w - 350, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, streak_color, 2, cv2.LINE_AA)
            
            # === TIMER BAR (full width, professional style) ===
            elapsed = time.time() - self.target_start_time
            time_left = max(0, self.time_limit - elapsed)
            
            # Color gradient based on time remaining
            if time_left > 2.5:
                time_color = (0, 255, 0)  # Green - plenty of time
            elif time_left > 1.5:
                time_color = (0, 200, 255)  # Orange - getting low
            else:
                time_color = (0, 50, 255)  # Red - hurry!
            
            # Background bar (gray)
            bar_y = 115
            bar_height = 18
            cv2.rectangle(frame, (30, bar_y), (w - 30, bar_y + bar_height), (60, 60, 60), -1)
            
            # Foreground bar (colored, shows time left)
            bar_width = int((time_left / self.time_limit) * (w - 60))
            if bar_width > 0:
                cv2.rectangle(frame, (30, bar_y), (30 + bar_width, bar_y + bar_height), time_color, -1)
            
            # Add border to timer bar
            cv2.rectangle(frame, (30, bar_y), (w - 30, bar_y + bar_height), (100, 100, 100), 2)
            
        else:
            # === PAUSED/WAITING STATE ===
            
            # Centered title
            title_text = "BOXING BOP IT"
            title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            title_x = (w - title_size[0]) // 2
            cv2.putText(frame, title_text, 
                       (title_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
            
            # Instruction (centered, below title)
            instruction = "Press SPACE to start!"
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            inst_x = (w - inst_size[0]) // 2
            cv2.putText(frame, instruction, 
                       (inst_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 150), 2, cv2.LINE_AA)
            
            # Last score (if exists)
            if self.total_attempts > 0:
                accuracy = (self.score / self.total_attempts * 100)
                score_summary = f"Last Round: {self.score}/{self.total_attempts} ({accuracy:.0f}%)"
                if self.best_streak > 0:
                    score_summary += f" | Best Streak: {self.best_streak}"
                
                summary_size = cv2.getTextSize(score_summary, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                summary_x = (w - summary_size[0]) // 2
                cv2.putText(frame, score_summary, 
                           (summary_x, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        
        # === BOTTOM BAR - Status Info ===
        
        # Left: Current Detection
        pred_color = COLORS.get(self.current_prediction, (150, 150, 150))
        detection_text = f"Detected: {self.current_prediction}"
        cv2.putText(frame, detection_text, 
                   (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2, cv2.LINE_AA)
        
        # Right: FPS
        fps_color = (0, 255, 0) if self.fps >= 25 else (0, 165, 255) if self.fps >= 15 else (0, 100, 255)
        cv2.putText(frame, f"FPS: {int(self.fps)}", 
                   (w - 120, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

if __name__ == "__main__":
    game = BoxingGame()
    game.run()