import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import joblib
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Analysis Parameters ---
DOWN_POSITION_ANGLE_THRESHOLD = 90
UP_POSITION_ANGLE_THRESHOLD = 160

class PushupAnalyzer:
    def __init__(self, model_path=None):
        """Initialize the analyzer with optional pre-trained model"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.is_trained = True
        else:
            self.is_trained = False
            
        # Data storage
        self.pushup_data = []
        self.training_data = []
        
        # Real-time analysis buffers
        self.angle_buffer = deque(maxlen=30)  # Store recent angles
        self.posture_buffer = deque(maxlen=10)  # Store recent posture predictions
        
    def calculate_angle(self, a, b, c):
        """Calculates the angle between three landmarks in degrees."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_body_alignment(self, landmarks):
        """Calculate body alignment score (straight line from head to feet)"""
        try:
            # Get key points for body alignment
            shoulder = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            hip = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])
            knee = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                           landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y])
            ankle = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                            landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            
            # Calculate deviation from straight line
            # Vector from shoulder to ankle
            shoulder_ankle = ankle - shoulder
            
            # Vector from shoulder to hip
            shoulder_hip = hip - shoulder
            
            # Vector from shoulder to knee
            shoulder_knee = knee - shoulder
            
            # Calculate perpendicular distances
            hip_deviation = np.abs(np.cross(shoulder_ankle, shoulder_hip)) / np.linalg.norm(shoulder_ankle)
            knee_deviation = np.abs(np.cross(shoulder_ankle, shoulder_knee)) / np.linalg.norm(shoulder_ankle)
            
            # Return average deviation (lower is better alignment)
            return (hip_deviation + knee_deviation) / 2
            
        except:
            return 0.5  # Default moderate alignment
    
    def extract_features(self, landmarks):
        """Extract features from pose landmarks for classification"""
        try:
            # Get coordinates
            left_shoulder = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                         landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
            left_hip = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                       landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate angles
            elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            shoulder_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
            hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            # Calculate body alignment
            body_alignment = self.calculate_body_alignment(landmarks)
            
            # Calculate additional metrics
            arm_extension = np.linalg.norm(np.array(left_wrist) - np.array(left_shoulder))
            body_height = abs(left_shoulder[1] - left_ankle[1])
            
            features = {
                'elbow_angle': elbow_angle,
                'shoulder_angle': shoulder_angle,
                'hip_angle': hip_angle,
                'body_alignment': body_alignment,
                'arm_extension': arm_extension,
                'body_height': body_height,
                'shoulder_hip_distance': np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)),
                'hip_knee_distance': np.linalg.norm(np.array(left_hip) - np.array(left_knee))
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def process_training_videos(self, dataset_path):
        """Process all training videos and extract features"""
        print("Processing training videos...")
        
        for label in ['correct', 'incorrect']:
            video_folder = os.path.join(dataset_path, label)
            
            if not os.path.exists(video_folder):
                print(f"Warning: Folder {video_folder} does not exist")
                continue
                
            video_files = [f for f in os.listdir(video_folder) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            print(f"Processing {len(video_files)} {label} videos...")
            
            for video_file in video_files:
                video_path = os.path.join(video_folder, video_file)
                print(f"  Processing: {video_file}")
                
                features_list = self.extract_features_from_video(video_path, label)
                self.training_data.extend(features_list)
        
        print(f"Extracted features from {len(self.training_data)} frames")
        
    def extract_features_from_video(self, video_path, label):
        """Extract features from a single video"""
        cap = cv2.VideoCapture(video_path)
        features_list = []
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to avoid redundancy
                if frame_count % 5 == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    
                    if results.pose_landmarks:
                        features = self.extract_features(results.pose_landmarks)
                        
                        if features:
                            features['label'] = label
                            features['video_file'] = os.path.basename(video_path)
                            features_list.append(features)
                
                frame_count += 1
        
        cap.release()
        return features_list
    
    def train_model(self):
        """Train the classification model using extracted features"""
        if len(self.training_data) == 0:
            print("No training data available. Please process videos first.")
            return
        
        print("Training model...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        # Prepare features and labels
        feature_columns = ['elbow_angle', 'shoulder_angle', 'hip_angle', 'body_alignment',
                          'arm_extension', 'body_height', 'shoulder_hip_distance', 'hip_knee_distance']
        
        X = df[feature_columns].fillna(0)  # Fill any NaN values
        y = df['label']
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print feature importance
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importance:")
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.4f}")
        
        self.is_trained = True
        
    def predict_posture(self, landmarks):
        """Predict if the current posture is correct or incorrect"""
        if not self.is_trained:
            return None, 0.0
        
        features = self.extract_features(landmarks)
        if features is None:
            return None, 0.0
        
        # Convert to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names])
        feature_array = feature_array.reshape(1, -1)
        
        # Scale and predict
        feature_scaled = self.scaler.transform(feature_array)
        prediction = self.model.predict(feature_scaled)[0]
        confidence = np.max(self.model.predict_proba(feature_scaled))
        
        return prediction, confidence
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            print("No trained model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def process_video(self, video_path, show_posture=True):
        """Process video with enhanced posture analysis"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            pushups_count = 0
            is_up_position = False
            rep_start_frame = -1
            frame_number = 0
            
            # Posture tracking
            correct_frames = 0
            total_analyzed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )

                    # Get landmark coordinates for the left arm
                    left_shoulder = [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                     results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    left_elbow = [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                  results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                    left_wrist = [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                                  results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                    
                    # Calculate the elbow angle
                    elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                    
                    # Push-up counting logic
                    if elbow_angle > UP_POSITION_ANGLE_THRESHOLD and not is_up_position:
                        is_up_position = True
                        pushups_count += 1
                        
                        if rep_start_frame != -1:
                            rep_end_frame = frame_number
                            time_taken_s = (rep_end_frame - rep_start_frame) / fps
                            self.pushup_data.append(time_taken_s)
                            rep_start_frame = -1

                    elif elbow_angle < DOWN_POSITION_ANGLE_THRESHOLD and is_up_position:
                        is_up_position = False
                        rep_start_frame = frame_number
                    
                    # Posture analysis
                    posture_prediction = None
                    posture_confidence = 0.0
                    
                    if show_posture and self.is_trained:
                        posture_prediction, posture_confidence = self.predict_posture(results.pose_landmarks)
                        
                        if posture_prediction is not None:
                            total_analyzed_frames += 1
                            if posture_prediction == 'correct':
                                correct_frames += 1
                
                # Display metrics
                self.display_enhanced_metrics(
                    image, pushups_count, posture_prediction, 
                    posture_confidence, correct_frames, total_analyzed_frames
                )
                
                cv2.imshow('Enhanced Push-up Analyzer', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_number += 1

        cap.release()
        cv2.destroyAllWindows()
        
        # Final analysis
        if total_analyzed_frames > 0:
            accuracy_percentage = (correct_frames / total_analyzed_frames) * 100
            print(f"\nFinal Analysis:")
            print(f"Total Push-ups: {pushups_count}")
            print(f"Posture Accuracy: {accuracy_percentage:.1f}%")
            
            if accuracy_percentage >= 80:
                print("Great form! Keep it up!")
            elif accuracy_percentage >= 60:
                print("Good form, with room for improvement.")
            else:
                print("Form needs significant improvement. Focus on proper alignment.")
    
    def display_enhanced_metrics(self, image, pushups_count, posture_prediction, 
                                posture_confidence, correct_frames, total_frames):
        """Display enhanced metrics including posture analysis"""
        metrics_text = []
        metrics_text.append(f"Push-up Reps: {pushups_count}")
        
        if pushups_count > 0 and self.pushup_data:
            rep_speeds = [1.0 / t for t in self.pushup_data]
            avg_rep_speed = np.mean(rep_speeds)
            fastest_rep_time = min(self.pushup_data)
            
            metrics_text.append(f"Avg Speed: {avg_rep_speed:.2f} reps/s")
            metrics_text.append(f"Fastest Rep: {fastest_rep_time:.2f} s")
        
        # Posture analysis
        if posture_prediction is not None:
            color_text = "GREEN" if posture_prediction == 'correct' else "RED"
            metrics_text.append(f"Posture: {posture_prediction.upper()} ({color_text})")
            metrics_text.append(f"Confidence: {posture_confidence:.2f}")
        
        if total_frames > 0:
            accuracy = (correct_frames / total_frames) * 100
            metrics_text.append(f"Form Accuracy: {accuracy:.1f}%")
        
        # Display text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        y0, dy = 30, 30
        for i, line in enumerate(metrics_text):
            y = y0 + i * dy
            
            # Color coding for posture
            if "Posture:" in line:
                color = (0, 255, 0) if "CORRECT" in line else (0, 0, 255)
            else:
                color = (255, 255, 255)
            
            cv2.putText(image, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

# Training script
def train_pushup_model():
    """Training pipeline for the push-up model"""
    
    # Initialize analyzer
    analyzer = PushupAnalyzer()
    
    # Set up dataset path - modify this to your dataset location
    dataset_path = "pushup_training_videos"  # Folder containing 'correct' and 'incorrect' subfolders
    
    print("=== Push-up Posture Classification Training ===")
    
    # Process training videos
    analyzer.process_training_videos(dataset_path)
    
    # Train model
    analyzer.train_model()
    
    # Save model
    model_path = "pushup_posture_model.pkl"
    analyzer.save_model(model_path)
    
    return analyzer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        print("Starting training mode...")
        train_pushup_model()
    else:
        # Analysis mode
        video_file_path = 'pushups.mp4'  # Change this to your video file
        model_path = "pushup_posture_model.pkl"  # Path to trained model
        
        # Initialize analyzer with trained model
        analyzer = PushupAnalyzer(model_path)
        
        # Reset data before starting
        analyzer.pushup_data = []
        
        # Process video with posture analysis
        analyzer.process_video(video_file_path, show_posture=True)
