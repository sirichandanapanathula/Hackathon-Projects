import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Analysis Parameters ---
# The primary sit-up detection logic relies on the angle of the hips relative to the shoulders.
# These angles are based on the vertical position of the nose, shoulders, and hips.
# `y` is normalized from 0.0 (top) to 1.0 (bottom).
# An up position is when the hips and shoulders are close to each other vertically.
# A down position is when they are far apart.
UP_POSITION_Y_THRESHOLD = 0.5  
DOWN_POSITION_Y_THRESHOLD = 0.65
# The hips must move a certain vertical distance to count as a full sit-up.
SITUP_MIN_TRAVEL = 0.05 

# --- Data Storage for Analysis ---
situp_data = []  # To store individual sit-up metrics (rep time)

# --- Helper function to calculate angle ---
def calculate_angle(a, b, c):
    """Calculates the angle between three landmarks in degrees."""
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Mid point (hip)
    c = np.array(c)  # End point (knee)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Video Processing Loop ---
def process_video(video_path, ground_truth_reps=None):
    """
    Takes a video file as input, performs pose estimation, and displays the output.
    It tracks key landmarks to count sit-up repetitions and measure time per rep.
    """
    global situp_data
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_number = 0
        situps_count = 0
        is_up_position = False
        rep_start_frame = -1
        
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
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Get y-coordinates for key landmarks
                nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                shoulders_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                hips_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y +
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                
                # --- Sit-up counting logic ---
                is_down_position = (hips_y - shoulders_y) > SITUP_MIN_TRAVEL

                if is_down_position and not is_up_position:
                    is_up_position = True
                    rep_start_frame = frame_number

                if not is_down_position and is_up_position and rep_start_frame != -1:
                    is_up_position = False
                    situps_count += 1
                    
                    rep_end_frame = frame_number
                    time_taken_s = (rep_end_frame - rep_start_frame) / fps
                    situp_data.append(time_taken_s)
                    rep_start_frame = -1
            
            # Display metrics on the video frame
            display_metrics(image, frame_width, frame_height, situps_count, fps)
            
            cv2.imshow('MediaPipe Sit-up Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # --- Evaluation at the end of the video ---
    if ground_truth_reps is not None:
        evaluate_accuracy(situps_count, ground_truth_reps)

# --- Analysis Functions ---
def display_metrics(image, width, height, situps_count, fps):
    """Draws the metrics on the OpenCV frame."""
    metrics_text = []
    metrics_text.append(f"Sit-up Reps: {situps_count}")
    
    if situps_count > 0 and situp_data:
        rep_speeds = [1.0 / t for t in situp_data]
        avg_rep_speed = np.mean(rep_speeds)
        fastest_rep_time = min(situp_data)
        
        metrics_text.append(f"Average Speed: {avg_rep_speed:.2f} reps/s")
        metrics_text.append(f"Fastest Rep: {fastest_rep_time:.2f} s")
    
    # Display the text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)  # White color
    
    y0, dy = 30, 30
    for i, line in enumerate(metrics_text):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

def evaluate_accuracy(counted_reps, ground_truth_reps):
    """Calculates and prints the accuracy and precision of the counting logic."""
    print("\n--- Model Evaluation ---")
    print(f"Ground Truth Sit-ups: {ground_truth_reps}")
    print(f"Counted by Algorithm: {counted_reps}")
    
    # Accuracy can be defined as how close the count is to the ground truth
    if ground_truth_reps > 0:
        accuracy = (1 - abs(counted_reps - ground_truth_reps) / ground_truth_reps) * 100
    else:
        accuracy = 100 if counted_reps == 0 else 0
        
    # Precision is typically about avoiding false positives.
    # We can simplify this here as the ratio of correct reps to the total counted reps.
    # If counted reps is greater than ground truth, some are false positives.
    precision = min(1.0, counted_reps / ground_truth_reps) * 100 if ground_truth_reps > 0 else 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")

if __name__ == "__main__":
    # --- Integration from your Dataset ---
    # 1. Download a video from a dataset like the one on Kaggle.
    #    Example: 'Exercise Recognition Dataset' on Kaggle.
    # 2. Manually count the number of sit-ups in your chosen video.
    
    # Set the path to your video file
    VIDEO_FILE_PATH = 'situps.mp4'
    
    # Set the ground truth (manually counted) number of sit-ups in the video
    GROUND_TRUTH_REPS = 2
    
    # Reset data before starting
    situp_data = []
    
    # Pass the ground truth value to the main processing function
    process_video(VIDEO_FILE_PATH, GROUND_TRUTH_REPS)
