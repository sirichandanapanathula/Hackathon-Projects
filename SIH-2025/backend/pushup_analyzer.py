import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Analysis Parameters ---
# The core logic for push-up counting is based on the angle of the elbow.
# A push-up is considered a full rep when the elbow angle changes significantly
# between the 'down' and 'up' positions.
DOWN_POSITION_ANGLE_THRESHOLD = 90  # Angle of the elbow when the body is lowered
UP_POSITION_ANGLE_THRESHOLD = 160   # Angle of the elbow when the body is raised

# --- Data Storage for Analysis ---
pushup_data = []  # To store individual push-up metrics (rep time)

# --- Helper function to calculate angle ---
def calculate_angle(a, b, c):
    """Calculates the angle between three landmarks in degrees."""
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Mid point (elbow)
    c = np.array(c)  # End point (wrist)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Video Processing Loop ---
def process_video(video_path):
    """
    Takes a video file as input, performs pose estimation, and displays the output.
    It tracks key landmarks to count push-up repetitions and measure time per rep.
    """
    global pushup_data
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use a Pose model from MediaPipe
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        pushups_count = 0
        is_up_position = False
        rep_start_frame = -1
        frame_number = 0
        
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

                # Get landmark coordinates for the left arm
                left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
                
                # Calculate the elbow angle
                elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # --- Push-up counting logic (state machine) ---
                if elbow_angle > UP_POSITION_ANGLE_THRESHOLD and not is_up_position:
                    # Transition from down to up
                    is_up_position = True
                    pushups_count += 1
                    
                    if rep_start_frame != -1:
                        rep_end_frame = frame_number
                        time_taken_s = (rep_end_frame - rep_start_frame) / fps
                        pushup_data.append(time_taken_s)
                        rep_start_frame = -1

                elif elbow_angle < DOWN_POSITION_ANGLE_THRESHOLD and is_up_position:
                    # Transition from up to down
                    is_up_position = False
                    rep_start_frame = frame_number
            
            # Display metrics on the video frame
            display_metrics(image, frame_width, frame_height, pushups_count, fps)
            
            cv2.imshow('MediaPipe Push-up Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# --- Analysis Functions ---
def display_metrics(image, width, height, pushups_count, fps):
    """Draws the metrics on the OpenCV frame."""
    metrics_text = []
    metrics_text.append(f"Pushup Reps: {pushups_count}")
    
    if pushups_count > 0 and pushup_data:
        rep_speeds = [1.0 / t for t in pushup_data]
        avg_rep_speed = np.mean(rep_speeds)
        fastest_rep_time = min(pushup_data)
        
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

if __name__ == "__main__":
    # Change the path to your video file here
    video_file_path = 'pushups.mp4'
    
    # Reset data before starting
    pushup_data = []
    
    process_video(video_file_path)
