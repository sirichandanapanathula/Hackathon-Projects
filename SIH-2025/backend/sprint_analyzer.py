import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Analysis Parameters ---
# We'll track the hip's x-coordinate to measure horizontal movement.
HIP_LANDMARK_INDEX = mp_pose.PoseLandmark.LEFT_HIP.value
# A sprint is detected when the hip's x-coordinate changes rapidly.
SPRINT_DETECTION_THRESHOLD = 0.005 # Sensitivity for detecting movement
# Conceptual factor to convert normalized distance (0 to 1) to meters.
# In a real app, this would be calibrated using a reference object in the video.
NORMALIZED_TO_METERS_FACTOR = 50.0

# --- Data Storage for Analysis ---
hip_x_positions = []
sprint_data = [] # To store individual sprint metrics

# --- Main Video Processing Loop ---
def process_video(video_path):
    """
    Takes a video file as input, performs pose estimation, and displays the output.
    It tracks hip position to calculate sprint speed, distance, and time.
    """
    global hip_x_positions, sprint_data
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties for analysis
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use a Pose model from MediaPipe
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_number = 0
        is_sprinting = False
        sprint_start_frame = -1
        sprint_start_x = -1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process the image to get pose landmarks
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Get the hip x-coordinate
                hip_x = results.pose_landmarks.landmark[HIP_LANDMARK_INDEX].x
                hip_x_positions.append(hip_x)
                
                # Sprint detection logic
                if len(hip_x_positions) > 1:
                    x_diff = abs(hip_x_positions[-1] - hip_x_positions[-2])
                    
                    # Start of a sprint
                    if x_diff > SPRINT_DETECTION_THRESHOLD and not is_sprinting:
                        is_sprinting = True
                        sprint_start_frame = frame_number
                        sprint_start_x = hip_x_positions[-1]

                    # End of a sprint (movement stops or slows down)
                    if x_diff < SPRINT_DETECTION_THRESHOLD and is_sprinting and (frame_number - sprint_start_frame) > (fps * 2):
                        is_sprinting = False
                        sprint_end_frame = frame_number
                        sprint_end_x = hip_x_positions[-1]
                        
                        # A sprint has just completed, analyze its data
                        analyze_sprint(sprint_start_frame, sprint_end_frame, sprint_start_x, sprint_end_x, fps)

            # Display metrics on the video frame
            display_metrics(image, frame_width, frame_height)
            
            # Display the processed frame
            cv2.imshow('MediaPipe Pose Detection', image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# --- Analysis Functions ---
def analyze_sprint(start_frame, end_frame, start_x, end_x, fps):
    """Calculates speed, distance, and duration for a single sprint."""
    global sprint_data
    
    sprint_duration_frames = end_frame - start_frame
    if sprint_duration_frames <= 0:
        return
        
    time_taken_s = sprint_duration_frames / fps
    
    # Calculate horizontal distance using normalized x-coordinates
    distance_normalized = abs(end_x - start_x)
    
    # Convert normalized distance to meters
    distance_m = distance_normalized * NORMALIZED_TO_METERS_FACTOR
    
    # Calculate speed in meters per second
    speed_mps = distance_m / time_taken_s
    
    sprint_data.append({
        "speed_mps": speed_mps,
        "distance_m": distance_m,
        "time_s": time_taken_s
    })

def display_metrics(image, width, height):
    """Draws the metrics on the OpenCV frame."""
    metrics_text = []
    
    if not sprint_data:
        metrics_text.append("No sprints detected yet.")
    else:
        speeds = [s['speed_mps'] for s in sprint_data]
        distances = [s['distance_m'] for s in sprint_data]
        
        max_speed = max(speeds)
        avg_speed = np.mean(speeds)
        total_distance = sum(distances)

        metrics_text.append(f"Max Speed: {max_speed:.2f} m/s")
        metrics_text.append(f"Avg Speed: {avg_speed:.2f} m/s")
        metrics_text.append(f"Total Distance: {total_distance:.2f} m")
        
        last_sprint = sprint_data[-1]
        metrics_text.append(f"Last Sprint Time: {last_sprint['time_s']:.2f} s")
    
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
    video_file_path = 'sprint.mp4'
    
    # Reset data before starting
    hip_x_positions = []
    sprint_data = []
    
    process_video(video_file_path)
