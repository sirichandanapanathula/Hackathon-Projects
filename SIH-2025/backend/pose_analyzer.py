import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Analysis Parameters ---
# A jump is detected when the hip landmark's y-coordinate rises above a certain threshold.
# The `y` coordinate is normalized from 0.0 (top) to 1.0 (bottom).
JUMP_DETECTION_THRESHOLD = 0.01 
HIP_LANDMARK_INDEX = mp_pose.PoseLandmark.LEFT_HIP.value # We'll track the left hip

# --- Data Storage for Analysis ---
hip_y_positions = []
jump_data = [] # To store individual jump metrics

# --- Main Video Processing Loop ---
def process_video(video_path):
    """
    Takes a video file as input, performs pose estimation, and displays the output.
    It tracks hip position to calculate jump height and time.
    """
    global hip_y_positions, jump_data
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
        is_in_jump = False
        jump_start_frame = -1
        
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
            
            hip_y = None
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Get the hip y-coordinate
                hip_y = results.pose_landmarks.landmark[HIP_LANDMARK_INDEX].y
                hip_y_positions.append(hip_y)
                
                # Simple jump detection logic
                if hip_y_positions and len(hip_y_positions) > 1:
                    y_diff = hip_y_positions[-2] - hip_y_positions[-1] # Negative when moving up
                    
                    if y_diff > JUMP_DETECTION_THRESHOLD and not is_in_jump:
                        is_in_jump = True
                        jump_start_frame = frame_number
                        
                    if y_diff < -JUMP_DETECTION_THRESHOLD and is_in_jump:
                        is_in_jump = False
                        # A jump has just completed, analyze its data
                        jump_end_frame = frame_number
                        analyze_jump(jump_start_frame, jump_end_frame, fps)
                        jump_start_frame = -1
            
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
def analyze_jump(start_frame, end_frame, fps):
    """Calculates jump height and duration for a single jump."""
    global hip_y_positions, jump_data
    
    jump_segment = hip_y_positions[start_frame:end_frame+1]
    
    if not jump_segment:
        return

    # Normalized y-coordinates: min value is highest point, max value is lowest point
    min_y = min(jump_segment) 
    max_y = max(jump_segment)
    
    # Calculate jump height in meters (assuming a real-world scale could be applied here)
    # The jump height is the difference between the lowest point (max_y) and highest point (min_y)
    jump_height_normalized = max_y - min_y
    
    # Convert normalized height to a more readable, pseudo-centimeter value for display
    # This is a conceptual conversion, not a real physical measurement.
    jump_height_cm = jump_height_normalized * 100 
    
    # Calculate time taken for the jump
    jump_duration_frames = end_frame - start_frame
    time_taken_s = jump_duration_frames / fps
    
    jump_data.append({
        "height_cm": jump_height_cm,
        "time_s": time_taken_s
    })

def display_metrics(image, width, height):
    """Draws the metrics on the OpenCV frame."""
    metrics_text = []
    
    if not jump_data:
        metrics_text.append("No jumps detected yet.")
    else:
        jump_heights = [j['height_cm'] for j in jump_data]
        
        max_jump = max(jump_heights)
        min_jump = min(jump_heights)
        avg_jump = np.mean(jump_heights)

        metrics_text.append(f"Max Jump: {max_jump:.2f} cm")
        metrics_text.append(f"Min Jump: {min_jump:.2f} cm")
        metrics_text.append(f"Avg Jump: {avg_jump:.2f} cm")
        
        # Display time for the most recent jump
        last_jump = jump_data[-1]
        metrics_text.append(f"Last Jump Time: {last_jump['time_s']:.2f} s")
    
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
    video_file_path = 'jumpy.mp4'
    
    # Reset data before starting
    hip_y_positions = []
    jump_data = []
    
    process_video(video_file_path)
