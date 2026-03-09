import pandas as pd
import numpy as np

# --- Analysis Parameters ---
# The primary sit-up detection logic relies on the vertical distance between the hips and shoulders.
# This threshold may need to be adjusted depending on the specific video.
SITUP_MIN_TRAVEL = 0.05 

def evaluate_situps_from_data(landmarks_df, labels_df):
    """
    Analyzes sit-up reps from pre-extracted landmark data and compares against ground truth.
    
    Args:
        landmarks_df (pd.DataFrame): DataFrame containing MediaPipe landmark coordinates.
        labels_df (pd.DataFrame): DataFrame with 'pose_id' and 'pose' labels.
        
    Returns:
        float: The calculated accuracy of the sit-up counting logic.
    """
    try:
        # Merge landmark and label dataframes on 'pose_id'
        merged_df = pd.merge(landmarks_df, labels_df, on='pose_id')
        
        # Filter for only sit-up data based on the 'pose' column
        situp_df = merged_df[merged_df['pose'].str.contains('situp')].copy()
        
        if situp_df.empty:
            print("No sit-up data found in the dataset.")
            return 0.0
            
    except KeyError as e:
        print(f"Error: A column was not found. Please check your CSV file column names. {e}")
        return 0.0

    # --- Sit-up counting logic (re-implemented for dataframe analysis) ---
    is_up_position = False
    counted_reps = 0
    correctly_identified_frames = 0
    
    # Calculate hip-to-shoulder vertical distance for each frame
    # Using the corrected column names from your `landmarks.csv` file
    situp_df['hips_y'] = (situp_df['y_left_hip'] + situp_df['y_right_hip']) / 2
    situp_df['shoulders_y'] = (situp_df['y_left_shoulder'] + situp_df['y_right_shoulder']) / 2
    situp_df['vertical_travel'] = situp_df['hips_y'] - situp_df['shoulders_y']
    
    # Iterate through each frame of the sit-up data
    for index, row in situp_df.iterrows():
        # Our algorithm's prediction for the current frame
        is_down_position_predicted = row['vertical_travel'] > SITUP_MIN_TRAVEL

        # Check if the algorithm's prediction matches the ground truth
        predicted_state = "situps_down" if is_down_position_predicted else "situps_up"
        
        if row['pose'] == predicted_state:
            correctly_identified_frames += 1

        # Rep counting logic (same as the video analyzer)
        if is_down_position_predicted and not is_up_position:
            is_up_position = True

        if not is_down_position_predicted and is_up_position:
            is_up_position = False
            counted_reps += 1
            
    # --- Final Report Generation ---
    print("\n--- Sit-up Counting Report ---")
    
    # Accuracy: Percentage of frames correctly identified as 'up' or 'down'
    total_frames = len(situp_df)
    accuracy = (correctly_identified_frames / total_frames) * 100
    
    # The ground truth reps can be estimated by counting the transitions in the labels
    true_positive_reps = len(situp_df[situp_df['pose'] == 'situps_up'])
    
    print(f"Total Frames Analyzed: {total_frames}")
    print(f"Algorithm Accuracy (Frame-by-Frame): {accuracy:.2f}%")
    print(f"Repetitions Counted by Algorithm: {counted_reps}")
    print(f"Ground Truth Repetitions (from labels): {true_positive_reps}")
    
    if true_positive_reps > 0:
        precision = (min(true_positive_reps, counted_reps) / max(true_positive_reps, 1)) * 100
        print(f"Counting Precision: {precision:.2f}%")
    else:
        print("Cannot calculate precision: Ground truth has no 'situps_up' frames.")
    
    return accuracy

if __name__ == "__main__":
    try:
        # Load the dataset from the CSV files you provided
        landmarks_df = pd.read_csv('landmarks.csv')
        labels_df = pd.read_csv('labels.csv')
        
        # Start the evaluation
        evaluate_situps_from_data(landmarks_df, labels_df)
        
    except FileNotFoundError as e:
        print(f"Error: One of the required CSV files was not found. Please ensure all files are in the same directory as the script.")
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
