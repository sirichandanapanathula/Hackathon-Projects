from push import train_pushup_model

if __name__ == "__main__":
    print("Training Push-up Posture Classification Model...")
    print("Make sure your videos are organized as:")
    print("  pushup_training_videos/correct/")
    print("  pushup_training_videos/incorrect/")
    
    # Train the model
    analyzer = train_pushup_model()
    
    print("Training completed!")
    print("Model saved as 'pushup_posture_model.pkl'")
