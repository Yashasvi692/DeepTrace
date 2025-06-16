# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array

# # Model and image parameters
# MODEL_PATH = os.path.join('models', 'saved_model', 'deepfake_model.h5')
# IMG_HEIGHT, IMG_WIDTH = 64, 64
# MAX_FRAMES = 10

# def load_model(model_path):
#     """Load the trained deepfake detection model"""
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print("Model loaded successfully")
#         return model
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         return None

# def extract_frames(video_path):
#     """Extract frames from video file"""
#     frames = []
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video file: {video_path}")

#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if frame_count == 0:
#             raise ValueError("Video file is empty")

#         step = max(1, frame_count // MAX_FRAMES)
        
#         for i in range(0, frame_count, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = img_to_array(frame) / 255.0
#                 frames.append(frame)
#             if len(frames) >= MAX_FRAMES:
#                 break
                
#     except Exception as e:
#         print(f"Error extracting frames: {str(e)}")
#         return []
#     finally:
#         cap.release()
    
#     return frames

# def predict_video(model, frames):
#     """Predict if video is real or fake"""
#     if not frames:
#         print("No frames to analyze")
#         return None, 0
    
#     predictions = []
#     for frame in frames:
#         try:
#             frame = np.expand_dims(frame, axis=0)
#             pred = model.predict(frame, verbose=0)[0][0]
#             predictions.append(pred)
#         except Exception as e:
#             print(f"Error making prediction: {str(e)}")
#             continue
    
#     if not predictions:
#         return None, 0
    
#     avg_pred = np.mean(predictions)
#     is_fake = avg_pred > 0.5
#     confidence = avg_pred if is_fake else (1 - avg_pred)
    
#     return 'FAKE' if is_fake else 'REAL', confidence * 100

# def main():
#     # Get video path from user
#     video_path = input("Enter the path to your video file: ").strip('"')
    
#     if not os.path.exists(video_path):
#         print(f"Error: File not found: {video_path}")
#         return
    
#     # Load the model
#     model = load_model(MODEL_PATH)
#     if model is None:
#         return
    
#     print("\nAnalyzing video...")
    
#     # Extract frames
#     frames = extract_frames(video_path)
#     if not frames:
#         print("Could not extract frames from video")
#         return
    
#     # Make prediction
#     result, confidence = predict_video(model, frames)
    
#     if result is None:
#         print("Could not make a prediction")
#         return
    
#     # Print results
#     print("\n" + "="*50)
#     print(f"Video Analysis Result:")
#     print(f"Classification: {result}")
#     print(f"Confidence: {confidence:.2f}%")
#     print("="*50)

# if __name__ == "__main__":
#     main()

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Model and image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match model's expected input size
MAX_FRAMES = 30  # Increased for better accuracy
BATCH_SIZE = 16

def load_model(model_path):
    """Load the trained deepfake detection model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_frame(frame):
    """Preprocess a single frame for prediction"""
    try:
        # Resize frame to match model's expected input size
        resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb.astype(float) / 255.0
        return normalized
    except Exception as e:
        print(f"Error preprocessing frame: {str(e)}")
        return None

def extract_frames(video_path):
    """Extract frames from video file"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            raise ValueError("Video file is empty")

        step = max(1, frame_count // MAX_FRAMES)
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                processed_frame = preprocess_frame(frame)
                if processed_frame is not None:
                    frames.append(processed_frame)
                if len(frames) >= MAX_FRAMES:
                    break
                
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return []
    finally:
        cap.release()
    
    return frames

def predict_video(model, frames):
    """Predict if video is real or fake"""
    if not frames:
        print("No frames to analyze")
        return None, 0
    
    try:
        # Convert frames list to numpy array
        frames_array = np.array(frames)
        
        # Make predictions in batches
        predictions = []
        for i in range(0, len(frames_array), BATCH_SIZE):
            batch = frames_array[i:i + BATCH_SIZE]
            preds = model.predict(batch, verbose=0)
            predictions.extend(preds.flatten())
        
        if not predictions:
            return None, 0
        
        # Calculate final prediction and confidence
        avg_pred = np.mean(predictions)
        confidence = abs(0.5 - avg_pred) * 2  # Scale confidence to 0-1
        is_fake = avg_pred < 0.5  # Adjust threshold based on your model
        
        return 'FAKE' if is_fake else 'REAL', confidence * 100
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, 0

def main():
    # Configure paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = "models\saved_model\deepfake_model.h5"
    
    # Get video path from user
    video_path = input("Enter the path to your video file: ").strip('"')
    
    # Verify paths
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    print("\nAnalyzing video...")
    
    # Extract and process frames
    frames = extract_frames(video_path)
    if not frames:
        print("Could not extract frames from video")
        return
    
    # Make prediction
    result, confidence = predict_video(model, frames)
    
    if result is None:
        print("Could not make a prediction")
        return
    
    # Print results
    print("\n" + "="*50)
    print("Video Analysis Results:")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()