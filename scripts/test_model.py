# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # import numpy as np
# # from sklearn.metrics import classification_report, confusion_matrix
# # import os

# # # Define paths
# # test_dir = '../data/test'
# # model_path = 'models/saved_model/deepfake_model.h5'

# # # Image parameters
# # IMG_HEIGHT, IMG_WIDTH = 224, 224
# # BATCH_SIZE = 32

# # # Data generator for testing (no augmentation)
# # test_datagen = ImageDataGenerator(rescale=1./255)

# # # Load test data
# # test_generator = test_datagen.flow_from_directory(
# #     test_dir,
# #     target_size=(IMG_HEIGHT, IMG_WIDTH),
# #     batch_size=BATCH_SIZE,
# #     class_mode='binary',
# #     classes=['fake', 'real'],
# #     shuffle=False
# # )

# # # Load trained model
# # model = tf.keras.models.load_model(model_path)

# # # Evaluate model
# # test_loss, test_accuracy = model.evaluate(test_generator)
# # print(f"Test Accuracy: {test_accuracy*100:.2f}%")
# # print(f"Test Loss: {test_loss:.4f}")

# # # Predict on test data
# # predictions = model.predict(test_generator)
# # predicted_classes = (predictions > 0.5).astype(int).flatten()
# # true_classes = test_generator.classes

# # # Print classification report
# # print("\nClassification Report:")
# # print(classification_report(true_classes, predicted_classes, target_names=['fake', 'real']))

# # # Print confusion matrix
# # print("\nConfusion Matrix:")
# # print(confusion_matrix(true_classes, predicted_classes))


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import os

# # Configure absolute paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test_dir = os.path.join(BASE_DIR, 'data', 'testsubset')
# model_path = os.path.join(BASE_DIR, 'models', 'saved_model', 'deepfake_model.h5')

# # Verify paths exist
# if not os.path.exists(test_dir):
#     raise FileNotFoundError(f"Test directory not found: {test_dir}")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found: {model_path}")

# # Image parameters - match training dimensions
# IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match the dimensions used in training
# BATCH_SIZE = 16  # Smaller batch size for memory efficiency

# try:
#     # Data generator for testing
#     test_datagen = ImageDataGenerator(rescale=1./255)

#     # Load test data
#     test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(IMG_HEIGHT, IMG_WIDTH),
#         batch_size=BATCH_SIZE,
#         class_mode='binary',
#         classes=['fake', 'real'],
#         shuffle=False
#     )

#     # Verify we have test data
#     if test_generator.samples == 0:
#         raise ValueError("No test images found in the specified directory")

#     print(f"Found {test_generator.samples} images for testing")

#     # Load and evaluate model
#     print("Loading model...")
#     model = tf.keras.models.load_model(model_path)
    
#     print("\nEvaluating model...")
#     test_loss, test_accuracy = model.evaluate(
#         test_generator,
#         verbose=1
#     )
#     print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Generate predictions
#     print("\nGenerating predictions...")
#     predictions = model.predict(
#         test_generator,
#         verbose=1
#     )
#     predicted_classes = (predictions > 0.5).astype(int).flatten()
#     true_classes = test_generator.classes

#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(
#         true_classes,
#         predicted_classes,
#         target_names=['fake', 'real']
#     ))

#     # Print confusion matrix
#     print("\nConfusion Matrix:")
#     conf_matrix = confusion_matrix(true_classes, predicted_classes)
#     print(conf_matrix)
    
#     # Print interpretation of confusion matrix
#     tn, fp, fn, tp = conf_matrix.ravel()
#     print("\nDetailed Results:")
#     print(f"True Negatives (Correct Fake Detection): {tn}")
#     print(f"False Positives (Incorrect Real Detection): {fp}")
#     print(f"False Negatives (Incorrect Fake Detection): {fn}")
#     print(f"True Positives (Correct Real Detection): {tp}")

# except Exception as e:
#     print(f"Error during testing: {str(e)}")
#     raise


# import tensorflow as tf
# import cv2
# import numpy as np
# import os

# # Model parameters
# IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match the model's expected input size

# def load_model(model_path):
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print("Model loaded successfully\n")
#         return model
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         return None

# def preprocess_frame(frame):
#     try:
#         # Resize frame to match model's expected input size
#         resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
#         # Convert to RGB (from BGR)
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # Normalize pixel values
#         normalized = rgb.astype(float) / 255.0
#         # Add batch dimension
#         return np.expand_dims(normalized, axis=0)
#     except Exception as e:
#         print(f"Error preprocessing frame: {str(e)}")
#         return None

# def analyze_video(video_path, model):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Could not open video file")

#         frame_count = 0
#         predictions = []
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             # Process every 30th frame
#             if frame_count % 30 == 0:
#                 processed_frame = preprocess_frame(frame)
#                 if processed_frame is not None:
#                     try:
#                         pred = model.predict(processed_frame, verbose=0)
#                         predictions.append(pred[0][0])
#                     except Exception as e:
#                         print(f"Error making prediction: {str(e)}")
            
#             frame_count += 1
            
#         cap.release()
        
#         if predictions:
#             avg_prediction = np.mean(predictions)
#             confidence = abs(0.5 - avg_prediction) * 2  # Scale confidence to 0-1
            
#             print("\nAnalysis Results:")
#             print(f"Processed {len(predictions)} frames")
#             if avg_prediction > 0.5:
#                 print(f"Result: REAL video")
#                 print(f"Confidence: {confidence*100:.2f}%")
#             else:
#                 print(f"Result: FAKE video")
#                 print(f"Confidence: {confidence*100:.2f}%")
#         else:
#             print("Could not make a prediction")
            
#     except Exception as e:
#         print(f"Error analyzing video: {str(e)}")

# def main():
#     # Configure model path
#     model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
#                              'models', 'saved_model', 'deepfake_model.h5')
    
#     # Get video path from user
#     video_path = input("Enter the path to your video file: ").strip('"')  # Remove quotes if present
    
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found at {video_path}")
#         return
        
#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         return
    
#     # Load model
#     model = load_model(model_path)
#     if model is None:
#         return
        
#     print("Analyzing video...")
#     analyze_video(video_path, model)

# if __name__ == "__main__":
#     main()



import tensorflow as tf
import cv2
import numpy as np
import os
from datetime import datetime

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
MIN_FRAMES = 40  # Minimum number of frames to analyze

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully\n")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_frame(frame):
    try:
        # Resize frame to match model's expected input size
        resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb.astype(float) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        print(f"Error preprocessing frame: {str(e)}")
        return None

def save_results(video_path, predictions, is_fake, confidence):
    """Save analysis results to a JSON file"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"analysis_{timestamp}.json")
    
    results = {
        "video_name": os.path.basename(video_path),
        "timestamp": timestamp,
        "frames_analyzed": len(predictions),
        "frame_predictions": [float(p) for p in predictions],
        "classification": "FAKE" if is_fake else "REAL",
        "confidence": f"{confidence:.2f}%",
        "analysis_date": datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    
    return result_file

def analyze_video(video_path, model):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < MIN_FRAMES:
            raise ValueError(f"Video too short. Needs at least {MIN_FRAMES} frames")

        # Calculate frame sampling rate to get at least MIN_FRAMES
        sample_rate = max(1, total_frames // MIN_FRAMES)
        
        frame_count = 0
        predictions = []
        processed_count = 0
        
        print("\nAnalyzing frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frames based on calculated sample rate
            if frame_count % sample_rate == 0:
                processed_frame = preprocess_frame(frame)
                if processed_frame is not None:
                    try:
                        pred = model.predict(processed_frame, verbose=0)
                        pred_value = float(pred[0][0])
                        predictions.append(pred_value)
                        processed_count += 1
                        print(f"Processed frame {processed_count}/{MIN_FRAMES}", end='\r')
                    except Exception as e:
                        print(f"\nError making prediction: {str(e)}")
            
            frame_count += 1
            if processed_count >= MIN_FRAMES:
                break
            
        cap.release()
        
        if len(predictions) >= MIN_FRAMES:
            avg_prediction = np.mean(predictions)
            confidence = abs(0.5 - avg_prediction) * 2 * 100  # Scale to percentage
            
            # Calculate additional statistics
            std_dev = np.std(predictions)
            fake_frames = sum(1 for p in predictions if p > 0.5)
            real_frames = len(predictions) - fake_frames
            
            print("\n\nAnalysis Results:")
            print(f"Total frames processed: {len(predictions)}")
            print(f"Average prediction value: {avg_prediction:.3f}")
            print(f"Prediction std deviation: {std_dev:.3f}")
            print(f"Frames classified as fake: {fake_frames}")
            print(f"Frames classified as real: {real_frames}")
            
            # Make final decision
            is_fake = avg_prediction > 0.5
            
            # Save results
            result_file = save_results(video_path, predictions, is_fake, confidence)
            print(f"\nDetailed analysis saved to: {os.path.basename(result_file)}")
            
            print(f"\nFinal Result: {'FAKE' if is_fake else 'REAL'} video")
            print(f"Confidence: {confidence:.2f}%")
            
            return is_fake, confidence
            
        else:
            print(f"\nError: Could not analyze enough frames. Only processed {len(predictions)} frames")
            return None, 0
            
    except Exception as e:
        print(f"\nError analyzing video: {str(e)}")
        return None, 0

def main():
    # Configure model path
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'models', 'saved_model', 'deepfake_model.h5')
    
    # Get video path from user
    video_path = input("Enter the path to your video file: ").strip('"')  # Remove quotes if present
    
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
        
    print("Analyzing video...")
    analyze_video(video_path, model)

if __name__ == "__main__":
    main()