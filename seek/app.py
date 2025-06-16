# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, redirect, url_for
# from werkzeug.utils import secure_filename
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'static/uploads/'
# MODEL_PATH = 'models/saved_model/deepfake_model.h5'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Image parameters
# IMG_HEIGHT, IMG_WIDTH = 224,224

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_frames(video_path, max_frames=10):
#     """Extract frames from video."""
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     step = max(1, frame_count // max_frames)
    
#     for i in range(0, frame_count, step):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = img_to_array(frame) / 255.0
#             frames.append(frame)
#         if len(frames) >= max_frames:
#             break
    
#     cap.release()
#     return frames

# def predict_video(frames):
#     """Predict if video is real or fake based on frames."""
#     predictions = []
#     for frame in frames:
#         frame = np.expand_dims(frame, axis=0)
#         pred = model.predict(frame)[0][0]
#         predictions.append(pred)
    
#     # Average predictions; threshold at 0.5
#     avg_pred = np.mean(predictions)
#     return 'Real' if avg_pred <= 0.5 else 'Fake', avg_pred

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#         file.save(file_path)
        
#         # Extract frames and predict
#         frames = extract_frames(file_path)
#         if not frames:
#             return render_template('result.html', prediction='Error: Could not process video', confidence=0)
        
#         prediction, confidence = predict_video(frames)
#         confidence = (1 - confidence) * 100 if prediction == 'Real' else confidence * 100
        
#         # Clean up uploaded file
#         os.remove(file_path)
        
#         return render_template('result.html', prediction=prediction, confidence=f"{confidence:.2f}%")
    
#     return redirect(request.url)

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import cv2
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_model', 'deepfake_model.h5')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
MAX_FRAMES = 20

def save_analysis_results(prediction, confidence, video_name, frames_analyzed):
    """Save analysis results to a JSON file"""
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"analysis_{timestamp}.json")
    
    results = {
        "video_name": video_name,
        "timestamp": timestamp,
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "frames_analyzed": frames_analyzed,
        "analysis_date": datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return result_file

# Load the model
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, max_frames=MAX_FRAMES):
    """Extract frames from video."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return frames

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            logger.error("Empty video file")
            return frames

        step = max(1, frame_count // max_frames)
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = img_to_array(frame) / 255.0
                frames.append(frame)
            if len(frames) >= max_frames:
                break

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
    finally:
        cap.release()
    
    return frames

def predict_video(frames):
    """Predict if video is real or fake based on frames."""
    if not frames:
        return None, 0
    
    try:
        predictions = []
        for frame in frames:
            frame = np.expand_dims(frame, axis=0)
            pred = model.predict(frame, verbose=0)[0][0]
            predictions.append(pred)
        
        avg_pred = np.mean(predictions)
        confidence = abs(0.5 - avg_pred) * 2
        
        return 'Real' if avg_pred <= 0.5 else 'Fake', confidence * 100
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        flash('Model not loaded. Please check server configuration.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            frames = extract_frames(file_path)
            if not frames:
                flash('Error: Could not process video', 'error')
                return redirect(url_for('index'))
            
            prediction, confidence = predict_video(frames)
            if prediction is None:
                flash('Error: Could not analyze video', 'error')
                return redirect(url_for('index'))
            
            # Save analysis results
            analysis_file = save_analysis_results(
                prediction=prediction,
                confidence=confidence,
                video_name=filename,
                frames_analyzed=len(frames)
            )
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return render_template('result.html', 
                                 prediction=prediction, 
                                 confidence=f"{confidence:.2f}%",
                                 analysis_file=os.path.basename(analysis_file))
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            flash(f'Error processing video: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if model is None:
        logger.warning("Model not loaded. Please check the model path.")
    app.run(debug=True, host='0.0.0.0', port=5000)