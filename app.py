# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
import uvicorn
from typing import Dict, Any
import logging
import soundfile as sf
import wave
import audioop

# Load environment variables
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Voice Emotion Recognition API")

# Add CORS middleware - Updated for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mindbridge-alpha.vercel.app/",  # Replace with your actual Vercel URL
        "https://*.vercel.app",  # Allow all Vercel subdomains
        "http://localhost:3000",  # For local development
        "https://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load the model at startup
model_path = "./model.keras"  # Ensure this file is in your repo
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def is_valid_wav(file_path):
    """Check if a WAV file is valid by trying to open it with wave"""
    try:
        with wave.open(file_path, 'rb') as wf:
            return True
    except Exception as e:
        logger.error(f"Invalid WAV file: {str(e)}")
        return False

def fix_wav_header(file_path):
    """Attempt to fix WAV file header issues by rewriting it"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        if not data.startswith(b'RIFF'):
            logger.warning("WAV file doesn't start with RIFF header. Attempting to fix...")
            
            with open(file_path + '.fixed.wav', 'wb') as f:
                channels = 1
                sample_width = 2
                sample_rate = 44100
                
                audio_data = data
                
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
            return file_path + '.fixed.wav'
    except Exception as e:
        logger.error(f"Error fixing WAV header: {str(e)}")
    
    return file_path

def extract_features(file_path, n_mfcc=40):
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing audio file: {file_path}, size: {file_size} bytes")
        
        if not is_valid_wav(file_path):
            fixed_path = fix_wav_header(file_path)
            if fixed_path != file_path:
                logger.info(f"Using fixed WAV file: {fixed_path}")
                file_path = fixed_path
        
        try:
            audio, sample_rate = sf.read(file_path)
            logger.info(f"Loaded audio with soundfile, shape: {audio.shape}, sample_rate: {sample_rate}")
        except Exception as sf_error:
            logger.warning(f"Soundfile failed: {str(sf_error)}, trying librosa...")
            try:
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
                logger.info(f"Loaded audio with librosa, length: {len(audio)}, sample_rate: {sample_rate}")
            except Exception as librosa_error:
                logger.error(f"Librosa also failed: {str(librosa_error)}")
                try:
                    with wave.open(file_path, 'rb') as wf:
                        frames = wf.getnframes()
                        buffer = wf.readframes(frames)
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        channels = wf.getnchannels()
                        
                        if channels == 2:
                            buffer = audioop.tomono(buffer, sample_width, 0.5, 0.5)
                        
                        if sample_width == 2:
                            audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:
                            audio = np.frombuffer(buffer, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            audio = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        
                        logger.info(f"Loaded audio with wave, length: {len(audio)}, sample_rate: {sample_rate}")
                except Exception as wave_error:
                    logger.error(f"All audio loading methods failed: {str(wave_error)}")
                    raise Exception(f"Could not load audio with any method: {str(sf_error)} | {str(librosa_error)} | {str(wave_error)}")
        
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        logger.info(f"Audio prepared: shape={audio.shape}, sample_rate={sample_rate}")
        
        if len(audio) < sample_rate * 0.5:
            logger.warning("Audio too short, padding")
            audio = np.pad(audio, (0, int(sample_rate * 0.5) - len(audio)), 'constant')
        
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            logger.info(f"MFCCs extracted: shape={mfccs.shape}")
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {str(e)}")
            logger.warning("Using fallback dummy features")
            mfccs = np.random.rand(n_mfcc, max(1, int(len(audio) / 512)))
        
        mfccs = np.mean(mfccs, axis=1).reshape(n_mfcc, 1)
        return mfccs
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Voice Emotion Recognition API is running", 
        "status": "healthy",
        "model_loaded": model is not None
    }

# Health check endpoint for Azure
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.post("/predict")
async def predict_emotion(audio_file: UploadFile = File(...), expected_word: str = Form(None), category: str = Form(None)) -> Dict[str, Any]:
    global model
    
    logger.info(f"Received prediction request: file={audio_file.filename}, content_type={audio_file.content_type}")
    logger.info(f"Parameters: expected_word={expected_word}, category={category}")
    
    if model is None:
        try:
            logger.info(f"Attempting to load model from {model_path}")
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Model could not be loaded: {str(e)}")
            return {
                "filename": audio_file.filename,
                "predicted_emotion": "neutral",
                "confidence_scores": {
                    "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                    "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
                },
                "text": "Model not available",
                "error": str(e),
                "matches": False
            }
    
    valid_audio_types = [".wav", ".webm", ".mp3", ".ogg", ".m4a"]
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if not file_ext and audio_file.content_type == 'audio/wav':
        file_ext = '.wav'
    
    if not file_ext and not any(audio_file.content_type.startswith(f"audio/{t.replace('.', '')}") for t in valid_audio_types):
        logger.warning(f"Unsupported content type: {audio_file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_audio_types)}"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)
        
        logger.info(f"Saved uploaded file to {temp_file_path}, size: {len(content)} bytes")
        
        try:
            features = extract_features(temp_file_path)
            features = np.expand_dims(features, axis=0)
            logger.info(f"Features extracted successfully, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {
                "filename": audio_file.filename,
                "predicted_emotion": "neutral",
                "confidence_scores": {
                    "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                    "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
                },
                "text": "Could not process audio. Please try again and speak clearly.",
                "error": str(e),
                "matches": False
            }
        
        logger.info("Making prediction")
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)
        
        emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        confidence_scores = prediction[0].tolist()
        
        logger.info(f"Prediction result: {emotions[predicted_label]}")
        
        try:
            os.unlink(temp_file_path)
            if os.path.exists(temp_file_path + '.fixed.wav'):
                os.unlink(temp_file_path + '.fixed.wav')
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")
        
        return {
            "filename": audio_file.filename,
            "predicted_emotion": emotions[predicted_label],
            "confidence_scores": {
                emotions[i]: float(confidence_scores[i]) for i in range(len(emotions))
            },
            "text": "placeholder for transcription",
            "matches": True if predicted_label == 0 else False
        }
        
    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                if os.path.exists(temp_file_path + '.fixed.wav'):
                    os.unlink(temp_file_path + '.fixed.wav')
            except:
                pass
        
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        return {
            "filename": audio_file.filename,
            "predicted_emotion": "neutral",
            "confidence_scores": {
                "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
            },
            "text": "Error processing your speech. Please try again.",
            "error": str(e),
            "matches": False
        }

# Azure App Service expects the app to be available as 'app'
if __name__ == '__main__':
    # Use PORT environment variable if available (Azure), otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)