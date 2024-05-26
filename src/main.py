import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from datasets import Audio
import torch
import torchaudio

app = FastAPI()

# Define the directory where your model, tokenizer, and feature extractor are saved
model_dir = "./trained_model"

# Load the fine-tuned model, tokenizer, and feature extractor
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
tokenizer = WhisperTokenizer.from_pretrained(model_dir)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

# Set the model configuration for Russian language
model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ru", task="transcribe")

def save_file_tmp(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(prefix="uploaded_", suffix=extract_file_extension(file.filename), delete=False) as temp_file:
        temp_file.write(file.file.read())
        return temp_file.name

def extract_file_extension(filename: str) -> str:
    _, file_extension = os.path.splitext(filename)
    return file_extension

@app.get("/")
async def read_root():
    return "To recognize an audio file, upload it using a POST request with the '/recognize' route."

@app.post('/recognize')
async def recognize(file: UploadFile = File(...)):
    try:
        file_path = save_file_tmp(file)
        
        # Load the audio file
        audio_input, sample_rate = torchaudio.load(file_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_input = transform(audio_input)
        
        # Ensure the audio is a single channel
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)
        
        # Extract features
        input_features = feature_extractor(audio_input.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

        # Generate transcription
        predicted_ids = model.generate(input_features)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return {"transcription": transcription}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
