import os
import tempfile
from fastapi import FastAPI, File, UploadFile
import whisper

app = FastAPI()
model_path = "./trained_model/model.pt"
model = whisper.load_model(model_path)

def save_file_tmp(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(prefix="uploaded_", suffix=extract_file_extension(file.filename), delete=False) as temp_file:
        temp_file.write(file.file.read())
        return temp_file.name

def extract_file_extension(filename: str) -> str:
    _, file_extension = os.path.splitext(filename)
    return file_extension

@app.get("/")
async def read_root():
    return "To recognize an audio file, upload it using a POST request with '/recognize' route."

@app.post('/recognize')
def recognize(file: UploadFile = File(...)):
    file_path = save_file_tmp(file)
    result = model.transcribe(file_path)
    os.remove(file_path)  
    return result
