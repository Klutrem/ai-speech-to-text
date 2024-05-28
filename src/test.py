from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
import torchaudio
import torch

import whisper

model_path = "./trained_model/model.pt"

model = whisper.load_model(model_path)
tokenizer = WhisperTokenizer.from_pretrained("trained_model")
feature_extractor = WhisperFeatureExtractor.from_pretrained("trained_model")

audio_input, sample_rate = torchaudio.load("58.mp3")

print(f"Original sample rate: {sample_rate}")


if sample_rate != 16000:
    print("Resampling audio to 16000 Hz")
    audio_input = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_input)

if audio_input.shape[0] > 1:
    print("Averaging multiple channels to create a single channel audio")
    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

input_features = feature_extractor(audio_input.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

print(f"Input features shape: {input_features.shape}")

transcription = model.transcribe("776.mp3")

print(f"Transcription: {transcription}")
