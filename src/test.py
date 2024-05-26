from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
import torchaudio
import torch

# Load the fine-tuned model, tokenizer, and feature extractor
model = WhisperForConditionalGeneration.from_pretrained("../trained_model")
tokenizer = WhisperTokenizer.from_pretrained("../trained_model")
feature_extractor = WhisperFeatureExtractor.from_pretrained("../trained_model")

# Set the model configuration for Russian language
model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ru", task="transcribe")

# Load and preprocess a test audio file
audio_input, sample_rate = torchaudio.load("/home/klutrem/Downloads/chto-vy-hotite.wav")

# Print original sample rate
print(f"Original sample rate: {sample_rate}")

# Resample if necessary
if sample_rate != 16000:
    print("Resampling audio to 16000 Hz")
    audio_input = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_input)

# Ensure the audio is a single channel
if audio_input.shape[0] > 1:
    print("Averaging multiple channels to create a single channel audio")
    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

# Extract features
input_features = feature_extractor(audio_input.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

# Print input features shape
print(f"Input features shape: {input_features.shape}")

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcription: {transcription}")
