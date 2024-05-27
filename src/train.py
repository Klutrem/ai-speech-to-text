from datasets import DatasetDict, load_from_disk, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from transformers import DataCollatorForSeq2Seq

# Load feature extractor, tokenizer, and model
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Russian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load dataset
common_voice = DatasetDict()
common_voice["train"] = load_from_disk(dataset_path="/home/klutrem/Desktop/ai-speech-to-text/dataset")

# Preprocess dataset
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Add a unique identifier for each audio file for better logging
common_voice = common_voice["train"].map(lambda batch, idx: {"audio_id": idx}, with_indices=True)

def prepare_dataset(batch):
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    audio_id = batch["audio_id"]

    # Log audio ID for tracking
    print(f"Processing audio ID: {audio_id}")

    # Compute log-Mel input features from input audio array 
    audio_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Log the number of audio features
    print(f"Number of audio features: {audio_features.shape[0]}")

    # Encode target text to label ids 
    sentence = batch["sentence"]
    sentence_ids = tokenizer(sentence).input_ids

    # Log the original sentence and its corresponding token IDs
    print(f"Original sentence: {sentence}")
    print(f"Sentence token IDs: {sentence_ids}")

    return {"input_features": audio_features, "labels": sentence_ids}

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4) 

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Load evaluation metric
wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Adjusted batch size
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
