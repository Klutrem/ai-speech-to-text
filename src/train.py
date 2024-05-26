import os
import time
from datasets import DatasetDict, load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, 
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate

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

# Subset the dataset for a quick test
small_train_dataset = common_voice["train"].select(range(100))  # Select the first 100 samples for a quick test

# Preprocess the subset dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Run the map function on the subset
small_train_dataset = small_train_dataset.map(prepare_dataset, remove_columns=small_train_dataset.column_names, num_proc=1)

# Custom data collator
class CustomDataCollator:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Padding the input features
        input_features_padded = self.feature_extractor.pad(
            {"input_features": input_features}, padding=True, return_tensors="pt"
        )["input_features"]
        
        # Padding the labels
        labels_padded = self.tokenizer.pad(
            {"input_ids": labels}, padding=True, return_tensors="pt"
        )["input_ids"]
        
        return {"input_features": input_features_padded, "labels": labels_padded}

data_collator = CustomDataCollator(feature_extractor, tokenizer)

# Load evaluation metric
wer = evaluate.load("wer")

# Define training arguments for the quick test
quick_test_args = Seq2SeqTrainingArguments(
    output_dir="./quick_test_results",
    per_device_train_batch_size=3,  # Smaller batch size to reduce memory load
    evaluation_strategy="no",
    learning_rate=2e-5,
    num_train_epochs=1,  # Just one epoch for the test
    save_total_limit=1,
    predict_with_generate=True,
    fp16=True,
    dataloader_num_workers=4,  # Number of subprocesses to use for data loading
)

# Initialize trainer for the quick test
quick_test_trainer = Seq2SeqTrainer(
    model=model,
    args=quick_test_args,
    train_dataset=small_train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Measure training time
start_time = time.time()
quick_test_trainer.train()
end_time = time.time()

# Calculate elapsed time and estimate for full training
elapsed_time = end_time - start_time
print(f"Time for one epoch on 100 samples: {elapsed_time:.2f} seconds")

# Estimate for full dataset
total_samples = len(common_voice["train"])
num_epochs = 3
estimated_time = (elapsed_time / 100) * total_samples * num_epochs
print(f"Estimated total training time for full dataset: {estimated_time / 3600:.2f} hours")

# Save the trained model, tokenizer, and feature extractor
output_dir = "./trained_model"

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model.save_pretrained(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)

# Save the feature extractor
feature_extractor.save_pretrained(output_dir)

print(f"Model, tokenizer, and feature extractor saved to {output_dir}")
