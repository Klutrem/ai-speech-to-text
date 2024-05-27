import os
from datasets import DatasetDict, load_from_disk, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
from transformers import DataCollatorForSeq2Seq


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Extract input features and labels
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input features to the maximum length
        input_features_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(f) for f in input_features], batch_first=True)

        # Pad labels to the maximum length
        labels_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels], batch_first=True)

        # Create attention masks for the input features
        attention_mask = (input_features_padded != 0).float()

        return {
            "input_features": input_features_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }



def main():
    # Load feature extractor, tokenizer, and model
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Russian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # Load dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk(dataset_path="dataset").select(range(1500))

    # Preprocess dataset
    common_voice["train"] = common_voice["train"].remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice["train"] = common_voice["train"].cast_column("audio", Audio(sampling_rate=16000))

    # Add a unique identifier for each audio file for better logging
    common_voice["train"] = common_voice["train"].map(lambda batch, idx: {"audio_id": idx}, with_indices=True)

    def prepare_dataset(batch):
        max_input_length = 128  # Adjust as necessary
        
        inputs = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=max_input_length, truncation=True)
        batch["input_features"] = inputs.input_features.squeeze().tolist()
        
        max_target_length = 128  # Adjust as necessary
        labels = processor.tokenizer(batch["sentence"], return_tensors="pt", padding="max_length", max_length=max_target_length, truncation=True).input_ids
        batch["labels"] = labels.squeeze().tolist()
        
        return batch
    # Use num_proc=4 to utilize multiprocessing, adjust based on your system's capacity
    common_voice["train"] = common_voice["train"].map(
        prepare_dataset, 
        remove_columns=common_voice["train"].column_names, 
        num_proc=6  # Increase to leverage multiple CPUs
    )

    data_collator = CustomDataCollator(tokenizer, model=model, padding=True)

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
        per_device_train_batch_size=4,  # Adjusted batch size for your system's capacity
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=4,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,  # Use mixed precision training
        dataloader_num_workers=1,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    trainer.train()

    # Save the model
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

if __name__ == "__main__":
    main()
