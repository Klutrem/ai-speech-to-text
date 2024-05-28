from datasets import DatasetDict, load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
import torch
from transformers import DataCollatorForSeq2Seq

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):

        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]

        input_features_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(f) for f in input_features], batch_first=True)

        labels_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels], batch_first=True)

        attention_mask = (input_features_padded != 0).float()

        return {
            "input_features": input_features_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }

def prepare_dataset(batch, feature_extractor, tokenizer):

    audio = batch["audio"]

    audio_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    sentence = batch["sentence"]
    sentence_ids = tokenizer(sentence).input_ids

    return {"input_features": audio_features, "labels": sentence_ids}

def main():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk(dataset_path="dataset").select(range(1500))

    fraction = 0.1
    eval_size = int(len(common_voice["train"]) * fraction)
    train_dataset = common_voice["train"].select(range(eval_size, len(common_voice["train"])))
    eval_dataset = common_voice["train"].select(range(eval_size))

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = train_dataset.map(lambda batch: prepare_dataset(batch, processor.feature_extractor, processor.tokenizer), remove_columns=train_dataset.column_names, num_proc=1)
    eval_dataset = eval_dataset.map(lambda batch: prepare_dataset(batch, processor.feature_extractor, processor.tokenizer), remove_columns=eval_dataset.column_names, num_proc=1)

    data_collator = CustomDataCollator(tokenizer=processor.tokenizer, model=model, padding=True)

    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_score = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_score}


    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        dataloader_num_workers=3,
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    trainer.train()


    model.save_pretrained("./trained_model2")
    processor.save_pretrained("./trained_model2")

if __name__ == "__main__":
    main()
