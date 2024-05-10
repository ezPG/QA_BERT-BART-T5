import pandas as pd
train_privacy_df = pd.read_csv("/scratch/m22cs057/PrivacyQA/policy_train_data.csv", sep="\t")
test_privacy_df = pd.read_csv("/scratch/m22cs057/PrivacyQA/policy_test_data.csv", sep="\t")

train_privacy_df = train_privacy_df[['Query', 'Segment']]
test_privacy_df = test_privacy_df[['Query', 'Segment']]

train_privacy_df = train_privacy_df.drop_duplicates(keep=False)
test_privacy_df = test_privacy_df.drop_duplicates(keep=False)


import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
  )
import torch

import collections
import evaluate
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.text import BLEUScore, SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model = AutoModelForSeq2SeqLM.from_pretrained('PrivacyQA/bart/checkpoint-1000')


train_df, val_df = train_test_split(train_privacy_df, test_size=0.10, shuffle=True)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_privacy_df)

train_dataset = train_dataset.shuffle()
train_dataset = train_dataset.select(range(10000))
print(train_dataset)


val_dataset = val_dataset.select(range(1000))
print(val_dataset)



def compute_metrics(start_logits, end_logits, features, examples, metrics):

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["Query"]
        context = example["Segment"]
        answers = []
    
    
    for i in range(len(start_logits)):
        predict_answer_tokens = examples['Segment'][i, start_idx : end_idx + 1]
        predicted_answer = tokenizer.decode(predict_answer_tokens)
        predicted_answers.append({"Query": example['Query'][i], "prediction_text": predicted_answer})
        
        
    theoretical_answers = [
        {'Query': ex['Query'], 'Segment': ex['Segment']} for ex in examples
    ]
            
    print(f"Actual: {theoretical_answers[0]}")
    print(f"Predicted: {predicted_answers[0]}")
    
    for name, metric in metrics.items():
        print(metric.compute(predictions=predicted_answers, references=theoretical_answers))

class BLEU():
    def __init__(self, bleu, n_gram=4, smooth=False, weights=None):
        # super().__init__()
        self.bleu_score = bleu
        self.scare_bleu = SacreBLEUScore()
    
    def compute(self, predictions, references):
        bleu_scores = []  # Store individual scores
        scare_bleu_scores = []
        for prediction, reference in zip(predictions, references):
            pred = prediction['prediction_text']
            target = reference['answers']['text']
            bleu_scores.append(self.bleu_score([pred], [target]))
            scare_bleu_scores.append(self.scare_bleu([pred], [target]))

        average_bleu = sum(bleu_scores) / len(bleu_scores)
        average_sacre = sum(scare_bleu_scores) / len(scare_bleu_scores)
        
        return {'bleu': average_bleu, 'sacre_bleu_score': average_sacre}
        # return super().forward(self, preds, targets)

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=64, pad_to_max_length=True):
        self.tokenizer = tokenizer
        self.data = df
        self.max_len = max_len
        self.pad_to_max_length = pad_to_max_length

    def __len__(self):
        return len(self.data)

    def tokenize_data(self, example):
        input_, target_ = example['Query'], example['Segment']

        # Tokenize inputs
        input_set = self.tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
                                          max_length=self.max_len, return_attention_mask=True)

        # Tokenize targets
        target_set = self.tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
                                           max_length=self.max_len, return_attention_mask=True)

        inputs = {
            "input_ids": input_set['input_ids'],
            "attention_mask": input_set['attention_mask'],
            "labels": target_set['input_ids']
        }

        return inputs

    def __getitem__(self, index):
        inputs = self.tokenize_data(self.data[index])

        return inputs

batch_size=16

#collator to create batches. It preprocess data with the given tokenizer
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir="/scratch/m22cs057/PrivacyQA/bart",
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.3,
    save_steps=1000,
    num_train_epochs=5,
    dataloader_num_workers= 2,
    predict_with_generate=True,
    fp16=True

)

trainer = Seq2SeqTrainer(model=model,
                args=args,
                train_dataset= Seq2SeqDataset(train_dataset, tokenizer),
                eval_dataset=Seq2SeqDataset(val_dataset, tokenizer),
                tokenizer=tokenizer,
                data_collator=collator)

# trainer.train()

n_best = 20
max_answer_length = 100
predicted_answers = []

predictions, _, _ = trainer.predict(Seq2SeqDataset(val_dataset, tokenizer))

start_logits, end_logits = predictions

BLUE_Score = BLEUScore()
metrics = {'exact_match': evaluate.load("squad"), 'bleu': BLEU(BLUE_Score)}#, 'rouge' : ROUGE()}
compute_metrics(start_logits, end_logits, tokenized_val_data, val_data, metrics)

# from datasets import load_metric

# def process_tokenize_val(example):
#     source = example['Query']
#     target = example['Segment']

#     inputs = tokenizer(source, return_tensors="pt", max_length=64, truncation=True, padding=True)
#     labels = tokenizer(target, return_tensors="pt", max_length=64, truncation=True, padding=True)

#     return {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "labels": labels["input_ids"]
#     }



# tokenized_val_data = test_privacy_df.map(process_tokenize_val)

# predictions = []
# for example in tokenized_val_data:
#     # input_ids = torch.tensor(example["input_ids"]).unsqueeze(0) # Batch size of 1
#     # attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0)  # Batch size of 1
#     input_ids = example["input_ids"].unsqueeze(0) # Batch size of 1
#     attention_mask = example["attention_mask"].unsqueeze(0)  # Batch size of 1

#     # Generate output tokens using the BART model
#     output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)

#     # Decode the predicted tokens to text
#     predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     predictions.append(predicted_text)


# metric = load_metric("bleu")
# references = val_df["Segment"].tolist()

# bleu_score = metric.compute(predictions=predictions, references=references)



# start_logits, end_logits, *optional_outputs = predictions


# metrics = {'exact_match': evaluate.load("squad"), 'bleu': BLEU(BLUE_Score)}#, 'rouge' : ROUGE()}
# compute_metrics(start_logits, end_logits, tokenized_val_data, val_dataset, metrics)




