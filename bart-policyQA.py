import pandas as pd
import json
from pandas import json_normalize
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn
import collections
import evaluate
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.text import BLEUScore, SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
# import nltk
# nltk.download('punkt')


def set_seed(seed = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False  # set to False for final report
    
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)

def preprocess_to_csv(file_path, split = 'train'):

  with open(file_path, "r") as f:
    data = json.load(f)

  new_data = []

  for item in data['data']:
      for paragraph in item['paragraphs']:

          context = paragraph['context']

          for qa in paragraph['qas']:
              entry = {
                  'question': qa['question'],
                  'answer': qa['answers'][0]['text'] if qa['answers'] else "NA",
                  'answer_start' : qa['answers'][0]['answer_start'],
                  'answer_end' : qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text']),
                  'context': context,
                  'id': qa['id'],
              }
              new_data.append(entry)

  df = pd.DataFrame(new_data)
  df.to_csv(f"{split}.csv", index=False)
  data = pd.read_csv(f"{split}.csv")

  dataset = load_dataset("csv", data_files = {"data": f"{split}.csv"})
  
  return dataset

def preprocess_training_examples(examples):
    
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answer"]
    answer_start = examples["answer_start"]
    answer_end = examples["answer_end"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer_start[i]
        end_char = start_char + len(answer)
        sequence_ids = inputs.sequence_ids(i)

        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer_start[i]
        end_char = start_char + len(answer)
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def process_tokenize_val(examples):
    questions = [q.strip() for q in examples["question"]]
    
    print(tokenizer)
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

# class BLEU(torchmetrics.text.BLEUScore):
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

class ROUGE():
    def __init__(self):
        # super().__init__()
        # self.rouge_keys = rouge_keys
        self.rouge = ROUGEScore()
    
    def compute(self, predictions, references):
        rouge_scores = []
        rouge1_fmeasure = 0
        rouge1_precision = 0
        rouge1_recall = 0

        rouge2_fmeasure = 0
        rouge2_precision = 0
        rouge2_recall = 0

        rougeL_fmeasure = 0
        rougeL_precision = 0
        rougeL_recall = 0
        
        for prediction, reference in zip(predictions, references):
            pred = prediction['prediction_text']
            target = reference['answers']['text'][0]
            rouge_result.append(self.rouge(pred, target))

            rouge1_fmeasure += rouge_result['rouge1_fmeasure']
            rouge1_precision += rouge_result['rouge1_precision']
            rouge1_recall += rouge_result['rouge1_recall']

            rouge2_fmeasure += rouge_result['rouge2_fmeasure']
            rouge2_precision += rouge_result['rouge2_precision']
            rouge2_recall += rouge_result['rouge2_recall']

            rougeL_fmeasure += rouge_result['rougeL_fmeasure']
            rougeL_precision += rouge_result['rougeL_precision']
            rougeL_recall += rouge_result['rougeL_recall']
        
        
        rouge1_fmeasure /= len(rouge_result)
        rouge1_precision /= len(rouge_result)
        rouge1_recall /= len(rouge_result)

        rouge2_fmeasure /= len(rouge_result)
        rouge2_precision /= len(rouge_result)
        rouge2_recall /= len(rouge_result)

        rougeL_fmeasure /= len(rouge_result)
        rougeL_precision /= len(rouge_result)
        rougeL_recall /= len(rouge_result)
        
        return {
            'rouge1_fmeasure': rouge1_fmeasure,
            'rouge1_precision': rouge1_precision,
            'rouge1_recall': rouge1_recall,
            'rouge2_fmeasure': rouge2_fmeasure,
            'rouge2_precision': rouge2_precision,
            'rouge2_recall': rouge2_recall,
            'rougeL_fmeasure': rougeL_fmeasure,
            'rougeL_precision': rougeL_precision,
            'rougeL_recall': rougeL_recall,
        }
        return super().forward(self, preds, targets)
    
def compute_metrics(start_logits, end_logits, features, examples, metrics):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        # print(example)

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    # print(predicted_answers)
    theoretical_answers = [{'id': ex['id'], 'answers': {"text": [ex['answer']], 'answer_start': [ex['answer_start']]}} for ex in examples]
    
    print(f"Actual: {theoretical_answers[0]}")
    print(f"Predicted: {predicted_answers[0]}")
    
    for name, metric in metrics.items():
        print(metric.compute(predictions=predicted_answers, references=theoretical_answers))

# train_data = preprocess_to_csv('PolicyQA/data/train.json', split = 'train')['data']
# val_data = preprocess_to_csv('PolicyQA/data/dev.json', split = 'val')['data']
# test_data = preprocess_to_csv('PolicyQA/data/test.json', split = 'test')['data']

set_seed(seed = 1)

train_data = load_dataset("csv", data_files = {"data": "train.csv"})['data']
val_data = load_dataset("csv", data_files = {"data": "val.csv"})['data']
test_data = load_dataset("csv", data_files = {"data": "test.csv"})['data']

print(f"train data length before sampling: {len(train_data)}")
train_data = train_data.shuffle()
train_data = train_data.select(range(3000))

print(f"train data length after sampling: {len(train_data)}")
print(f"val data length: {len(val_data)}")

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForQuestionAnswering.from_pretrained("facebook/bart-large")


data_collator = DefaultDataCollator()


# tokenizer = AutoTokenizer.from_pretrained("bart-policyQA/checkpoint-500")
# model = AutoModelForQuestionAnswering.from_pretrained("bart-policyQA/checkpoint-500")

tokenized_train_data = train_data.map(preprocess_training_examples, batched=True)
tokenized_val_data = val_data.map(process_tokenize_val, batched=True)

training_args = TrainingArguments(
    output_dir="/scratch/m22cs057/bart-l-policyQA",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


n_best = 20
max_answer_length = 100
predicted_answers = []

predictions, _, _ = trainer.predict(tokenized_val_data)

# print(type(predictions))
# print(predictions)
start_logits, end_logits, *optional_outputs = predictions

BLUE_Score = BLEUScore()
metrics = {'exact_match': evaluate.load("squad"), 'bleu': BLEU(BLUE_Score)}#, 'rouge' : ROUGE()}
compute_metrics(start_logits, end_logits, tokenized_val_data, val_data, metrics)
