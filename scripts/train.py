#!/usr/bin/env python
# coding: utf-8

import json
import copy
import gc
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import argparse
import torch
from torch import nn
import numpy as np
import pandas as pd
from spacy.lang.en import English
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, concatenate_datasets
import wandb
sys.path.append('../tools/')
from ul_metrics import *
from ul_tokenizer import *

## Config & Parameters

# 引数解析用オブジェクトの作成
parser = argparse.ArgumentParser(description="Script that accepts a custom string")
# 引数の追加（ここではchoicesを指定しない）
parser.add_argument("custom_string", help="Custom string provided by the user")
# 引数の解析
args = parser.parse_args()
# 取得した文字列の使用
custom_string = args.custom_string

# 現在の日時を取得し、文字列に変換する
current_datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
print(current_datetime_str)

TRAINING_MODEL_PATH = "microsoft/deberta-v3-base" # "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 3072
EVAL_MAX_LENGTH = 3072
CONF_THRESH = 0.9
LR = 2.5e-5
LR_SCHEDULER_TYPE = "linear"
NUM_EPOCHS = 3
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 16 // BATCH_SIZE
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
AMP = True
FREEZE_EMBEDDING = False
FREEZE_LAYERS = 6
N_SPLITS = 4
NEGATIVE_RATIO = 0.3  # down sample ratio of negative samples in the training set
OUTPUT_DIR = f"../models/{custom_string}_{current_datetime_str}"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=AMP,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=50,
    eval_delay=100,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    metric_for_best_model="f5",
    greater_is_better=True,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
)


## Dataset Preparation

s3_path = "/kaggle/s3storage/01_public/2024_pii_data_detection/02_unzip_file"

with Path(os.path.join(s3_path,"pii-detection-removal-from-educational-data/train.json")).open("r") as f:
    original_data = json.load(f)

with Path(os.path.join(s3_path,"pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1-no-i-username.json")).open("r") as f:
    extra_data = json.load(f)

all_labels = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 
    'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]


## Instantiate the dataset

tokenizer = DebertaV2TokenizerFast.from_pretrained(TRAINING_MODEL_PATH)
train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=TRAINING_MAX_LENGTH)
eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=EVAL_MAX_LENGTH)

ds = DatasetDict()

for key, data in zip(["original", "extra"], [original_data, extra_data]):
    ds[key] = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })


## Metrics

class MetricsComputer:
    nlp = English()

    def __init__(self, eval_ds: Dataset, label2id: dict, conf_thresh: float = 0.9) -> None:
        self.ds = eval_ds.remove_columns("labels").rename_columns({"provided_labels": "labels"})
        self.gt_df = self.create_gt_df(self.ds)
        self.label2id = label2id
        self.confth = conf_thresh
        self._search_gt()

    def __call__(self, eval_preds: EvalPrediction) -> dict:
        pred_df = self.create_pred_df(eval_preds.predictions)
        return self.compute_metrics_from_df(self.gt_df, pred_df)

    def _search_gt(self) -> None:
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        self.emails = []
        self.phone_nums = []

        for _data in self.ds:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    self.emails.append(
                        {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                    )
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in self.nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    self.phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                    )

    @staticmethod
    def create_gt_df(ds: Dataset):
        gt = []
        for row in ds:
            for token_idx, (token, label) in enumerate(zip(row["tokens"], row["labels"])):
                if label == "O":
                    continue
                gt.append(
                    {"document": row["document"], "token": token_idx, "label": label, "token_str": token}
                )
        gt_df = pd.DataFrame(gt)
        gt_df["row_id"] = gt_df.index

        return gt_df

    def create_pred_df(self, logits: np.ndarray) -> pd.DataFrame:
        """
        Note:
            Thresholing is doen on logits instead of softmax, which could find better models on LB.
        """
        prediction = logits
        o_index = self.label2id["O"]
        preds = prediction.argmax(-1)
        preds_without_o = prediction.copy()
        preds_without_o[:,:,o_index] = 0
        preds_without_o = preds_without_o.argmax(-1)
        o_preds = prediction[:,:,o_index]
        preds_final = np.where(o_preds < self.confth, preds_without_o , preds)

        pairs = set()
        processed = []

        # Iterate over document
        for p_doc, token_map, offsets, tokens, doc in zip(
            preds_final, self.ds["token_map"], self.ds["offset_mapping"], self.ds["tokens"], self.ds["document"]
        ):
            # Iterate over sequence
            for p_token, (start_idx, end_idx) in zip(p_doc, offsets):
                label_pred = id2label[p_token]

                if start_idx + end_idx == 0:
                    # [CLS] token i.e. BOS
                    continue

                if token_map[start_idx] == -1:
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]
                pair = (doc, token_id)

                # ignore "O", preds, phone number and  email
                if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                    continue

                if pair in pairs:
                    continue

                processed.append(
                    {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
                )
                pairs.add(pair)

        pred_df = pd.DataFrame(processed + self.emails + self.phone_nums)
        pred_df["row_id"] = list(range(len(pred_df)))

        return pred_df

    def compute_metrics_from_df(self, gt_df, pred_df):
        """
        Compute the LB metric (lb) and other auxiliary metrics
        """

        references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
        predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

        score_per_type = defaultdict(PRFScore)
        references = set(references)

        for ex in predictions:
            pred_type = ex[-1] # (document, token, label)
            if pred_type != 'O':
                pred_type = pred_type[2:] # avoid B- and I- prefix

            if pred_type not in score_per_type:
                score_per_type[pred_type] = PRFScore()

            if ex in references:
                score_per_type[pred_type].tp += 1
                references.remove(ex)
            else:
                score_per_type[pred_type].fp += 1

        for doc, tok, ref_type in references:
            if ref_type != 'O':
                ref_type = ref_type[2:] # avoid B- and I- prefix

            if ref_type not in score_per_type:
                score_per_type[ref_type] = PRFScore()
            score_per_type[ref_type].fn += 1

        totals = PRFScore()

        for prf in score_per_type.values():
            totals += prf

        return {
            "precision": totals.precision,
            "recall": totals.recall,
            "f5": totals.f5,
            **{
                f"{v_k}-{k}": v_v
                for k in set([l[2:] for l in self.label2id.keys() if l!= 'O'])
                for v_k, v_v in score_per_type[k].to_dict().items()
            },
        }


## Model

class ModelInit:
    def __init__(
        self,
        checkpoint: str,
        id2label: dict,
        label2id: dict,
        freeze_embedding: bool,
        freeze_layers: int,
    ) -> None:
        self.model = DebertaV2ForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False if freeze_embedding else True
        for layer in self.model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        self.weight = copy.deepcopy(self.model.state_dict())

    def __call__(self) -> DebertaV2ForTokenClassification:
        self.model.load_state_dict(self.weight)
        return self.model

model_init = ModelInit(
    TRAINING_MODEL_PATH,
    id2label=id2label,
    label2id=label2id,
    freeze_embedding=FREEZE_EMBEDDING,
    freeze_layers=FREEZE_LAYERS,
)


## Split 
'''
Split the original dataset into 4 folds according to `document % 4` 
Only uses the first 30% of negative samples in the training set but they are NOT excluded from the eval set to make sure cross-evalidation is done on the entire training dataset.
split according to document id
'''

folds = [
    (
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS != s]),
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS == s])
    )
    for s in range(N_SPLITS)
]

negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]


## Train
# Performs cross-validation and save the best checkpoint's metrics as json.

for fold_idx, (train_idx, eval_idx) in enumerate(folds):
    args.run_name = f"fold-{fold_idx}"
    args.output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    original_ds = ds["original"].select([i for i in train_idx if i not in exclude_indices])
    train_ds = concatenate_datasets([original_ds, ds["extra"]])
    train_ds = train_ds.map(train_encoder, num_proc=os.cpu_count())
    eval_ds = ds["original"].select(eval_idx)
    eval_ds = eval_ds.map(eval_encoder, num_proc=os.cpu_count())
    trainer = Trainer(
        args=args,
        model_init=model_init,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=MetricsComputer(eval_ds=eval_ds, label2id=label2id),
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
    )
    # break # delete this line to reproduce the result.
    trainer.train()
    eval_res = trainer.evaluate(eval_dataset=eval_ds)
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(eval_res, f)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()