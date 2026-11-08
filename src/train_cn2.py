# -*- coding: utf-8 -*-
"""
v2 中文标签版 RoBERTa 全参数微调（4GB显存适配：batch4 + 梯度累积4）。
用法：
    python train_cn2.py --task binary_cn2
    python train_cn2.py --task type_cn2
"""
import argparse
import glob
import math
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
SPLITS_DIR = os.path.join(BASE, "data", "splits_cn2")
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 128

TASK_LABELS = {
    "binary_cn2": ["正常", "可疑"],
    "type_cn2": ["营销骚扰", "冒充公检法类", "贷款类", "冒充客服类", "冒充领导熟人类"],
}


def load_parts(task):
    out = {}
    for part in ("train", "val", "test"):
        out[part] = pd.read_csv(
            os.path.join(SPLITS_DIR, f"{task}_{part}.csv"), encoding="utf-8-sig")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASK_LABELS.keys(), required=True)
    parser.add_argument("--resume", action="store_true",
                        help="从 output_dir 下最新的 checkpoint 断点续训")
    args = parser.parse_args()
    task = args.task
    labels = TASK_LABELS[task]
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    print("标签映射:", label2id)

    frames = load_parts(task)
    print("数据量:", {k: len(v) for k, v in frames.items()})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["content"], truncation=True, max_length=MAX_LEN)

    ds = {}
    for part in ("train", "val", "test"):
        d = frames[part].copy()
        d["label"] = d["label"].map(label2id)
        ds[part] = Dataset.from_pandas(d).map(
            tokenize, batched=True, remove_columns=["content"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), label2id=label2id, id2label=id2label)

    def compute_metrics(eval_pred):
        logits, true = eval_pred
        pred = np.argmax(logits, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(
            true, pred, average="macro", zero_division=0)
        acc = (pred == true).mean()
        return {"accuracy": acc, "macro_f1": f1,
                "macro_precision": p, "macro_recall": r}

    output_dir = os.path.join(BASE, "models", f"roberta_{task}")
    batch_size, grad_accum, epochs = 4, 4, 4
    steps_per_epoch = math.ceil(len(frames["train"]) / (batch_size * grad_accum))
    warmup_steps = int(steps_per_epoch * epochs * 0.1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    resume_ckpt = None
    if args.resume:
        ckpts = sorted(
            [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
             if os.path.isdir(d)],
            key=lambda p: int(p.rsplit("checkpoint-", 1)[1]))
        if not ckpts:
            raise FileNotFoundError(f"没有找到可续训的 checkpoint: {output_dir}")
        resume_ckpt = ckpts[-1]
        print("从 checkpoint 续训:", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    result = trainer.evaluate(ds["test"])
    print("测试集指标:", result)
    pred = np.argmax(trainer.predict(ds["test"]).predictions, axis=1)
    true = ds["test"]["label"]
    cm = confusion_matrix(true, pred, labels=list(range(len(labels))))
    print("混淆矩阵(行真实,列预测)，列顺序:", labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print(classification_report(true, pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("模型已保存到:", output_dir)


if __name__ == "__main__":
    main()
