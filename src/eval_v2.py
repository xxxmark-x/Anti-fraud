# -*- coding: utf-8 -*-
"""
v2 评估：新模型在测试集、独立holdout、自定义短信上的效果。
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
MAX_LEN = 128
ORDER = ["正常", "营销骚扰", "冒充公检法类", "贷款类", "冒充客服类", "冒充领导熟人类"]


class Detector:
    def __init__(self, binary_dir, type_dir, threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(binary_dir)
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(
            binary_dir).to(self.device).eval()
        self.type_model = AutoModelForSequenceClassification.from_pretrained(
            type_dir).to(self.device).eval()
        self.binary_names = {int(k): v for k, v in
                             self.binary_model.config.id2label.items()}
        self.type_names = {int(k): v for k, v in
                           self.type_model.config.id2label.items()}

    def _logits(self, model, texts):
        inp = self.tokenizer(texts, truncation=True, max_length=MAX_LEN,
                             padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return model(**inp).logits

    def detect(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        b_logits = self._logits(self.binary_model, texts)
        b_probs = torch.softmax(b_logits, dim=-1)
        suspect_id = next(i for i, n in self.binary_names.items() if n == "可疑")
        sp = b_probs[:, suspect_id].cpu().numpy()
        flag = sp >= self.threshold
        idx = [i for i, f in enumerate(flag) if f]
        tpred = {}
        if idx:
            tl = self._logits(self.type_model, [texts[i] for i in idx])
            tpred = dict(zip(idx, torch.argmax(tl, -1).cpu().numpy()))
        out = []
        for i in range(len(texts)):
            if flag[i]:
                out.append({"一级": "可疑", "概率": round(float(sp[i]), 4),
                            "类型": self.type_names[int(tpred[i])]})
            else:
                out.append({"一级": "正常", "概率": round(float(sp[i]), 4), "类型": "正常"})
        return out


def report(name, detector, df, content_col="content"):
    res = detector.detect(df[content_col].tolist())
    true_b = df["binary_label"].to_numpy()
    pred_b = np.array([r["一级"] for r in res])
    true_t = df["type_label"].to_numpy()
    pred_t = np.array([r["类型"] for r in res])
    print("=" * 70)
    print(f"{name}（{len(df)}条） 一级准确率: {(pred_b == true_b).mean():.4f}")
    print(pd.DataFrame(confusion_matrix(true_b, pred_b, labels=["正常", "可疑"]),
                       index=["真实正常", "真实可疑"], columns=["预测正常", "预测可疑"]))
    print("端到端六分类报告：")
    print(classification_report(true_t, pred_t, labels=ORDER,
                                target_names=ORDER, zero_division=0, digits=4))


def add_binary(df):
    df = df.copy()
    df["binary_label"] = np.where(df["type_label"] == "正常", "正常", "可疑")
    return df


def main():
    # 数据
    test = add_binary(pd.read_csv(os.path.join(
        BASE, "data", "splits_cn2", "binary_cn2_test.csv"), encoding="utf-8-sig").merge(
        pd.read_csv(os.path.join(BASE, "data", "processed", "cleaned_all_cn2.csv"),
                    encoding="utf-8-sig")[["content", "type_label"]], on="content"))
    holdout = add_binary(pd.read_csv(os.path.join(
        BASE, "data", "holdout", "holdout_cn.csv"), encoding="utf-8-sig"))

    # 新模型
    t0 = time.time()
    new = Detector(os.path.join(BASE, "models", "roberta_binary_cn2"),
                   os.path.join(BASE, "models", "roberta_type_cn2"))
    print(f"模型加载完成，耗时{time.time()-t0:.1f}s")
    report("模型-同分布测试集", new, test)
    report("模型-独立holdout", new, holdout)

    # 之前6条短信
    print("=" * 70)
    print("之前6条自定义短信")
    tests = [
        "今晚下班一起吃饭吗？我在公司楼下等你。",
        "【银行】您的账户涉嫌洗钱，请立即配合公安机关调查，下载视频会议APP接受资金审查。",
        "尊敬的客户，您的信用额度已提升至50万，点击链接立即申请低息贷款，退订回T。",
        "您的快递丢失，我是客服，加微信给您双倍赔款。",
        "【拉卡拉】您的POS机即将终止服务，继续使用请回复1，费率0.38%，拒收请回复R。",
        "妈妈，我手机坏了，这是我新号，尽快给我转2000块交学费。",
    ]
    for text, r in zip(tests, new.detect(tests)):
        print(f"  {text[:40]}... -> {r}")


if __name__ == "__main__":
    main()
