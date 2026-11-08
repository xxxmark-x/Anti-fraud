# -*- coding: utf-8 -*-
"""
交互式短信判断工具：自己输入短信，模型当场判断。
运行：python predict.py
把短信内容粘贴进来回车看结果；输入 q 回车退出。
"""
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
MAX_LEN = 128


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bin_dir = os.path.join(BASE, "models", "roberta_binary_cn2")
    type_dir = os.path.join(BASE, "models", "roberta_type_cn2")

    tokenizer = AutoTokenizer.from_pretrained(bin_dir)
    bin_model = AutoModelForSequenceClassification.from_pretrained(
        bin_dir).to(device).eval()
    type_model = AutoModelForSequenceClassification.from_pretrained(
        type_dir).to(device).eval()
    bin_names = {int(k): v for k, v in bin_model.config.id2label.items()}
    type_names = {int(k): v for k, v in type_model.config.id2label.items()}
    suspect_id = next(i for i, n in bin_names.items() if n == "可疑")

    def judge(text):
        inp = tokenizer(text, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            b_prob = torch.softmax(bin_model(**inp).logits, dim=-1)[0]
        p_suspect = float(b_prob[suspect_id])
        if p_suspect < 0.5:
            return p_suspect, "正常", None
        with torch.no_grad():
            t_id = int(torch.argmax(type_model(**inp).logits, dim=-1)[0])
        return p_suspect, "可疑", type_names[t_id]

    print("=" * 60)
    print("短信判断工具已就绪")
    print("把短信内容粘贴进来，回车就能看结果。")
    print("输入 q 再回车，退出程序。")
    print("=" * 60)

    while True:
        try:
            text = input("\n请输入短信内容：\n").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit") or text == "退出":
            print("已退出。")
            break

        p, flag, typ = judge(text)
        print("-" * 60)
        if flag == "正常":
            print(f"判断：正常（可疑程度 {p * 100:.1f}%）")
            print("这条看起来没问题。")
        else:
            print(f"判断：可疑（可疑程度 {p * 100:.1f}%）")
            print(f"类型：{typ}")
            print("提醒：这条要当心，别点链接、别转账、别乱加联系方式。")
        print("-" * 60)


if __name__ == "__main__":
    main()
