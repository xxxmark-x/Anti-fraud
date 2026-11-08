# -*- coding: utf-8 -*-
"""
本地网页版短信判断工具（Gradio）。
运行：python web_app.py，会自动打开浏览器。
模型只在本机运行，短信内容不会传到外面。
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
MAX_LEN = 128

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
    text = (text or "").strip()
    if not text:
        return ('<div style="padding:16px;border-radius:10px;'
                'background:#f5f5f5;border:1px solid #ccc;color:#666;">'
                '请先把短信内容粘贴进来，再点“开始判断”。</div>')

    inp = tokenizer(text, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        b_prob = torch.softmax(bin_model(**inp).logits, dim=-1)[0]
    p = float(b_prob[suspect_id])

    if p < 0.5:
        return f"""
        <div style="padding:18px;border-radius:10px;background:#e8f5e9;
                    border:1px solid #4caf50;">
          <div style="font-size:22px;font-weight:bold;color:#2e7d32;">判断：正常</div>
          <div style="margin-top:10px;font-size:15px;color:#333;line-height:1.8;">
            可疑程度：<b>{p * 100:.1f}%</b><br>
            这条看起来没问题。
          </div>
        </div>"""

    with torch.no_grad():
        t_id = int(torch.argmax(type_model(**inp).logits, dim=-1)[0])
    typ = type_names[t_id]
    return f"""
    <div style="padding:18px;border-radius:10px;background:#ffebee;
                border:1px solid #f44336;">
      <div style="font-size:22px;font-weight:bold;color:#c62828;">
        判断：可疑 · {typ}</div>
      <div style="margin-top:10px;font-size:15px;color:#333;line-height:1.8;">
        可疑程度：<b>{p * 100:.1f}%</b><br>
        提醒：这条要当心，别点链接、别转账、别乱加联系方式。
      </div>
    </div>"""


examples = [
    "今晚下班一起吃饭吗？我在公司楼下等你。",
    "【银行】您的账户涉嫌洗钱，请立即配合公安机关调查，接受资金审查。",
    "【拉卡拉】您的POS机即将终止服务，费率0.38%，拒收请回复R。",
]

with gr.Blocks(title="短信判断工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 短信判断工具")
    gr.Markdown("把短信粘贴到下面，点“开始判断”，马上告诉你正常还是可疑。")
    inp = gr.Textbox(lines=5, label="短信内容",
                     placeholder="在这里粘贴短信……")
    btn = gr.Button("开始判断", variant="primary")
    out = gr.HTML(label="判断结果")
    btn.click(judge, inputs=inp, outputs=out)
    inp.submit(judge, inputs=inp, outputs=out)
    gr.Examples(examples=examples, inputs=inp, label="点一下可以试试这些例子")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860,
                inbrowser=True, share=False)
