# -*- coding: utf-8 -*-
"""
v2 数据构建：改进营销规则 + 加入真实风格增强样本。
流程：加载原始5CSV -> 清洗去重 -> normalize -> 再去重 ->
      新规则分离营销（含服务通知白名单）-> 加入增强训练样本 ->
      中文标签 -> 分层8:1:1划分；独立测试集单独留出不参与训练。
输出：
  data/processed/cleaned_all_cn2.csv
  data/processed/label_mapping_cn2.json
  data/splits_cn2/{binary,type}_cn2_{train,val,test}.csv
  data/holdout/holdout_cn.csv
"""
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_data import get_train_samples, get_holdout_samples

BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
RAW_DIR = os.path.join(BASE, "Telecom_Fraud_Texts_5-main")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
SPLITS_DIR = os.path.join(BASE, "data", "splits_cn2")
HOLDOUT_DIR = os.path.join(BASE, "data", "holdout")
SEED = 42
MIN_LEN = 6

FILE_MAP = {
    "label00-last.csv": ("utf-8-sig", "正常"),
    "label01-last.csv": ("utf-8-sig", "冒充公检法类"),
    "label02-last.csv": ("utf-8-sig", "贷款类"),
    "label03-last.csv": ("utf-8-sig", "冒充客服类"),
    "label04-last.csv": ("gbk", "冒充领导熟人类"),
}

# 敏感信息替换
URL_PAT = r"https?://[^\s，。；、）)】\]]+|www\.[^\s，。；、）)】\]]+"
TEL_PAT = r"\d{3,4}-?\d{7,8}|(?<!\d)1[3-9]\d{9}(?!\d)"
MONEY_PAT = r"\d+(?:\.\d+)?\s*(?:万元|元|块钱|块|万)"
LONG_NUM_PAT = r"\d{5,}"

# 营销规则（已在 verify_rules.py 验证）
UNSUB_PAT = r"退订|拒收|回TD|回T|回R|回复T|回复R|取消订阅"
SELL_PAT = (r"POS机|电签|刷卡|费率|招商|加盟|代理|诚招|新盘|开盘|看房|楼盘|商铺|"
            r"年卡|体验课|优惠券|秒杀|补贴|理财|保险|装修|健身|美容|汽配|车展|"
            r"彩铃.{0,20}(?:订购|下载|资费)|铃声.{0,20}(?:订购|下载|资费)|"
            r"代理加盟|诚招代理|推销|电销|回拨.{0,10}办理")
STRONG_PROMO = (r"免费领取|抽奖|奖品|秒杀|礼包|大促|回馈|赠送|折扣|优惠|抢购|福利|"
                r"返利|免单|砍价|特惠|促销|中奖|红包|开抢|夺宝|领取|钜惠|好礼")
SERVICE_PAT = r"流量|账单|余额|已用|剩余|曲线图|话费|套餐内|资费|缴费|使用情况|查询"


def normalize(s):
    s = s.str.replace(URL_PAT, "[URL]", regex=True)
    s = s.str.replace(TEL_PAT, "[PHONE]", regex=True)
    s = s.str.replace(MONEY_PAT, "[MONEY]", regex=True)
    s = s.str.replace(LONG_NUM_PAT, "[NUM]", regex=True)
    return s


def load_and_clean():
    frames = []
    for fname, (enc, name) in FILE_MAP.items():
        d = pd.read_csv(os.path.join(RAW_DIR, fname), encoding=enc)
        d["type_label"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].str.len() >= MIN_LEN]
    df = df.drop_duplicates(subset=["content"], keep="first")
    df["content"] = normalize(df["content"])
    df = df.drop_duplicates(subset=["content"], keep="first")  # 规范化后再去重
    return df.reset_index(drop=True)


def split_marketing(df):
    is_normal = df["type_label"] == "正常"
    normal = df[is_normal].copy()
    c = normal["content"]
    has_strong = c.str.contains(STRONG_PROMO, regex=True)
    is_service = c.str.contains(SERVICE_PAT, regex=True)
    service_whitelist = is_service & ~has_strong
    unsub = c.str.contains(UNSUB_PAT, regex=True)
    sell = c.str.contains(SELL_PAT, regex=True)
    mask = (has_strong | unsub | sell) & ~service_whitelist

    marketing = set(normal[mask]["content"])
    df.loc[df["content"].isin(marketing), "type_label"] = "营销骚扰"
    return df, len(marketing), int(service_whitelist.sum())


def add_augment_train(df):
    rows = []
    for text, label in get_train_samples():
        s = normalize(pd.Series([text]).str.strip())
        rows.append({"content": s.iloc[0], "type_label": label})
    aug = pd.DataFrame(rows)
    aug = aug[~aug["content"].isin(df["content"])]  # 去重
    return pd.concat([df, aug], ignore_index=True), len(aug)


def stratified_split(df, col):
    tv, test = train_test_split(df, test_size=0.1, stratify=df[col], random_state=SEED)
    train, val = train_test_split(tv, test_size=1 / 9, stratify=tv[col], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_holdout():
    rows = []
    for text, label in get_holdout_samples():
        s = normalize(pd.Series([text]).str.strip())
        rows.append({"content": s.iloc[0], "type_label": label})
    return pd.DataFrame(rows)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    df = load_and_clean()
    df, n_mkt, n_wl = split_marketing(df)
    df, n_added = add_augment_train(df)
    df["binary_label"] = df["type_label"].map(lambda x: "正常" if x == "正常" else "可疑")

    print("最终各类别数量：")
    print(df["type_label"].value_counts().to_string())
    print(f"\n营销分离出 {n_mkt} 条，服务白名单保留 {n_wl} 条，增强样本实际加入 {n_added} 条")

    df[["content", "binary_label", "type_label"]].to_csv(
        os.path.join(PROCESSED_DIR, "cleaned_all_cn2.csv"),
        index=False, encoding="utf-8-sig")

    mapping = {
        "一级": {"正常": "正常短信", "可疑": "营销或诈骗短信"},
        "二级": {
            "正常": "正常短信", "营销骚扰": "广告推销、营销骚扰短信",
            "冒充公检法类": "冒充公检法及政府机关", "贷款类": "贷款、代办信用卡",
            "冒充客服类": "冒充客服服务", "冒充领导熟人类": "冒充领导、熟人",
        },
    }
    with open(os.path.join(PROCESSED_DIR, "label_mapping_cn2.json"), "w",
              encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 一级
    b = stratified_split(df, "binary_label")
    for part, d in zip(("train", "val", "test"), b):
        d[["content", "binary_label"]].rename(columns={"binary_label": "label"}).to_csv(
            os.path.join(SPLITS_DIR, f"binary_cn2_{part}.csv"),
            index=False, encoding="utf-8-sig")
    print(f"一级 训练/验证/测试: {len(b[0])}/{len(b[1])}/{len(b[2])}")

    # 二级（仅可疑）
    suspect = df[df["binary_label"] == "可疑"]
    t = stratified_split(suspect, "type_label")
    for part, d in zip(("train", "val", "test"), t):
        d[["content", "type_label"]].rename(columns={"type_label": "label"}).to_csv(
            os.path.join(SPLITS_DIR, f"type_cn2_{part}.csv"),
            index=False, encoding="utf-8-sig")
    print(f"二级 训练/验证/测试: {len(t[0])}/{len(t[1])}/{len(t[2])}")

    # 独立测试集
    holdout = build_holdout()
    holdout.to_csv(os.path.join(HOLDOUT_DIR, "holdout_cn.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"独立测试集（不参与训练）: {len(holdout)} 条")


if __name__ == "__main__":
    main()
