import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv(r"c:\Users\chuny\Desktop\paper_progarm\data\versions\dataset_20260411_chifraud.csv")
print("=== EXISTING DATASET ANALYSIS ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

df["text_len"] = df["text"].str.len()
print(f"Text length stats:")
print(f"  Mean: {df['text_len'].mean():.1f}")
print(f"  Median: {df['text_len'].median():.1f}")
print(f"  Min: {df['text_len'].min()}, Max: {df['text_len'].max()}")
print(f"  Std: {df['text_len'].std():.1f}")

print(f"Scenario distribution: {df['scenario'].value_counts().to_dict()}")
print(f"Source distribution: {df['source'].value_counts().to_dict()}")
print(f"URL coverage: {df['url'].notna().sum()}/{len(df)} ({df['url'].notna().mean()*100:.1f}%)")
print(f"Duplicate texts: {df['text'].duplicated().sum()}")

print("\n=== PHISHING SAMPLE CHARACTERISTICS ===")
phish = df[df["label"] == 1]
print(f"Phishing text length: mean={phish['text_len'].mean():.1f}, std={phish['text_len'].std():.1f}")
keywords = ["支付宝","微信","银行","账户","安全","风险","解冻","验证","密码","登录","中奖","领取","紧急","点击","冻结","逾期"]
for kw in keywords:
    count = phish["text"].str.contains(kw, na=False).sum()
    if count > 0:
        print(f"  {kw}: {count} ({count/len(phish)*100:.1f}%)")

print("\n=== NORMAL SAMPLE CHARACTERISTICS ===")
normal = df[df["label"] == 0]
print(f"Normal text length: mean={normal['text_len'].mean():.1f}, std={normal['text_len'].std():.1f}")

print("\n=== GAPS ANALYSIS ===")
print(f"Scenario coverage: {df['scenario'].notna().sum()}/{len(df)} ({df['scenario'].notna().mean()*100:.1f}%)")
print(f"URL coverage: {df['url'].notna().sum()}/{len(df)} ({df['url'].notna().mean()*100:.1f}%)")
print(f"Missing scenarios: {df['scenario'].isna().sum()} samples")
print(f"Short texts (<10 chars): {(df['text_len'] < 10).sum()}")
print(f"Long texts (>500 chars): {(df['text_len'] > 500).sum()}")
