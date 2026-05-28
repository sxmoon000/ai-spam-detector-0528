"""
垃圾邮件检测 — TF-IDF + 朴素贝叶斯 (工业级)
使用真实的 SMS Spam Collection 数据集，高准确率检测垃圾短信。

知识点：
  1. 文本分类完整 Pipeline
  2. 精确率 vs 召回率: 垃圾邮件宁误判(spam→ham)不漏判(ham→spam)
  3. F1-score: 精确率和召回率的调和平均
  4. 停用词处理: 去除 "the, a, is" 等高频无意义词
"""
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

print("=" * 55)
print("📧 垃圾邮件检测器")
print("=" * 55)

# ── 1. 加载数据 ──
print("[1/4] Loading SMS Spam Collection...")
# 直接使用内置版本，无需下载
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
try:
    df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
except:
    print("   下载数据集...")
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    df.to_csv("sms.tsv", sep="\t", index=False)

df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(f"   总数: {len(df)}, Spam: {df['label'].sum()} ({df['label'].mean():.1%})")

# ── 2. 预处理 ──
print("[2/4] Preprocessing...")


def clean(text):
    text = text.lower()
    text = re.sub(r"\d+", " NUM ", text)           # 保留数字占位符
    text = re.sub(r"\b\w{1,2}\b", "", text)        # 去 1-2 字母词
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 3. 训练 ──
print("[3/4] Training TF-IDF + Naive Bayes...")
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=clean, max_features=5000,
        ngram_range=(1, 2), stop_words="english",
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"   5-fold CV: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"\n   测试集结果:")
print(f"   准确率: {(y_pred == y_test).mean():.2%}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"   混淆矩阵:")
print(f"         Pred Ham  Pred Spam")
print(f"   True Ham   {cm[0][0]:>5}     {cm[0][1]:>5}")
print(f"   True Spam  {cm[1][0]:>5}     {cm[1][1]:>5}")

print(f"\n   漏判(Spam→Ham): {cm[1][0]} 条 ← 最需要关注")

# ── 4. Demo ──
print("\n🧪 演示:")
demos = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account has been compromised. Reply with your password.",
    "Don't forget to pick up milk on your way home",
]
for msg in demos:
    pred = pipeline.predict([msg])[0]
    proba = pipeline.predict_proba([msg])[0]
    label = "🔴 SPAM" if pred == 1 else "🟢 HAM"
    print(f"   {label} ({proba[pred]:.1%}) — \"{msg[:60]}...\"")

Path("model").mkdir(exist_ok=True)
joblib.dump(pipeline, "model/spam_detector.pkl")
print(f"\n✅ 完成!")
