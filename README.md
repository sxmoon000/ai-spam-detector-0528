# 📧 垃圾邮件检测

> TF-IDF + Naive Bayes — 真实 SMS 数据集

## 🧠 知识点
- **精确率(Precision)**: 标记为Spam的中有多少真的是Spam — 减少误伤
- **召回率(Recall)**: 真正的Spam中有多少被找到 — 减少漏网
- **F1-score**: 两者冲突时的平衡指标
- **真实数据集**: SMS Spam Collection, 5574条短信, 13% spam

## 🚀 运行
```bash
pip install -r requirements.txt && python src/train.py
```

---

Day 10 | 2026-05-28 | [sxmoon000](https://github.com/sxmoon000)
