"""
规则引擎 + 邮件解析器 + 模型融合

v1.1 新增:
  • 混合引擎: ML模型 + 规则引擎 双重过滤
  • 邮件头解析: From/To/Subject/Headers分析
  • 10条反垃圾规则 (SPF/DKIM/IP/URL检查)
  • 模型融合: NB投票 + 规则加权
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter
from urllib.parse import urlparse


@dataclass
class Email:
    """邮件模型"""
    from_addr: str = ""
    to_addr: str = ""
    subject: str = ""
    body: str = ""
    headers: dict = field(default_factory=dict)
    raw: str = ""


@dataclass
class SpamVerdict:
    is_spam: bool
    score: float       # 0-100
    ml_score: float = 0.0
    rule_score: float = 0.0
    rules_triggered: List[str] = field(default_factory=list)
    explanation: str = ""


class EmailParser:
    """邮件解析器"""

    @classmethod
    def parse(cls, raw_email: str) -> Email:
        """解析原始邮件"""
        email = Email(raw=raw_email)

        # 解析头部
        header_patterns = {
            "From": r"^From:\s*(.+)$",
            "To": r"^To:\s*(.+)$",
            "Subject": r"^Subject:\s*(.+)$",
            "Date": r"^Date:\s*(.+)$",
            "Reply-To": r"^Reply-To:\s*(.+)$",
            "Message-ID": r"^Message-ID:\s*(.+)$",
            "Received": r"^Received:\s*(.+)$",
            "Content-Type": r"^Content-Type:\s*(.+)$",
        }

        for key, pattern in header_patterns.items():
            m = re.search(pattern, raw_email, re.MULTILINE | re.IGNORECASE)
            if m:
                email.headers[key] = m.group(1).strip()
                if key == "From":
                    email.from_addr = email.headers[key]
                elif key == "To":
                    email.to_addr = email.headers[key]
                elif key == "Subject":
                    email.subject = email.headers[key]

        # 解析正文
        body_match = re.search(r'\n\n(.+)', raw_email, re.DOTALL)
        if body_match:
            email.body = body_match.group(1).strip()

        return email


class RuleEngine:
    """反垃圾规则引擎"""

    # 垃圾邮件特征规则
    RULES = [
        {
            "name": "spf_fail",
            "weight": 15,
            "check": lambda e: "spf=fail" in e.raw.lower() or "spf=softfail" in e.raw.lower(),
            "desc": "SPF 验证失败，发件人可能伪造",
        },
        {
            "name": "dkim_missing",
            "weight": 10,
            "check": lambda e: "dkim=" not in e.raw.lower() and "dkim-signature" not in e.raw.lower(),
            "desc": "缺少 DKIM 签名",
        },
        {
            "name": "urgent_language",
            "weight": 12,
            "check": lambda e: bool(re.search(r'(?i)(urgent|immediate action|account.*suspend|limited time|act now|click here.*now)',
                                              e.subject + " " + e.body[:500])),
            "desc": "包含紧急/恐吓性语言",
        },
        {
            "name": "prize_claim",
            "weight": 15,
            "check": lambda e: bool(re.search(r'(?i)(congratulations.*won|prize|lottery|claim.*reward|you.*selected|free.*gift)',
                                              e.subject + " " + e.body[:300])),
            "desc": "中奖/奖品诈骗模式",
        },
        {
            "name": "suspicious_url",
            "weight": 10,
            "check": lambda e: bool(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', e.body)),
            "desc": "包含 IP 地址形式的可疑链接",
        },
        {
            "name": "shortened_url",
            "weight": 5,
            "check": lambda e: bool(re.search(r'(?i)(bit\.ly|tinyurl|t\.co|ow\.ly|goo\.gl|is\.gd)', e.body)),
            "desc": "使用短链接服务 (隐藏真实URL)",
        },
        {
            "name": "poor_grammar",
            "weight": 8,
            "check": lambda e: len(re.findall(r'(?i)\b(?:kindly|dear.*sir|greetings.*day|am\s+\w+ing)\b', e.body)) > 2,
            "desc": "钓鱼邮件常见语法错误",
        },
        {
            "name": "mismatched_domain",
            "weight": 10,
            "check": lambda e: cls._check_domain_mismatch(e),
            "desc": "发件域名与Reply-To域名不一致",
        },
        {
            "name": "excessive_caps",
            "weight": 5,
            "check": lambda e: len(re.findall(r'[A-Z]{4,}', e.subject)) > 2,
            "desc": "标题过多大写字母",
        },
        {
            "name": "money_transfer",
            "weight": 18,
            "check": lambda e: bool(re.search(r'(?i)(wire transfer|western union|money gram|bank.*account.*number|crypto.*wallet)',
                                              e.body[:500])),
            "desc": "涉及资金转账要求",
        },
    ]

    @staticmethod
    def _check_domain_mismatch(email: Email) -> bool:
        """检查域名不一致"""
        def extract_domain(addr):
            m = re.search(r'@([\w.-]+)', addr)
            return m.group(1) if m else ""
        from_domain = extract_domain(email.from_addr)
        reply_domain = extract_domain(email.headers.get("Reply-To", ""))
        if from_domain and reply_domain and from_domain != reply_domain:
            return True
        return False

    def evaluate(self, email: Email) -> Tuple[float, List[str]]:
        """评估邮件，返回规则分和触发的规则"""
        total_weight = 0
        triggered = []

        for rule in self.RULES:
            try:
                if rule["check"](email):
                    total_weight += rule["weight"]
                    triggered.append(rule["desc"])
            except:
                pass

        return min(100, total_weight), triggered


class HybridDetector:
    """混合检测器: ML + 规则"""

    def __init__(self, ml_model=None, vectorizer=None):
        self.ml_model = ml_model
        self.vectorizer = vectorizer
        self.rule_engine = RuleEngine()
        self.parser = EmailParser()

    def detect(self, text_or_email: str) -> SpamVerdict:
        """综合检测"""
        # 1. 规则引擎
        email = self.parser.parse(text_or_email) if text_or_email.startswith("From:") \
                else Email(subject="", body=text_or_email, raw=text_or_email)
        rule_score, triggered = self.rule_engine.evaluate(email)

        # 2. ML 模型
        ml_score = 0.0
        if self.ml_model and self.vectorizer:
            try:
                text = email.subject + " " + email.body
                proba = self.ml_model.predict_proba([text])[0]
                ml_score = proba[1] * 100
            except:
                ml_score = 0.0

        # 3. 融合: 规则权重 40%, ML 权重 60%
        final_score = rule_score * 0.4 + ml_score * 0.6 if ml_score > 0 else rule_score
        is_spam = final_score >= 40

        return SpamVerdict(
            is_spam=is_spam,
            score=round(final_score, 1),
            ml_score=round(ml_score, 1),
            rule_score=round(rule_score, 1),
            rules_triggered=triggered,
            explanation=self._explain(final_score, triggered),
        )

    def _explain(self, score: float, triggered: List[str]) -> str:
        if score >= 70:
            return f"高概率垃圾邮件 (综合分 {score})"
        elif score >= 40:
            return f"疑似垃圾邮件 (综合分 {score})"
        else:
            return "正常邮件"

    def report(self, text: str):
        verdict = self.detect(text)
        print("=" * 55)
        print("📧 垃圾邮件混合检测报告")
        print("=" * 55)
        print(f"\n   🏷️  判定: {'🔴 SPAM' if verdict.is_spam else '🟢 HAM'}")
        print(f"   📊 得分: {verdict.score}/100")
        print(f"   🤖 ML得分: {verdict.ml_score}")
        print(f"   📏 规则得分: {verdict.rule_score}")
        print(f"   📝 说明: {verdict.explanation}")
        if verdict.rules_triggered:
            print(f"\n   🚩 触发的规则:")
            for r in verdict.rules_triggered:
                print(f"      · {r}")


def main():
    print("=" * 55)
    print("📧 反垃圾混合引擎 v1.1")
    print("=" * 55)

    detector = HybridDetector()

    # 测试邮件
    samples = [
        """From: prince@nigeria.gov
To: user@example.com
Subject: URGENT: Your $5,000,000 Prize Claim

Congratulations! You have been selected to receive $5,000,000.
Please send your bank account number and wire transfer details immediately.
Act now, this offer expires in 24 hours!
Click here: http://192.168.1.1/claim.php""",

        "Hey, are we still meeting for lunch tomorrow? Let me know what time works best for you.",
    ]

    for i, sample in enumerate(samples, 1):
        detector.report(sample)
        print()

    # 规则库展示
    print(f"\n📏 反垃圾规则库 ({len(RuleEngine.RULES)} 条):")
    print(f"   {'规则':<22} {'权重':>4}")
    print(f"   {'─'*28}")
    for rule in RuleEngine.RULES[:8]:
        print(f"   {rule['name']:<22} {rule['weight']:>3}%")

    print(f"\n✅ 混合检测器演示完成")


if __name__ == "__main__":
    main()
