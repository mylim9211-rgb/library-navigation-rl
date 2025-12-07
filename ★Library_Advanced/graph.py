# graph.py
import matplotlib.pyplot as plt

# 🔤 한글 폰트 설정 (Windows 기준)
plt.rcParams["font.family"] = "Malgun Gothic"  # 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False     # 마이너스 깨짐 방지

# ------------------------------------------------
# 1. S-start vs Random-start 성공률
#   (advanced_eval.py, seed=42 결과 기준)
# ------------------------------------------------
labels = ["S-start", "Random-start"]
rates = [88.0, 74.0]

plt.figure(figsize=(5, 4))
bars = plt.bar(labels, rates)

for i, v in enumerate(rates):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.ylim(0, 100)
plt.ylabel("Success Rate (%)")
plt.title("Advanced Grid – S 시작 vs Random 시작 성공률")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 2. 랜덤 시작 시 타겟별 성공률
# ------------------------------------------------
targets = ["A", "B", "C", "D", "E", "F", "G"]
target_rates = [82.6, 91.4, 75.0, 44.8, 87.5, 91.7, 48.5]

plt.figure(figsize=(6, 4))
bars = plt.bar(targets, target_rates)

for i, v in enumerate(target_rates):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.ylim(0, 100)
plt.ylabel("Success Rate (%)")
plt.title("Advanced Grid – 랜덤 시작 시 타겟별 성공률")
plt.tight_layout()
plt.show()
