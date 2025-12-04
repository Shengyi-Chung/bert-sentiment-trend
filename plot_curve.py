import matplotlib.pyplot as plt

# 把刚才终端里的 4 个 epoch 数值照抄即可
epochs = [1, 2, 3, 4]
f1     = [0.7219, 0.7287, 0.7325, 0.7251]

plt.figure(figsize=(4, 3))               # NeurIPS 单栏宽度
plt.plot(epochs, f1, marker='o', color='#1f77b4', linewidth=1.8)
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Macro-F1", fontsize=10)
plt.title("RoBERTa-base TweetEval Sentiment", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assets/stage3_final/f1_curve.png", dpi=300)
print("✅ assets/stage3_final/f1_curve.png saved")