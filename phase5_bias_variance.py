"""
=============================================================
  FAZ 5: YANLIYIK ve VARYANS (Bias-Variance Trade-off)
  KNN modeli üzerinden K=1…50 arası hata analizi
=============================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics   import (accuracy_score, mean_squared_error,
                                log_loss, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import GaussianNB

warnings.filterwarnings("ignore")

# ── KOYU TEMA ──────────────────────────────────────────────
DARK="#0d1117"; CARD="#161b22"; GRID="#21262d"
C1="#58a6ff"; C2="#3fb950"; C3="#ff7b72"
C4="#d2a8ff"; C5="#ffa657"; C6="#79c0ff"; TXT="#c9d1d9"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT,
    "axes.titlecolor": TXT, "xtick.color": TXT,
    "ytick.color": TXT,  "text.color": TXT,
    "grid.color": GRID,  "grid.linestyle": "--", "grid.alpha": .4,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "font.family": "monospace", "figure.dpi": 120,
})

OUT = "phase5_outputs"; os.makedirs(OUT, exist_ok=True)

def save(fname):
    plt.savefig(f"{OUT}/{fname}", bbox_inches="tight", facecolor=DARK, dpi=150)
    plt.close(); print(f"  ✅  → {OUT}/{fname}")

def hdr(t): print(f"\n{'═'*62}\n  {t}\n{'═'*62}")

# ── VERİ ───────────────────────────────────────────────────
hdr("ADIM 0 — VERİ YÜKLEME")
train = pd.read_csv("phase1_outputs/train_scaled.csv")
val   = pd.read_csv("phase1_outputs/val_scaled.csv")
test  = pd.read_csv("phase1_outputs/test_scaled.csv")
all_tv= pd.concat([train, val], ignore_index=True)

FEAT  = [c for c in train.columns if c != "Outcome"]
X_tr  = train[FEAT].values;   y_tr = train["Outcome"].values
X_va  = val[FEAT].values;     y_va = val["Outcome"].values
X_te  = test[FEAT].values;    y_te = test["Outcome"].values
X_tv  = all_tv[FEAT].values;  y_tv = all_tv["Outcome"].values

print(f"  Train={len(y_tr)} | Val={len(y_va)} | Test={len(y_te)}")

# ════════════════════════════════════════════════════════════
#  ADIM 1 — K=1…50 HATA HESAPLAMASI
# ════════════════════════════════════════════════════════════
hdr("ADIM 1 — K=1…50 HATA HESAPLAMASI")

K_MAX = 50
K_RANGE = range(1, K_MAX + 1)

train_err, val_err, test_err = [], [], []
train_acc, val_acc,  test_acc  = [], [], []
train_auc, test_auc  = [], []
complexity = []   # 1/K (model karmaşıklığı göstergesi)

for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_tr, y_tr)

    # Hata = 1 - Accuracy
    tr_err = 1 - accuracy_score(y_tr, knn.predict(X_tr))
    va_err = 1 - accuracy_score(y_va, knn.predict(X_va))
    te_err = 1 - accuracy_score(y_te, knn.predict(X_te))

    train_err.append(tr_err)
    val_err.append(va_err)
    test_err.append(te_err)
    train_acc.append(1 - tr_err)
    val_acc.append(1 - va_err)
    test_acc.append(1 - te_err)
    train_auc.append(roc_auc_score(y_tr, knn.predict_proba(X_tr)[:,1]))
    test_auc.append(roc_auc_score(y_te,  knn.predict_proba(X_te)[:,1]))
    complexity.append(1 / k)

train_err = np.array(train_err); val_err = np.array(val_err)
test_err  = np.array(test_err)
train_acc = np.array(train_acc); val_acc = np.array(val_acc)
test_acc  = np.array(test_acc)

best_k_val  = list(K_RANGE)[np.argmin(val_err)]
best_k_test = list(K_RANGE)[np.argmin(test_err)]

print(f"\n  {'K':>3}  {'Train Err':>10}  {'Val Err':>10}  {'Test Err':>10}")
print(f"  {'-'*40}")
for k in [1, 3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 40, 50]:
    m1 = " ◀ BestVal"  if k == best_k_val else ""
    m2 = " ◀ BestTest" if k == best_k_test else ""
    mark = m1 or m2
    print(f"  {k:>3}  {train_err[k-1]:>10.4f}  {val_err[k-1]:>10.4f}  {test_err[k-1]:>10.4f}{mark}")

print(f"\n  Optimum K (Val hatası min.)  : K = {best_k_val}")
print(f"  Optimum K (Test hatası min.) : K = {best_k_test}")
print(f"  K=1  Train Err = {train_err[0]:.4f}  (≈0 → Overfit / Yüksek Varyans)")
print(f"  K=50 Train Err = {train_err[49]:.4f}  Test Err = {test_err[49]:.4f}  (Underfit / Yüksek Yanlılık)")

# ════════════════════════════════════════════════════════════
#  GRAFİK 1 — ANA: Error vs. K (Tam Bias-Variance Grafiği)
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 1 — ERROR vs. K")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 5 — Bias-Variance Trade-off: Error vs. K",
             fontsize=15, fontweight="bold", color=TXT, y=1.01)

# ─ Panel A: Hata eğrileri ─────────────────────────────────
ax = axes[0]; ax.set_facecolor(CARD)
ks = list(K_RANGE)

ax.plot(ks, train_err, color=C2, lw=2.2, marker="o",
        ms=3, label="Train Error", zorder=4)
ax.plot(ks, val_err,   color=C1, lw=2.2, marker="s",
        ms=3, linestyle="--", label="Val Error", zorder=4)
ax.plot(ks, test_err,  color=C3, lw=2.5, marker="D",
        ms=3, linestyle="-.", label="Test Error", zorder=4)

# Overfit bölgesi (K küçük — yüksek varyans)
ax.axvspan(1, 5, alpha=0.10, color=C3, label="Overfit Bölgesi (Yüksek Varyans)")
# Underfit bölgesi (K büyük — yüksek yanlılık)
ax.axvspan(35, 50, alpha=0.10, color=C4, label="Underfit Bölgesi (Yüksek Yanlılık)")

# Optimum nokta
opt_te = test_err[best_k_test - 1]
ax.axvline(best_k_val, color=C5, lw=2, linestyle=":",
           label=f"Opt. K={best_k_val} (Val)", zorder=5)
ax.scatter([best_k_test], [opt_te], color=C5, s=120,
           zorder=6, edgecolors=DARK, linewidths=2)
ax.annotate(f"  Opt. K={best_k_test}\n  Test Err={opt_te:.3f}",
            xy=(best_k_test, opt_te),
            xytext=(best_k_test + 3, opt_te + 0.02),
            color=C5, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C5, lw=1.5))

# K=1 ve K=50 işaretleri
for kk, lbl, clr in [(1, "K=1\nOverfit", C3), (50, "K=50\nUnderfit", C4)]:
    ax.axvline(kk, color=clr, lw=1.2, linestyle="--", alpha=0.6)
    ax.text(kk + 0.5, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0
            else 0.45, lbl, color=clr, fontsize=8, va="top")

ax.set_xlabel("K (Komşu Sayısı)", fontsize=11)
ax.set_ylabel("Hata Oranı (1 - Accuracy)", fontsize=11)
ax.set_title("Error vs. K\n(Tüm eğriler)", fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(list(range(0, 51, 5)))
ax.legend(fontsize=8, loc="upper right"); ax.grid(True)

# ─ Panel B: Bias-Variance teorik görselleştirme ───────────
ax2 = axes[1]; ax2.set_facecolor(CARD)
# Smoothed versiyonlar
from scipy.ndimage import gaussian_filter1d
sm_tr   = gaussian_filter1d(train_err, sigma=1.5)
sm_va   = gaussian_filter1d(val_err,   sigma=1.5)
sm_te   = gaussian_filter1d(test_err,  sigma=1.5)

ax2.fill_between(ks, sm_tr, alpha=0.18, color=C2)
ax2.fill_between(ks, sm_te, alpha=0.12, color=C3)
ax2.plot(ks, sm_tr, color=C2, lw=2.5, label="Train Error (smoothed)")
ax2.plot(ks, sm_te, color=C3, lw=2.5, label="Test Error  (smoothed)")
ax2.plot(ks, sm_te - sm_tr, color=C4, lw=1.8,
         linestyle=":", label="|Test - Train| = Varyans Göstergesi")

ax2.axvline(best_k_val, color=C5, lw=2, linestyle=":",
            label=f"Opt. K={best_k_val}", zorder=5)
ax2.scatter([best_k_val], [sm_te[best_k_val - 1]],
            color=C5, s=120, zorder=6, edgecolors=DARK, linewidths=2)

ax2.set_xlabel("K (Komşu Sayısı)  →  Karmaşıklık azalır", fontsize=10)
ax2.set_ylabel("Hata Oranı (yumuşatılmış)", fontsize=10)
ax2.set_title("Bias-Variance Trade-off\n(Yumuşatılmış + Varyans aralığı)",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xticks(list(range(0, 51, 5)))
ax2.legend(fontsize=8); ax2.grid(True)

# Ok ve bölge etiketleri
ax2.annotate("← Yüksek Varyans\n(Overfit)",
             xy=(3, sm_te[2]), xytext=(8, sm_te[2] + 0.05),
             color=C3, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=C3, lw=1.3))
ax2.annotate("Yüksek Yanlılık →\n(Underfit)",
             xy=(46, sm_te[45]), xytext=(33, sm_te[45] + 0.04),
             color=C4, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=C4, lw=1.3))

plt.tight_layout(); save("01_error_vs_k.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 2 — AUC vs. K
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 2 — AUC vs. K")

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
ax.fill_between(ks, test_auc, alpha=0.15, color=C1)
ax.plot(ks, train_auc, color=C2, lw=2, marker="o", ms=3, label="Train AUC")
ax.plot(ks, test_auc,  color=C1, lw=2.2, marker="D", ms=3,
        linestyle="--", label="Test AUC")
best_auc_k = list(K_RANGE)[np.argmax(test_auc)]
ax.axvline(best_auc_k, color=C5, lw=2, linestyle=":",
           label=f"Max AUC @ K={best_auc_k}")
ax.scatter([best_auc_k], [test_auc[best_auc_k-1]],
           color=C5, s=120, zorder=5, edgecolors=DARK, linewidths=2)
ax.set_xlabel("K", fontsize=11); ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("AUC-ROC vs. K — KNN Model Ayrımcılığı",
             fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(list(range(0, 51, 5)))
ax.legend(fontsize=9); ax.grid(True)
plt.tight_layout(); save("02_auc_vs_k.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 3 — BIAS & VARIANCE AYRIŞIMI (Monte Carlo tahmin)
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 3 — BIAS² & VARIANCE AYRIŞIMI")

# Bootstrap ile bias² ve varyans tahmini
N_BOOTSTRAP = 30
rng = np.random.default_rng(42)
K_SAMPLE = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

bias2_list, var_list, total_err_list = [], [], []

for k in K_SAMPLE:
    preds = []
    for _ in range(N_BOOTSTRAP):
        idx  = rng.integers(0, len(X_tr), size=len(X_tr))
        knn_ = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_.fit(X_tr[idx], y_tr[idx])
        preds.append(knn_.predict_proba(X_te)[:, 1])
    preds = np.array(preds)               # (N_BOOTSTRAP, n_test)
    mean_pred = preds.mean(axis=0)

    # Bias² ≈ MSE(E[f(x)], y)
    bias2 = np.mean((mean_pred - y_te) ** 2)
    # Variance ≈ E[MSE(f(x), E[f(x)])]
    var   = np.mean(np.var(preds, axis=0))
    total_err_list.append(bias2 + var)
    bias2_list.append(bias2)
    var_list.append(var)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("Bias² ve Varyans Ayrışımı (Bootstrap Tahmini)",
             fontsize=14, fontweight="bold", color=TXT)

# Panel A: Stacked area
ax = axes[0]; ax.set_facecolor(CARD)
ax.fill_between(K_SAMPLE, bias2_list, alpha=0.5,
                color=C4, label="Bias²")
ax.fill_between(K_SAMPLE,
                [b+v for b,v in zip(bias2_list, var_list)],
                bias2_list, alpha=0.5, color=C3, label="Varyans")
ax.plot(K_SAMPLE, total_err_list, color=C5, lw=2.5,
        marker="D", ms=5, label="Bias² + Varyans = Toplam Hata")
ax.plot(K_SAMPLE, bias2_list, color=C4, lw=1.8,
        ls="--", marker="o", ms=4, label="Bias²")
ax.plot(K_SAMPLE, var_list, color=C3, lw=1.8,
        ls=":", marker="s", ms=4, label="Varyans")
opt_idx = np.argmin(total_err_list)
ax.scatter([K_SAMPLE[opt_idx]], [total_err_list[opt_idx]],
           color=C2, s=150, zorder=6, edgecolors=DARK,
           linewidths=2, label=f"Opt. K={K_SAMPLE[opt_idx]}")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Hata", fontsize=10)
ax.set_title("Bias² + Varyans = Toplam Hata\n(K arttıkça varyans↓, bias↑)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True)

# Panel B: bar chart
ax2 = axes[1]; ax2.set_facecolor(CARD)
x_pos = range(len(K_SAMPLE))
bar_w = 0.35
ax2.bar([x - bar_w/2 for x in x_pos], bias2_list, bar_w,
        color=C4, edgecolor=DARK, label="Bias²", alpha=0.85)
ax2.bar([x + bar_w/2 for x in x_pos], var_list, bar_w,
        color=C3, edgecolor=DARK, label="Varyans", alpha=0.85)
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels([str(k) for k in K_SAMPLE], fontsize=9)
ax2.set_xlabel("K", fontsize=10); ax2.set_ylabel("Hata Bileşeni", fontsize=10)
ax2.set_title("K'ya Göre Bias² ve Varyans\n(Gruplu Bar)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(axis="y")

plt.tight_layout(); save("03_bias_variance_decomposition.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 4 — KARAR SINIRI SEZGISEL
#  (ilk 2 özellik ile 2D projeksiyon)
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 4 — KARAR SINIRI (K=1 vs K=opt vs K=50)")

feat_idx = [FEAT.index("Glucose"), FEAT.index("BMI")]
X2_tr = X_tr[:, feat_idx]; X2_te = X_te[:, feat_idx]

h = 0.04
x_min, x_max = X2_tr[:,0].min()-0.5, X2_tr[:,0].max()+0.5
y_min, y_max = X2_tr[:,1].min()-0.5, X2_tr[:,1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("KNN Karar Sınırı: K=1 (Overfit) / K=Optimal / K=50 (Underfit)",
             fontsize=13, fontweight="bold", color=TXT)

for ax, k, title_k in zip(axes,
                           [1, best_k_val, 50],
                           [f"K=1\n(Yüksek Varyans / Overfit)",
                            f"K={best_k_val}\n(Optimal – En Düşük Hata)",
                            "K=50\n(Yüksek Yanlılık / Underfit)"]):
    ax.set_facecolor(CARD)
    knn2 = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn2.fit(X2_tr, y_tr)
    Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35,
                cmap=plt.cm.RdBu, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, levels=[0.5], colors=[C5],
               linewidths=1.8)

    for cls, clr, mrkr in [(0, C1, "o"), (1, C3, "^")]:
        mask = y_tr == cls
        ax.scatter(X2_tr[mask, 0], X2_tr[mask, 1],
                   c=clr, marker=mrkr, s=18, alpha=0.6,
                   edgecolors="none", label=f"Train cls={cls}")

    tr_err_k = 1 - accuracy_score(y_tr, knn2.predict(X2_tr))
    te_err_k = 1 - accuracy_score(y_te, knn2.predict(X2_te))
    ax.set_title(f"{title_k}\nTr.Err={tr_err_k:.3f}  Te.Err={te_err_k:.3f}",
                 fontsize=10, fontweight="bold",
                 color=C5 if k == best_k_val else TXT)
    ax.set_xlabel("Glucose (z-skor)", fontsize=9)
    ax.set_ylabel("BMI (z-skor)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=.2)

plt.tight_layout(); save("04_decision_boundary.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 5 — TÜM MODEL HATA KARŞILAŞTIRMASI
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 5 — TÜM MODEL KARŞILAŞTIRMASI")

# Diğer modellerin sabit hataları
lr   = LogisticRegression(max_iter=5000, C=1e6, random_state=42)
gnb  = GaussianNB()
lr.fit(X_tr, y_tr); gnb.fit(X_tr, y_tr)
lr_te_err  = 1 - accuracy_score(y_te, lr.predict(X_te))
gnb_te_err = 1 - accuracy_score(y_te, gnb.predict(X_te))
lr_tr_err  = 1 - accuracy_score(y_tr, lr.predict(X_tr))
gnb_tr_err = 1 - accuracy_score(y_tr, gnb.predict(X_tr))

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 5 — Tüm Modeller: Error Karşılaştırması",
             fontsize=14, fontweight="bold", color=TXT)

ax = axes[0]; ax.set_facecolor(CARD)
ax.plot(ks, train_err, color=C2, lw=2, label="KNN Train Error")
ax.plot(ks, test_err,  color=C1, lw=2, linestyle="--", label="KNN Test Error")
ax.axhline(lr_te_err,  color=C3, lw=2, linestyle="-.",
           label=f"LR Test Error={lr_te_err:.3f}")
ax.axhline(gnb_te_err, color=C4, lw=2, linestyle=":",
           label=f"GNB Test Error={gnb_te_err:.3f}")
ax.axvline(best_k_val, color=C5, lw=1.8, linestyle=":",
           label=f"KNN Opt. K={best_k_val}")
ax.set_xlabel("K", fontsize=11); ax.set_ylabel("Hata Oranı", fontsize=11)
ax.set_title("KNN vs. LR vs. GNB — Test Hataları",
             fontsize=11, fontweight="bold", pad=10)
ax.set_xticks(list(range(0,51,5)))
ax.legend(fontsize=8); ax.grid(True)

ax2 = axes[1]; ax2.set_facecolor(CARD)
bar_labels = [f"KNN\nK={best_k_val}", "Lojistik\nReg.", "Gaussian\nNB"]
tr_errs    = [train_err[best_k_val-1], lr_tr_err, gnb_tr_err]
te_errs    = [test_err[best_k_val-1],  lr_te_err, gnb_te_err]
x   = np.arange(3); w = 0.35
b1  = ax2.bar(x - w/2, tr_errs, w, color=C2,
              edgecolor=DARK, label="Train Error", alpha=0.85)
b2  = ax2.bar(x + w/2, te_errs, w, color=C3,
              edgecolor=DARK, label="Test Error", alpha=0.85)
for bar, v in zip(list(b1)+list(b2), tr_errs+te_errs):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.003,
             f"{v:.3f}", ha="center", fontsize=9,
             fontweight="bold", color=TXT)
ax2.set_xticks(x); ax2.set_xticklabels(bar_labels, fontsize=10)
ax2.set_ylabel("Hata Oranı", fontsize=11)
ax2.set_title("Tüm Modeller — Train vs. Test Hata",
              fontsize=11, fontweight="bold", pad=10)
ax2.legend(fontsize=9); ax2.grid(axis="y")

plt.tight_layout(); save("05_all_models_error.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 6 — ÖZET DASHBOARD
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 6 — ÖZET DASHBOARD")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 5 — Bias-Variance Trade-off Özet Dashboard",
             fontsize=16, fontweight="bold", color=TXT, y=1.00)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# P1: Error vs K (ana grafik)
ax1 = fig.add_subplot(gs[0, 0:2]); ax1.set_facecolor(CARD)
ax1.plot(ks, train_err, color=C2, lw=2, marker="o", ms=2, label="Train Err")
ax1.plot(ks, val_err,   color=C1, lw=2, linestyle="--", marker="s", ms=2, label="Val Err")
ax1.plot(ks, test_err,  color=C3, lw=2.2, linestyle="-.", marker="D", ms=2, label="Test Err")
ax1.axvspan(1, 5, alpha=0.08, color=C3)
ax1.axvspan(38, 50, alpha=0.08, color=C4)
ax1.axvline(best_k_val, color=C5, lw=2, linestyle=":")
ax1.scatter([best_k_val], [val_err[best_k_val-1]],
            color=C5, s=100, zorder=6, edgecolors=DARK, linewidths=2)
ax1.set_xlabel("K →  (Sağa: Daha Basit Model)"); ax1.set_ylabel("Hata Oranı")
ax1.set_title(f"Error vs. K  |  Opt. K={best_k_val}  |  Test Err @ Opt={test_err[best_k_val-1]:.3f}",
              fontsize=11, fontweight="bold"); ax1.set_xticks(range(0,51,5))
ax1.text(2.5, max(test_err)*0.97, "Overfit\n(Yüksek Var.)",
         color=C3, fontsize=8, ha="center", va="top")
ax1.text(44, max(test_err)*0.97, "Underfit\n(Yüksek Bias)",
         color=C4, fontsize=8, ha="center", va="top")
ax1.legend(fontsize=9); ax1.grid(True)

# P2: Bias-Var pasta (Opt K)
ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(CARD)
opt_b2  = bias2_list[K_SAMPLE.index(K_SAMPLE[opt_idx])]
opt_var = var_list[K_SAMPLE.index(K_SAMPLE[opt_idx])]
ax2.pie([opt_b2, opt_var],
        labels=["Bias²", "Varyans"],
        colors=[C4, C3], autopct="%1.1f%%",
        textprops={"color": TXT, "fontsize": 10},
        wedgeprops={"edgecolor": DARK, "linewidth": 2},
        startangle=90)
ax2.set_title(f"K={K_SAMPLE[opt_idx]}: Bias² / Varyans Dağılımı",
              fontsize=10, fontweight="bold")

# P3: K=1 vs K_opt vs K=50 karşılaştırma bar
ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(CARD)
k_cmp   = [1, best_k_val, 50]
tr_cmp  = [train_err[k-1] for k in k_cmp]
te_cmp  = [test_err[k-1]  for k in k_cmp]
xlbl    = [f"K={k}" for k in k_cmp]
xp = np.arange(3); w2 = 0.35
ax3.bar(xp-w2/2, tr_cmp, w2, color=C2, edgecolor=DARK, label="Train")
ax3.bar(xp+w2/2, te_cmp, w2, color=C3, edgecolor=DARK, label="Test")
for xv, tr, te in zip(xp, tr_cmp, te_cmp):
    ax3.text(xv-w2/2, tr+0.005, f"{tr:.3f}", ha="center", fontsize=8, color=TXT)
    ax3.text(xv+w2/2, te+0.005, f"{te:.3f}", ha="center", fontsize=8, color=TXT)
ax3.set_xticks(xp); ax3.set_xticklabels(xlbl)
ax3.set_ylabel("Hata Oranı")
ax3.set_title("K=1 / Opt / K=50 Karşılaştırması", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(axis="y")

# P4: Bias-Var line
ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(CARD)
ax4.plot(K_SAMPLE, bias2_list, color=C4, lw=2, marker="o", ms=5, label="Bias²")
ax4.plot(K_SAMPLE, var_list,   color=C3, lw=2, marker="s", ms=5, label="Varyans")
ax4.plot(K_SAMPLE, total_err_list, color=C5, lw=2.5, marker="D",
         ms=5, label="Toplam Hata")
ax4.axvline(K_SAMPLE[opt_idx], color=C2, lw=1.8, linestyle=":",
            label=f"Opt. K={K_SAMPLE[opt_idx]}")
ax4.set_xlabel("K"); ax4.set_ylabel("Hata Bileşeni")
ax4.set_title("Bias² & Varyans Eğrileri", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(True)

# P5: AUC vs K
ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor(CARD)
ax5.fill_between(ks, test_auc, alpha=0.15, color=C1)
ax5.plot(ks, test_auc, color=C1, lw=2, marker=".", ms=4, label="Test AUC")
ax5.axvline(best_auc_k, color=C5, lw=1.8, linestyle=":",
            label=f"Max AUC @ K={best_auc_k}")
ax5.set_xlabel("K"); ax5.set_ylabel("AUC-ROC")
ax5.set_title("AUC vs. K", fontsize=10, fontweight="bold")
ax5.set_xticks(list(range(0,51,5)))
ax5.legend(fontsize=8); ax5.grid(True)

save("06_summary_dashboard.png")

# ── SONUÇ ──────────────────────────────────────────────────
hdr("FAZ 5 — TAMAMLANDI ✅")

opt_te_err = test_err[best_k_val - 1]
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │               FAZ 5 ÖZET                               │
  ├─────────────────────────────────────────────────────────┤
  │  K=1  (Yüksek Varyans)                                 │
  │    Train Err = {train_err[0]:.4f}  Test Err = {test_err[0]:.4f}         │
  │    → Model eğitim verisini EZBERLEDI (Overfit)         │
  ├─────────────────────────────────────────────────────────┤
  │  K={best_k_val} (Optimum — En Düşük Val. Hatası)             │
  │    Train Err = {train_err[best_k_val-1]:.4f}  Test Err = {opt_te_err:.4f}       │
  │    → Bias-Variance DENGESİ sağlandı ✅                 │
  ├─────────────────────────────────────────────────────────┤
  │  K=50 (Yüksek Yanlılık)                                │
  │    Train Err = {train_err[49]:.4f}  Test Err = {test_err[49]:.4f}         │
  │    → Model çok BASIT kaldı (Underfit)                  │
  ├─────────────────────────────────────────────────────────┤
  │  Bootstrap Bias² Analizi (K={K_SAMPLE[opt_idx]} opt.)            │
  │    Bias²    = {bias2_list[opt_idx]:.4f}                          │
  │    Varyans  = {var_list[opt_idx]:.4f}                          │
  │    Toplam   = {total_err_list[opt_idx]:.4f}                          │
  ├─────────────────────────────────────────────────────────┤
  │  Grafikler (6 adet) → {OUT}/            │
  └─────────────────────────────────────────────────────────┘
""")

# CSV kaydet
df_out = pd.DataFrame({
    "K": list(K_RANGE),
    "Train_Err": train_err, "Val_Err": val_err, "Test_Err": test_err,
    "Train_Acc": train_acc, "Val_Acc":  val_acc,  "Test_Acc":  test_acc,
    "Test_AUC": test_auc,
})
df_out.to_csv(f"{OUT}/error_vs_k.csv", index=False)
print(f"  CSV kaydedildi → {OUT}/error_vs_k.csv")
