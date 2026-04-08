"""
=============================================================
  FAZ 4: SINIFLANDIRMA ve İSTATİSTİKSEL YORUM
  • Lojistik Regresyon (sklearn + scipy z/p istatistikleri)
  • KNN (K=1…20, Öklid mesafesi)
  • Gaussian Naive Bayes
============================================================="""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

from sklearn.linear_model   import LogisticRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.metrics        import (accuracy_score, confusion_matrix,
                                    roc_auc_score, roc_curve,
                                    classification_report, f1_score)
from scipy                  import stats as sp_stats
from scipy.special import expit as sigmoid_fn

warnings.filterwarnings("ignore")

# ── KOYU TEMA ──────────────────────────────────────────────
DARK = "#0d1117"; CARD = "#161b22"; GRID = "#21262d"
C1="#58a6ff"; C2="#3fb950"; C3="#ff7b72"
C4="#d2a8ff"; C5="#ffa657"; C6="#79c0ff"
TXT="#c9d1d9"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT,
    "axes.titlecolor": TXT, "xtick.color": TXT,
    "ytick.color": TXT, "text.color": TXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": .4,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "font.family": "monospace", "figure.dpi": 120,
})

OUT = "phase4_outputs"; os.makedirs(OUT, exist_ok=True)

def save(fname):
    plt.savefig(f"{OUT}/{fname}", bbox_inches="tight", facecolor=DARK, dpi=150)
    plt.close(); print(f"  ✅  → {OUT}/{fname}")

def hdr(t): print(f"\n{'═'*62}\n  {t}\n{'═'*62}")

# ── VERİ ───────────────────────────────────────────────────
hdr("ADIM 0 — VERİ YÜKLEME")
train = pd.read_csv("phase1_outputs/train_scaled.csv")
val   = pd.read_csv("phase1_outputs/val_scaled.csv")
test  = pd.read_csv("phase1_outputs/test_scaled.csv")

FEAT = [c for c in train.columns if c != "Outcome"]
X_tr = train[FEAT].values;  y_tr = train["Outcome"].values
X_va = val[FEAT].values;    y_va = val["Outcome"].values
X_te = test[FEAT].values;   y_te = test["Outcome"].values

print(f"  Train={len(y_tr)}  Val={len(y_va)}  Test={len(y_te)}")

# ════════════════════════════════════════════════════════════
#  A — LOJİSTİK REGRESYON (statsmodels istatistikleri)
# ════════════════════════════════════════════════════════════
hdr("A — LOJİSTİK REGRESYON (sklearn + scipy istatistikleri)")

# sklearn ile fit et
sk_lr_stats = LogisticRegression(max_iter=5000, C=1e6, random_state=42)
sk_lr_stats.fit(X_tr, y_tr)

coef_names = ["intercept"] + FEAT
coefs      = np.concatenate([[sk_lr_stats.intercept_[0]], sk_lr_stats.coef_[0]])

# Fisher Information Matrix → std hataları
X_aug   = np.hstack([np.ones((len(X_tr),1)), X_tr])
y_hat   = sigmoid_fn(X_aug @ coefs)
W_diag  = y_hat * (1 - y_hat)           # ağırlık matrisi diyagoneli
XTWX    = X_aug.T @ (W_diag[:, None] * X_aug)
try:
    cov_mat  = np.linalg.inv(XTWX)
    std_errs = np.sqrt(np.diag(cov_mat))
except np.linalg.LinAlgError:
    std_errs = np.full_like(coefs, np.nan)

z_stats     = coefs / (std_errs + 1e-15)
p_values    = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stats)))
conf_lo     = coefs - 1.96 * std_errs
conf_hi     = coefs + 1.96 * std_errs
odds_ratios = np.exp(coefs)

n_params = len(coefs)
df_stats = pd.DataFrame({
    "Parametre"  : coef_names[:n_params],
    "Katsayı"    : coefs[:n_params],
    "Std.Hata"   : std_errs[:n_params],
    "Z-ist."     : z_stats[:n_params],
    "p-değeri"   : p_values[:n_params],
    "CI_lo"      : conf_lo[:n_params],
    "CI_hi"      : conf_hi[:n_params],
    "Odds Oranı" : odds_ratios[:n_params],
    "Anlamlı"    : p_values[:n_params] < 0.05,
})
df_stats.to_csv(f"{OUT}/logistic_stats.csv", index=False)

print(f"\n  {'Parametre':<28} {'Katsayı':>9} {'Z':>7} {'p':>8}  Sig")
print(f"  {'-'*58}")
for _, row in df_stats.iterrows():
    sig = "★" if row["Anlamlı"] else " "
    print(f"  {row['Parametre']:<28} {row['Katsayı']:>9.4f} "
          f"{row['Z-ist.']:>7.3f} {row['p-değeri']:>8.4f}  {sig}")

# predict için aynı modeli kullan
sk_lr = sk_lr_stats

# ── GRAFİK 1: İstatistik Özeti ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("Lojistik Regresyon — İstatistiksel Özet", fontsize=14,
             fontweight="bold", color=TXT)

feat_only = df_stats.iloc[1:]  # intercept hariç

# Panel 1 – Z istatistikleri
ax = axes[0]; ax.set_facecolor(CARD)
colors_z = [C2 if v > 0 else C3 for v in feat_only["Z-ist."]]
bars = ax.barh(feat_only["Parametre"], feat_only["Z-ist."],
               color=colors_z, edgecolor=DARK, linewidth=1)
ax.axvline(1.96, color=C5, lw=1.5, ls="--", label="+1.96 (p<.05)")
ax.axvline(-1.96, color=C5, lw=1.5, ls="--")
ax.axvline(0, color=TXT, lw=.8, alpha=.5)
ax.set_title("Z-İstatistiği", fontsize=11, pad=8, fontweight="bold")
ax.set_xlabel("Z", fontsize=9); ax.legend(fontsize=8); ax.grid(axis="x")

# Panel 2 – p-değerleri (log ölçeği)
ax = axes[1]; ax.set_facecolor(CARD)
p_colors = [C2 if p < 0.05 else C3 for p in feat_only["p-değeri"]]
ax.barh(feat_only["Parametre"], feat_only["p-değeri"],
        color=p_colors, edgecolor=DARK, linewidth=1)
ax.axvline(0.05, color=C5, lw=2, ls="--", label="α=0.05")
ax.set_xscale("log")
ax.set_title("p-değerleri (log ölçeği)", fontsize=11, pad=8, fontweight="bold")
ax.set_xlabel("p-değeri", fontsize=9); ax.legend(fontsize=8); ax.grid(axis="x")
for i, (p, name) in enumerate(zip(feat_only["p-değeri"], feat_only["Parametre"])):
    if p < 0.05:
        ax.text(p * 1.05, i, "★", va="center", fontsize=11, color=C5)

# Panel 3 – Odds oranları (log-odds görselleştirilmiş)
ax = axes[2]; ax.set_facecolor(CARD)
ors = feat_only["Odds Oranı"]
or_colors = [C2 if v > 1 else C3 for v in ors]
ax.barh(feat_only["Parametre"], ors, color=or_colors,
        edgecolor=DARK, linewidth=1)
ax.axvline(1, color=C5, lw=1.5, ls="--", label="OR=1 (etki yok)")
ax.set_title("Odds Oranları (eᵝ)", fontsize=11, pad=8, fontweight="bold")
ax.set_xlabel("Odds Ratio", fontsize=9); ax.legend(fontsize=8); ax.grid(axis="x")
for bar, v in zip(ax.patches, ors):
    ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=7.5, color=TXT)

plt.tight_layout(); save("01_logistic_stats.png")

# ── GRAFİK 2: Güven Aralıkları (forest plot) ───────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
y_pos = range(len(feat_only))
for i, (_, row) in enumerate(feat_only.iterrows()):
    color = C2 if row["Anlamlı"] else C3
    ax.plot([row["CI_lo"], row["CI_hi"]], [i, i],
            lw=2.5, color=color, solid_capstyle="round")
    ax.scatter(row["Katsayı"], i, color=color, s=70, zorder=5)
ax.axvline(0, color=C5, lw=1.5, ls="--", alpha=.8, label="β=0")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(feat_only["Parametre"], fontsize=9)
ax.set_xlabel("Katsayı (95% CI)", fontsize=10)
ax.set_title("Forest Plot — Lojistik Regresyon Katsayıları (95% Güven Aralığı)",
             fontsize=12, pad=12, fontweight="bold")
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color=C2, label="p < 0.05 ★ Anlamlı"),
    mpatches.Patch(color=C3, label="p ≥ 0.05  Anlamsız"),
    plt.Line2D([0],[0], color=C5, ls="--", label="β = 0"),
], fontsize=9)
ax.grid(axis="x")
plt.tight_layout(); save("02_forest_plot.png")

# Log-odds yorumu yazdır
hdr("LOG-ODDS YORUMU")
sig_feats = df_stats[df_stats["Anlamlı"] & (df_stats["Parametre"] != "intercept")]
for _, row in sig_feats.iterrows():
    direction = "artırır" if row["Katsayı"] > 0 else "azaltır"
    print(f"\n  📌 {row['Parametre']}:")
    print(f"     β = {row['Katsayı']:.4f}  |  Z = {row['Z-ist.']:.3f}  |  p = {row['p-değeri']:.4f}")
    print(f"     ▸ 1 birim artış → log-odds'u {abs(row['Katsayı']):.4f} kadar {direction}")
    print(f"     ▸ Odds {row['Odds Oranı']:.3f}× çarpılır")

# ════════════════════════════════════════════════════════════
#  B — KNN (K=1…20)
# ════════════════════════════════════════════════════════════
hdr("B — KNN (K=1…20, Öklid)")

K_RANGE = range(1, 21)
knn_val_acc, knn_test_acc, knn_auc = [], [], []

for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_tr, y_tr)
    knn_val_acc.append(accuracy_score(y_va, knn.predict(X_va)))
    knn_test_acc.append(accuracy_score(y_te, knn.predict(X_te)))
    knn_auc.append(roc_auc_score(y_te, knn.predict_proba(X_te)[:, 1]))

best_k   = list(K_RANGE)[np.argmax(knn_val_acc)]
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
best_knn.fit(X_tr, y_tr)

print(f"\n  {'K':>3}  {'Val Acc':>9}  {'Test Acc':>10}  {'AUC':>7}")
print(f"  {'-'*34}")
for k, va, ta, au in zip(K_RANGE, knn_val_acc, knn_test_acc, knn_auc):
    mark = " ◀ BEST" if k == best_k else ""
    print(f"  {k:>3}  {va:>9.4f}  {ta:>10.4f}  {au:>7.4f}{mark}")

# ── GRAFİK 3: K vs Accuracy ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("KNN — K Değeri Analizi (Öklid Mesafesi)",
             fontsize=14, fontweight="bold", color=TXT)

ax = axes[0]; ax.set_facecolor(CARD)
ax.plot(K_RANGE, knn_val_acc,  color=C1, lw=2.2, marker="o", ms=5, label="Val Acc")
ax.plot(K_RANGE, knn_test_acc, color=C3, lw=2,   marker="s", ms=5, label="Test Acc", ls="--")
ax.axvline(best_k, color=C5, lw=1.8, ls=":", label=f"Best K={best_k}")
ax.scatter([best_k], [knn_val_acc[best_k-1]],
           color=C5, s=100, zorder=5)
ax.set_xlabel("K (Komşu Sayısı)", fontsize=10)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title("K vs. Accuracy", fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(list(K_RANGE)); ax.legend(fontsize=9); ax.grid(True)

ax = axes[1]; ax.set_facecolor(CARD)
ax.fill_between(K_RANGE, knn_auc, alpha=0.2, color=C2)
ax.plot(K_RANGE, knn_auc, color=C2, lw=2.2, marker="D", ms=5, label="Test AUC")
ax.axvline(best_k, color=C5, lw=1.8, ls=":", label=f"Best K={best_k}")
ax.set_xlabel("K", fontsize=10); ax.set_ylabel("AUC-ROC", fontsize=10)
ax.set_title("K vs. AUC-ROC", fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(list(K_RANGE)); ax.legend(fontsize=9); ax.grid(True)

plt.tight_layout(); save("03_knn_k_analysis.png")

# ── GRAFİK 4: KNN Underfitting / Overfitting ───────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
train_acc_k = []
for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_tr, y_tr)
    train_acc_k.append(accuracy_score(y_tr, knn.predict(X_tr)))

ax.plot(K_RANGE, train_acc_k,  color=C2, lw=2, marker="o", ms=4, label="Train Acc")
ax.plot(K_RANGE, knn_val_acc,  color=C1, lw=2, marker="s", ms=4, label="Val Acc")
ax.plot(K_RANGE, knn_test_acc, color=C3, lw=2, marker="D", ms=4, label="Test Acc", ls="--")
ax.axvline(best_k, color=C5, lw=2, ls=":", label=f"Best K={best_k}")
ax.fill_between(K_RANGE,
                [t - v for t, v in zip(train_acc_k, knn_val_acc)],
                0, alpha=0.08, color=C4, label="Overfitting Farkı")
ax.set_xlabel("K", fontsize=11); ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("K=1 (Overfit) → K=20 (Underfit) Geçişi", fontsize=12,
             fontweight="bold", pad=10)
ax.set_xticks(list(K_RANGE)); ax.legend(fontsize=9); ax.grid(True)
plt.tight_layout(); save("04_knn_bias_variance.png")

# ════════════════════════════════════════════════════════════
#  C — GAUSSIAN NAIVE BAYES
# ════════════════════════════════════════════════════════════
hdr("C — GAUSSIAN NAIVE BAYES")

gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
gnb_val_pred   = gnb.predict(X_va)
gnb_test_pred  = gnb.predict(X_te)
gnb_val_proba  = gnb.predict_proba(X_va)[:, 1]
gnb_test_proba = gnb.predict_proba(X_te)[:, 1]

gnb_val_acc  = accuracy_score(y_va, gnb_val_pred)
gnb_test_acc = accuracy_score(y_te, gnb_test_pred)
gnb_val_auc  = roc_auc_score(y_va, gnb_val_proba)
gnb_test_auc = roc_auc_score(y_te, gnb_test_proba)

print(f"\n  GNB Val  Acc={gnb_val_acc:.4f}  AUC={gnb_val_auc:.4f}")
print(f"  GNB Test Acc={gnb_test_acc:.4f}  AUC={gnb_test_auc:.4f}")

# Normal dağılım varsayımı görselleştirmesi
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
fig.patch.set_facecolor(DARK)
fig.suptitle("Gaussian Naive Bayes — Özellik Dağılımları (Sınıf Bazlı)",
             fontsize=13, fontweight="bold", color=TXT)
x_all = np.vstack([X_tr, X_va])
y_all = np.concatenate([y_tr, y_va])

for i, (feat, ax) in enumerate(zip(FEAT, axes.flatten())):
    ax.set_facecolor(CARD)
    for cls, clr, lbl in [(0, C1, "Negatif (0)"), (1, C3, "Pozitif (1)")]:
        vals = x_all[y_all == cls, i]
        ax.hist(vals, bins=22, color=clr, alpha=0.55,
                edgecolor=DARK, linewidth=0.6, density=True, label=lbl)
        # GNB'nin öğrendiği Gaussian
        mu_g  = gnb.theta_[cls, i]
        sig_g = np.sqrt(gnb.var_[cls, i])
        xr = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xr, sp_stats.norm.pdf(xr, mu_g, sig_g),
                color=clr, lw=2, ls="--")
    ax.set_title(feat, fontsize=9, pad=5)
    ax.set_xlabel("z-skor", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y")

plt.tight_layout(); save("05_gnb_distributions.png")

# ════════════════════════════════════════════════════════════
#  D — TÜMSÜ MODEL KARŞILAŞTIRMASI
# ════════════════════════════════════════════════════════════
hdr("D — MODEL KARŞILAŞTIRMASI")

# Metrikleri topla
lr_test_proba = sk_lr.predict_proba(X_te)[:, 1]
lr_test_pred  = sk_lr.predict(X_te)
knn_test_pred2 = best_knn.predict(X_te)
knn_test_prob2 = best_knn.predict_proba(X_te)[:, 1]

models = {
    "Lojistik Reg.": (lr_test_pred,   lr_test_proba),
    f"KNN (K={best_k})": (knn_test_pred2, knn_test_prob2),
    "Gaussian NB":  (gnb_test_pred,   gnb_test_proba),
}

rows = []
for mname, (pred, proba) in models.items():
    cm = confusion_matrix(y_te, pred)
    tn, fp, fn, tp = cm.ravel()
    rows.append({
        "Model"     : mname,
        "Accuracy"  : accuracy_score(y_te, pred),
        "AUC-ROC"   : roc_auc_score(y_te, proba),
        "F1"        : f1_score(y_te, pred),
        "Precision" : tp/(tp+fp+1e-9),
        "Recall"    : tp/(tp+fn+1e-9),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    })

df_comp = pd.DataFrame(rows)
df_comp.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(f"\n{df_comp[['Model','Accuracy','AUC-ROC','F1','Precision','Recall']].to_string(index=False)}")

# ── GRAFİK 5: Model Karşılaştırma Dashboard ────────────────
fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 4 — Model Karşılaştırması: LR · KNN · GNB",
             fontsize=15, fontweight="bold", color=TXT)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

MODEL_COLORS = [C1, C2, C4]

# Panel 1 – Metrik bar grafikleri
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(CARD)
metrics_plot = ["Accuracy", "AUC-ROC", "F1", "Precision", "Recall"]
x = np.arange(len(metrics_plot)); w = 0.22
for i, (_, row) in enumerate(df_comp.iterrows()):
    ax1.bar(x + i*w, [row[m] for m in metrics_plot], w,
            color=MODEL_COLORS[i], label=row["Model"],
            edgecolor=DARK, linewidth=1, alpha=0.85)
ax1.set_xticks(x + w)
ax1.set_xticklabels(metrics_plot, rotation=20, ha="right", fontsize=8)
ax1.set_ylim(0, 1); ax1.set_ylabel("Skor"); ax1.legend(fontsize=8)
ax1.set_title("Test Seti Metrikleri", fontsize=11, fontweight="bold")
ax1.grid(axis="y")

# Panel 2 – ROC eğrileri
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(CARD)
ax2.plot([0,1],[0,1], color=TXT, lw=1, ls=":", alpha=.5)
for (mname, (pred, proba)), clr in zip(models.items(), MODEL_COLORS):
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    ax2.plot(fpr, tpr, color=clr, lw=2.2, label=f"{mname} ({auc:.3f})")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Eğrileri (Test)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=.3)

# Panel 3 – Karışıklık matrisleri yan yana
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(CARD)
ax3.axis("off")
for col_off, (mname, (pred, _)), clr in zip([0, 0.35, 0.70],
                                              models.items(), MODEL_COLORS):
    cm = confusion_matrix(y_te, pred)
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            bg  = clr if (r == c) else C3
            rect = plt.Rectangle((col_off + c*0.12, 0.55 - r*0.25),
                                  0.11, 0.22,
                                  facecolor=bg, edgecolor=DARK,
                                  transform=ax3.transAxes, clip_on=False,
                                  linewidth=1.5, alpha=0.75)
            ax3.add_patch(rect)
            ax3.text(col_off + c*0.12 + 0.055,
                     0.55 - r*0.25 + 0.11,
                     str(val), ha="center", va="center",
                     fontsize=14, fontweight="bold", color="white",
                     transform=ax3.transAxes)
    ax3.text(col_off + 0.055, 0.88, mname.split("(")[0],
             ha="center", fontsize=7.5, color=clr,
             transform=ax3.transAxes, fontweight="bold")

ax3.set_title("Karışıklık Matrisleri (Test)",
              fontsize=11, fontweight="bold", pad=10)

# Panel 4 – KNN K sweep (alt sol)
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(CARD)
ax4.plot(K_RANGE, knn_val_acc, color=C1, lw=2, marker="o", ms=4, label="Val")
ax4.plot(K_RANGE, knn_test_acc, color=C3, lw=1.8, marker="s", ms=4,
         ls="--", label="Test")
ax4.axvline(best_k, color=C5, lw=1.5, ls=":")
ax4.set_xlabel("K"); ax4.set_ylabel("Accuracy")
ax4.set_title(f"KNN – K Analizi (Best K={best_k})",
              fontsize=11, fontweight="bold")
ax4.set_xticks(list(K_RANGE)); ax4.legend(fontsize=8); ax4.grid(True)

# Panel 5 – GNB prior/posterior çizgisi
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(CARD)
feat_idx = FEAT.index("Glucose")
for cls, clr, lbl in [(0, C1, "Neg(0)"), (1, C3, "Pos(1)")]:
    vals = X_tr[y_tr == cls, feat_idx]
    mu_g  = gnb.theta_[cls, feat_idx]
    sig_g = np.sqrt(gnb.var_[cls, feat_idx])
    xr = np.linspace(-3, 4, 300)
    ax5.fill_between(xr, sp_stats.norm.pdf(xr, mu_g, sig_g),
                     alpha=0.3, color=clr)
    ax5.plot(xr, sp_stats.norm.pdf(xr, mu_g, sig_g),
             color=clr, lw=2, label=f"{lbl} μ={mu_g:.2f}")
ax5.set_xlabel("Glucose (z-skor)"); ax5.set_ylabel("f(x)")
ax5.set_title("GNB – Glucose Sınıf Dağılımları",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=8); ax5.grid(True, alpha=.3)

# Panel 6 – LR Z istatistikleri
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(CARD)
feat_df  = df_stats[df_stats["Parametre"] != "intercept"].copy()
bar_clrs = [C2 if s else C3 for s in feat_df["Anlamlı"]]
ax6.barh(feat_df["Parametre"], feat_df["Z-ist."],
         color=bar_clrs, edgecolor=DARK, linewidth=1, alpha=0.85)
ax6.axvline(1.96, color=C5, lw=1.5, ls="--", label="±1.96")
ax6.axvline(-1.96, color=C5, lw=1.5, ls="--")
ax6.axvline(0, color=TXT, lw=.8, alpha=.5)
ax6.set_xlabel("Z-istatistiği"); ax6.legend(fontsize=8)
ax6.set_title("LR – Z İstatistikleri (★=Anlamlı)",
              fontsize=11, fontweight="bold")
ax6.grid(axis="x")

save("06_model_comparison_dashboard.png")

# ── GRAFİK 6: Tüm modeller CM yan yana ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle("Karışıklık Matrisleri — Test Seti", fontsize=13,
             fontweight="bold", color=TXT)
for ax, (mname, (pred, _)), clr in zip(axes, models.items(), MODEL_COLORS):
    ax.set_facecolor(CARD)
    cm = confusion_matrix(y_te, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=2, linecolor=DARK,
                annot_kws={"size": 18, "weight": "bold"},
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"])
    acc = accuracy_score(y_te, pred)
    ax.set_title(f"{mname}\nAcc={acc:.4f}", fontsize=11,
                 fontweight="bold", color=clr)

plt.tight_layout(); save("07_confusion_matrices.png")

# ── SONUÇ ÖZET ─────────────────────────────────────────────
hdr("FAZ 4 — TAMAMLANDI ✅")
print(df_comp[["Model","Accuracy","AUC-ROC","F1"]].to_string(index=False))

# Anlamlı değişkenler
sig_list = df_stats[df_stats["Anlamlı"] & (df_stats["Parametre"] != "intercept")]["Parametre"].tolist()
print(f"\n  İstatistiksel olarak anlamlı değişkenler (p<0.05):")
for s in sig_list:
    row = df_stats[df_stats["Parametre"]==s].iloc[0]
    print(f"    ★ {s:<28} β={row['Katsayı']:.4f}  p={row['p-değeri']:.4f}")

print(f"\n  Grafikler (7 adet) → {OUT}/")
