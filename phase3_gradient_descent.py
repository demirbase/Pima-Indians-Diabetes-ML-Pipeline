"""
=============================================================
  FAZ 3: GRADYAN ALÇALMA (Gradient Descent)
  Lojistik Regresyon — El ile Uygulama
=============================================================
  Hedef: Outcome (Diyabet var/yok) ikili sınıflandırması
  Yöntem: Batch Gradient Descent (sıfırdan)
  Analiz:
    • Farklı öğrenme oranları karşılaştırması
    • Loss vs. İterasyon grafiği
    • Yakınsama / ıraksama analizi
    • Validation eğrileri
    • Sklearn karşılaştırması (doğrulama)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score,
                             roc_curve, log_loss)
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  KOYU TEMA
# ─────────────────────────────────────────────
DARK_BG    = "#0d1117"
CARD_BG    = "#161b22"
ACCENT1    = "#58a6ff"
ACCENT2    = "#3fb950"
ACCENT3    = "#ff7b72"
ACCENT4    = "#d2a8ff"
ACCENT5    = "#ffa657"
ACCENT6    = "#79c0ff"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"

LR_COLORS = {
    0.1    : "#ff7b72",   # kırmızı  – yüksek / tehlikeli
    0.01   : "#ffa657",   # turuncu  – orta-yüksek
    0.001  : "#3fb950",   # yeşil    – iyi
    0.0001 : "#58a6ff",   # mavi     – yavaş
}

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor":   GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor":  TEXT_COLOR, "xtick.color":     TEXT_COLOR,
    "ytick.color":      TEXT_COLOR, "text.color":      TEXT_COLOR,
    "grid.color":       GRID_COLOR, "grid.linestyle":  "--",
    "grid.alpha": 0.4,  "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COLOR, "font.family": "monospace",
    "figure.dpi": 120,
})

OUTPUT_DIR = "phase3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight", facecolor=DARK_BG, dpi=150)
    plt.close()
    print(f"  ✅  → {path}")

def section_header(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")


# ═══════════════════════════════════════════════════════════
#  ADIM 0 — VERİ YÜKLEME (Faz 1 ölçeklenmiş setler)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 0 — VERİ YÜKLEME")

train = pd.read_csv("phase1_outputs/train_scaled.csv")
val   = pd.read_csv("phase1_outputs/val_scaled.csv")
test  = pd.read_csv("phase1_outputs/test_scaled.csv")

TARGET   = "Outcome"
FEATURES = [c for c in train.columns if c != TARGET]

X_train = train[FEATURES].values;  y_train = train[TARGET].values
X_val   = val[FEATURES].values;    y_val   = val[TARGET].values
X_test  = test[FEATURES].values;   y_test  = test[TARGET].values

# Bias sütunu ekle
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train_b = add_bias(X_train)
X_val_b   = add_bias(X_val)
X_test_b  = add_bias(X_test)

n, p   = X_train_b.shape
n_feat = p - 1  # intercept hariç

print(f"\n  Özellik sayısı   : {n_feat}")
print(f"  Train boyutu     : {len(y_train)}")
print(f"  Val  / Test      : {len(y_val)} / {len(y_test)}")
print(f"  Pozitif oran     : Train={y_train.mean():.3f}  Val={y_val.mean():.3f}  Test={y_test.mean():.3f}")


# ═══════════════════════════════════════════════════════════
#  ADIM 1 — LOJİSTİK REGRESYON FONKSİYONLARI
# ═══════════════════════════════════════════════════════════
section_header("ADIM 1 — LOJİSTİK REGRESYON FONKSİYONLARI")

def sigmoid(z):
    """Numerik kararlı sigmoid: σ(z) = 1 / (1 + e^{-z})"""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    BCE Loss = -1/n · Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
    Clipping ile numerik kararlılık sağlanır.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict_proba(X_bias, weights):
    return sigmoid(X_bias @ weights)

def predict(X_bias, weights, threshold=0.5):
    return (predict_proba(X_bias, weights) >= threshold).astype(int)

def gradient(X_bias, y_true, weights):
    """
    ∇L = 1/n · Xᵀ(ŷ - y)
    Lojistik kayıp fonksiyonunun gradyanı.
    """
    y_pred = predict_proba(X_bias, weights)
    return (1 / len(y_true)) * (X_bias.T @ (y_pred - y_true))

print("  sigmoid, binary_cross_entropy, gradient fonksiyonları tanımlandı.")


# ═══════════════════════════════════════════════════════════
#  ADIM 2 — GRADIENT DESCENT ALGORİTMASI
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2 — GRADIENT DESCENT ALGORİTMASI")

def gradient_descent(X_train_b, y_train, X_val_b, y_val,
                     lr=0.01, n_iter=3000, tol=1e-8,
                     verbose=False, seed=42):
    """
    Batch Gradient Descent — Lojistik Regresyon
    w_{t+1} = w_t − α · ∇L(w_t)

    Parametreler:
        lr      : öğrenme oranı (α)
        n_iter  : maksimum iterasyon
        tol     : yakınsama toleransı (loss değişimi)
        verbose : ara çıktı

    Döndürür:
        weights, train_losses, val_losses, val_accs, converged_at
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, size=X_train_b.shape[1])

    train_losses = []
    val_losses   = []
    val_accs     = []
    grad_norms   = []
    converged_at = n_iter

    for i in range(n_iter):
        # İleri geçiş
        y_pred_train = predict_proba(X_train_b, weights)
        loss_train   = binary_cross_entropy(y_train, y_pred_train)

        # Gradyan
        grad = gradient(X_train_b, y_train, weights)
        grad_norm = np.linalg.norm(grad)

        # Divergence kontrolü
        if not np.isfinite(loss_train) or loss_train > 1e6:
            if verbose:
                print(f"    [iter {i}] IRAKSAMA! loss={loss_train:.4f}")
            # Kalan değerleri NaN ile doldur
            for _ in range(n_iter - i):
                train_losses.append(np.nan)
                val_losses.append(np.nan)
                val_accs.append(np.nan)
                grad_norms.append(np.nan)
            converged_at = -1  # ıraksamadı
            break

        # Güncelleme: w ← w − α·∇L
        weights = weights - lr * grad

        # Validation metrikleri
        y_pred_val  = predict_proba(X_val_b, weights)
        loss_val    = binary_cross_entropy(y_val, y_pred_val)
        acc_val     = accuracy_score(y_val, (y_pred_val >= 0.5).astype(int))

        train_losses.append(loss_train)
        val_losses.append(loss_val)
        val_accs.append(acc_val)
        grad_norms.append(grad_norm)

        # Yakınsama testi
        if i > 0 and abs(train_losses[-2] - train_losses[-1]) < tol:
            converged_at = i + 1
            if verbose:
                print(f"    Yakınsandı → iterasyon {converged_at}")
            break

    return (np.array(train_losses), np.array(val_losses),
            np.array(val_accs),    np.array(grad_norms),
            weights, converged_at)

print("  gradient_descent() fonksiyonu tanımlandı.")


# ═══════════════════════════════════════════════════════════
#  ADIM 3 — FARKLI ÖĞRENME ORANLARI
# ═══════════════════════════════════════════════════════════
section_header("ADIM 3 — ÖĞRENME ORANLARI KARŞILAŞTIRMASI")

LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
N_ITER         = 5000
results        = {}

print(f"\n  {'LR':>8}  {'Son Loss':>10}  {'Val Acc':>9}  {'Yakınsama':>12}")
print(f"  {'-'*47}")

for lr in LEARNING_RATES:
    tr_l, val_l, val_a, gnorms, w, conv = gradient_descent(
        X_train_b, y_train, X_val_b, y_val,
        lr=lr, n_iter=N_ITER, tol=1e-9, verbose=False
    )
    results[lr] = {
        "train_loss"   : tr_l,
        "val_loss"     : val_l,
        "val_acc"      : val_a,
        "grad_norms"   : gnorms,
        "weights"      : w,
        "converged_at" : conv,
    }
    # Son geçerli loss
    valid_tr = tr_l[~np.isnan(tr_l)]
    final_loss = valid_tr[-1] if len(valid_tr) > 0 else float("nan")
    valid_ac   = val_a[~np.isnan(val_a)]
    final_acc  = valid_ac[-1] if len(valid_ac) > 0 else float("nan")
    conv_str   = f"iter {conv}" if conv > 0 else "IRAKSADI ❌"

    print(f"  {lr:>8.4f}  {final_loss:>10.6f}  {final_acc:>9.4f}  {conv_str:>12}")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 1 — LOSS vs. İTERASYON (4 öğrenme oranı)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 1 — LOSS vs. İTERASYON")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 3 — Loss vs. İterasyon: Farklı Öğrenme Oranları",
             fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01)

lr_labels = {
    0.1:    "α = 0.1  (Çok Yüksek — Iraksama Riski)",
    0.01:   "α = 0.01 (Orta-Yüksek — İdeal Bölge)",
    0.001:  "α = 0.001 (İyi Performans — Stabil)",
    0.0001: "α = 0.0001 (Çok Yavaş — Yavaş Öğrenme)",
}

for ax, lr in zip(axes.flatten(), LEARNING_RATES):
    res  = results[lr]
    iters = np.arange(1, len(res["train_loss"]) + 1)
    color = LR_COLORS[lr]
    ax.set_facecolor(CARD_BG)

    valid_mask = ~np.isnan(res["train_loss"])
    if valid_mask.any():
        ax.plot(iters[valid_mask], res["train_loss"][valid_mask],
                color=color, linewidth=2.2, label="Train Loss", zorder=4)
        ax.plot(iters[valid_mask], res["val_loss"][valid_mask],
                color=ACCENT4, linewidth=1.8, linestyle="--",
                label="Val Loss", alpha=0.9, zorder=3)

        # Yakınsama noktası
        if res["converged_at"] > 0 and res["converged_at"] < N_ITER:
            conv_idx  = res["converged_at"] - 1
            conv_loss = res["train_loss"][conv_idx]
            ax.axvline(res["converged_at"], color=ACCENT2,
                       linewidth=1.5, linestyle=":", alpha=0.8, zorder=2)
            ax.scatter([res["converged_at"]], [conv_loss],
                       color=ACCENT2, s=80, zorder=5,
                       label=f"Yakınsama @ iter {res['converged_at']}")

        # İraksama uyarısı
        if res["converged_at"] == -1:
            ax.text(0.5, 0.5, "⚠️ IRAKSAMA\n(NaN/Inf)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=16, color=ACCENT3, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK_BG,
                              edgecolor=ACCENT3, alpha=0.9))
    else:
        ax.text(0.5, 0.5, "⚠️ IRAKSAMA\nTüm Değerler NaN",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=ACCENT3, fontweight="bold")

    ax.set_title(lr_labels[lr], fontsize=10, pad=10,
                 color=color, fontweight="bold")
    ax.set_xlabel("İterasyon", fontsize=9)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig("01_loss_vs_iteration.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 2 — TÜM LR KARŞILAŞTIRMASI (tek eksen, log)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 2 — KARŞILAŞTIRMALI LOSS GRAFİĞİ")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Tüm Öğrenme Oranları — Train & Val Loss Karşılaştırması",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

for ax, (loss_key, title) in zip(axes, [("train_loss", "Train Loss"),
                                          ("val_loss",   "Val Loss")]):
    ax.set_facecolor(CARD_BG)
    for lr in LEARNING_RATES:
        res  = results[lr]
        iters = np.arange(1, len(res[loss_key]) + 1)
        valid = ~np.isnan(res[loss_key])
        if valid.any():
            label = f"α={lr}"
            if res["converged_at"] == -1:
                label += " ⚠️ IRAKSADI"
            ax.plot(iters[valid], res[loss_key][valid],
                    color=LR_COLORS[lr], linewidth=2,
                    label=label, alpha=0.9)
    ax.set_xlabel("İterasyon", fontsize=10)
    ax.set_ylabel("BCE Loss", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Log ölçeği (eğer tüm değerler pozitif ise)
    try:
        ax.set_yscale("log")
    except Exception:
        pass

plt.tight_layout()
save_fig("02_all_lr_comparison.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 3 — VAL ACCURACY vs. İTERASYON
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 3 — VAL ACCURACY vs. İTERASYON")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
ax.set_title("Validation Accuracy — Tüm Öğrenme Oranları",
             fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)

for lr in LEARNING_RATES:
    res   = results[lr]
    iters = np.arange(1, len(res["val_acc"]) + 1)
    valid = ~np.isnan(res["val_acc"])
    if valid.any():
        final_acc = res["val_acc"][valid][-1]
        ax.plot(iters[valid], res["val_acc"][valid],
                color=LR_COLORS[lr], linewidth=2.2,
                label=f"α={lr}  (son acc={final_acc:.4f})")

ax.set_xlabel("İterasyon", fontsize=11)
ax.set_ylabel("Validation Accuracy", fontsize=11)
ax.set_ylim(0.5, 0.9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Sklearn referans doğruluğu (benchmark)
sk_model = LogisticRegression(max_iter=5000, C=1e10, random_state=42)
sk_model.fit(X_train, y_train)
sk_acc = accuracy_score(y_val, sk_model.predict(X_val))
ax.axhline(sk_acc, color=ACCENT5, linewidth=2, linestyle=":",
           label=f"sklearn referans = {sk_acc:.4f}")
ax.legend(fontsize=10)

save_fig("03_val_accuracy.png")


# ═══════════════════════════════════════════════════════════
#  ADIM 4 — OPTİMAL MODEL SEÇİMİ (lr=0.01)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4 — OPTİMAL MODEL (lr=0.01)")

BEST_LR    = 0.01
best_res   = results[BEST_LR]
best_w     = best_res["weights"]

y_proba_val  = predict_proba(X_val_b,  best_w)
y_proba_test = predict_proba(X_test_b, best_w)
y_pred_val   = (y_proba_val  >= 0.5).astype(int)
y_pred_test  = (y_proba_test >= 0.5).astype(int)

acc_val  = accuracy_score(y_val,  y_pred_val)
acc_test = accuracy_score(y_test, y_pred_test)
auc_val  = roc_auc_score(y_val,  y_proba_val)
auc_test = roc_auc_score(y_test, y_proba_test)
ll_val   = log_loss(y_val,  y_proba_val)
ll_test  = log_loss(y_test, y_proba_test)

print(f"\n  Optimal LR = {BEST_LR}")
print(f"\n  {'Metrik':<12}  {'Validation':>12}  {'Test':>12}")
print(f"  {'-'*38}")
print(f"  {'Accuracy':<12}  {acc_val:>12.4f}  {acc_test:>12.4f}")
print(f"  {'AUC-ROC':<12}  {auc_val:>12.4f}  {auc_test:>12.4f}")
print(f"  {'Log-Loss':<12}  {ll_val:>12.4f}  {ll_test:>12.4f}")

print(f"\n  Öğrenilen Ağırlıklar (intercept dahil):")
feat_names = ["intercept"] + FEATURES
for name, w in zip(feat_names, best_w):
    print(f"    {name:<30} {w:>10.6f}")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 4 — GRADYAN NORMU vs. İTERASYON
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 4 — GRADYAN NORMU vs. İTERASYON")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Gradyan Normu ‖∇L‖ — Yakınsama Kanıtı",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

for ax, lr in [(axes[0], 0.01), (axes[1], 0.001)]:
    ax.set_facecolor(CARD_BG)
    gn    = results[lr]["grad_norms"]
    iters = np.arange(1, len(gn) + 1)
    valid = ~np.isnan(gn)
    color = LR_COLORS[lr]
    ax.plot(iters[valid], gn[valid], color=color,
            linewidth=2, alpha=0.9)
    ax.fill_between(iters[valid], gn[valid], alpha=0.15, color=color)
    ax.axhline(1e-4, color=ACCENT3, linestyle="--",
               linewidth=1.5, label="Eşik: 1e-4")
    ax.set_yscale("log")
    ax.set_xlabel("İterasyon", fontsize=10)
    ax.set_ylabel("‖∇L‖ (log ölçeği)", fontsize=10)
    ax.set_title(f"α = {lr}", fontsize=11, color=color,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

save_fig("04_gradient_norm.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 5 — KARİŞİKLİK MATRİSİ (Optimal Model)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 5 — KARIŞİKLIK MATRİSİ")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(f"Karışıklık Matrisi — GD Modeli (α={BEST_LR})",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

for ax, (y_true, y_pred, title) in zip(axes, [
        (y_val,  y_pred_val,  "Validation Seti"),
        (y_test, y_pred_test, "Test Seti"),
]):
    ax.set_facecolor(CARD_BG)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=ax, linewidths=2, linecolor=DARK_BG,
                annot_kws={"size": 18, "weight": "bold", "color": TEXT_COLOR},
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["True: 0", "True: 1"])
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(colors=TEXT_COLOR)

    # Metrikleri köşeye yaz
    prec = cm[1,1] / (cm[0,1]+cm[1,1]+1e-9)
    rec  = cm[1,1] / (cm[1,0]+cm[1,1]+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    ax.text(1.02, 0.5,
            f"Acc  : {accuracy_score(y_true,y_pred):.3f}\n"
            f"Prec : {prec:.3f}\nRec  : {rec:.3f}\nF1   : {f1:.3f}",
            transform=ax.transAxes, fontsize=9,
            color=TEXT_COLOR, va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK_BG,
                      edgecolor=GRID_COLOR, alpha=0.9))

plt.tight_layout()
save_fig("05_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 6 — ROC EĞRİSİ (GD vs. Sklearn)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 6 — ROC EĞRİSİ")

sk_proba_test = sk_model.predict_proba(X_test)[:, 1]
fpr_gd,  tpr_gd,  _ = roc_curve(y_test, y_proba_test)
fpr_sk,  tpr_sk,  _ = roc_curve(y_test, sk_proba_test)
auc_gd  = roc_auc_score(y_test, y_proba_test)
auc_sk_ = roc_auc_score(y_test, sk_proba_test)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
ax.plot(fpr_gd, tpr_gd, color=ACCENT2, linewidth=2.5,
        label=f"GD Modeli    (AUC={auc_gd:.4f})", zorder=4)
ax.plot(fpr_sk, tpr_sk, color=ACCENT5, linewidth=2,
        linestyle="--", label=f"sklearn Ref. (AUC={auc_sk_:.4f})", zorder=3)
ax.plot([0,1],[0,1], color=TEXT_COLOR, linewidth=1,
        linestyle=":", alpha=0.5, label="Rastgele (AUC=0.5)")
ax.fill_between(fpr_gd, tpr_gd, alpha=0.12, color=ACCENT2)
ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=11)
ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=11)
ax.set_title(f"ROC Eğrisi — GD vs. sklearn (Test Seti)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.grid(True, alpha=0.3)
save_fig("06_roc_curve.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 7 — ÖZELLIK AĞIRLIKLARI (sigmoid katsayıları)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 7 — ÖZELLIK AĞIRLIKLARI")

weights_no_bias = best_w[1:]    # intercept hariç
order = np.argsort(np.abs(weights_no_bias))[::-1]
feat_sorted = [FEATURES[i] for i in order]
w_sorted    = weights_no_bias[order]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
bar_colors = [ACCENT2 if w > 0 else ACCENT3 for w in w_sorted]
bars = ax.barh(range(len(feat_sorted)), w_sorted,
               color=bar_colors, edgecolor=DARK_BG,
               linewidth=1.2, alpha=0.88)
ax.axvline(0, color=TEXT_COLOR, linewidth=1, linestyle="--", alpha=0.6)
ax.set_yticks(range(len(feat_sorted)))
ax.set_yticklabels(feat_sorted, fontsize=10)
ax.set_xlabel("Ağırlık Değeri (z-skorda)", fontsize=11)
ax.set_title(f"GD Modeli Özellik Ağırlıkları (α={BEST_LR})",
             fontsize=13, fontweight="bold", pad=12)
for bar, val in zip(bars, w_sorted):
    ax.text(val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5,
            ha="left" if val >= 0 else "right", color=TEXT_COLOR)
ax.grid(axis="x", alpha=0.3)
save_fig("07_feature_weights.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 8 — ÖĞRENME ORANI ANALİZ DASHBOARD
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 8 — ÖĞRENME ORANI ANALİZ DASHBOARD")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 3 — Gradient Descent: Öğrenme Oranı Analizi",
             fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.00)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.38)

# Panel 1-4: Her LR için loss eğrisi (mini)
for col, lr in enumerate(LEARNING_RATES):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(CARD_BG)
    res   = results[lr]
    iters = np.arange(1, len(res["train_loss"])+1)
    valid = ~np.isnan(res["train_loss"])
    color = LR_COLORS[lr]
    if valid.any():
        ax.plot(iters[valid], res["train_loss"][valid],
                color=color, linewidth=2, label="Train")
        ax.plot(iters[valid], res["val_loss"][valid],
                color=ACCENT4, linewidth=1.5,
                linestyle="--", label="Val")
    if res["converged_at"] == -1:
        ax.text(0.5, 0.5, "IRAKSAMA", transform=ax.transAxes,
                ha="center", va="center", fontsize=12,
                color=ACCENT3, fontweight="bold")
    ax.set_title(f"α={lr}", fontsize=10, color=color, fontweight="bold")
    ax.set_xlabel("İterasyon", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel 5: Val accuracy karşılaştırması
ax5 = fig.add_subplot(gs[1, 0:2])
ax5.set_facecolor(CARD_BG)
for lr in LEARNING_RATES:
    res   = results[lr]
    iters = np.arange(1, len(res["val_acc"])+1)
    valid = ~np.isnan(res["val_acc"])
    if valid.any():
        final = res["val_acc"][valid][-1]
        ax5.plot(iters[valid], res["val_acc"][valid],
                 color=LR_COLORS[lr], linewidth=2,
                 label=f"α={lr} → {final:.3f}")
ax5.axhline(sk_acc, color=ACCENT5, linewidth=1.5,
            linestyle=":", label=f"sklearn={sk_acc:.3f}")
ax5.set_xlabel("İterasyon", fontsize=9); ax5.set_ylabel("Val Accuracy", fontsize=9)
ax5.set_title("Val Accuracy Eğrileri", fontsize=10, fontweight="bold")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# Panel 6: Final val acc bar
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD_BG)
final_accs = []
for lr in LEARNING_RATES:
    va = results[lr]["val_acc"][~np.isnan(results[lr]["val_acc"])]
    final_accs.append(va[-1] if len(va) > 0 else 0)
bars = ax6.bar([str(lr) for lr in LEARNING_RATES], final_accs,
               color=[LR_COLORS[lr] for lr in LEARNING_RATES],
               edgecolor=DARK_BG, linewidth=1.2)
ax6.axhline(sk_acc, color=ACCENT5, linewidth=1.5,
            linestyle=":", label=f"sklearn={sk_acc:.3f}")
for bar, v in zip(bars, final_accs):
    ax6.text(bar.get_x()+bar.get_width()/2, v+0.003,
             f"{v:.3f}", ha="center", fontsize=8, color=TEXT_COLOR)
ax6.set_xlabel("Öğrenme Oranı (α)", fontsize=9)
ax6.set_ylabel("Val Accuracy", fontsize=9)
ax6.set_ylim(0.6, 0.85)
ax6.set_title("Son Val Accuracy", fontsize=10, fontweight="bold")
ax6.legend(fontsize=8); ax6.grid(axis="y")

# Panel 7: Ağırlık büyüklükleri
ax7 = fig.add_subplot(gs[1, 3])
ax7.set_facecolor(CARD_BG)
ax7.barh(range(len(feat_sorted)), np.abs(w_sorted),
         color=[ACCENT2 if w > 0 else ACCENT3 for w in w_sorted],
         edgecolor=DARK_BG, linewidth=1, alpha=0.85)
ax7.set_yticks(range(len(feat_sorted)))
ax7.set_yticklabels(feat_sorted, fontsize=8)
ax7.set_xlabel("|Ağırlık|", fontsize=9)
ax7.set_title(f"Özellik Önemi\n(α={BEST_LR})", fontsize=10, fontweight="bold")
ax7.grid(axis="x", alpha=0.3)

save_fig("08_lr_analysis_dashboard.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 9 — SIGMOID FONKSİYONU VE GD ADIMI VİZÜELİ
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 9 — SİGMOID ve GD ADIMI")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Matematiksel Arka Plan — Sigmoid · BCE Loss · GD Güncellemesi",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

# Panel 1: Sigmoid
ax = axes[0]
ax.set_facecolor(CARD_BG)
z  = np.linspace(-6, 6, 300)
ax.plot(z, sigmoid(z), color=ACCENT1, linewidth=2.5, label="σ(z)")
ax.axhline(0.5, color=ACCENT5, linewidth=1.5, linestyle="--",
           label="Karar sınırı (0.5)")
ax.axvline(0,   color=TEXT_COLOR, linewidth=1, alpha=0.4)
ax.fill_between(z[z>=0], sigmoid(z[z>=0]), 0.5, alpha=0.12, color=ACCENT2,
                label="Pozitif bölge")
ax.fill_between(z[z<=0], sigmoid(z[z<=0]), 0.5, alpha=0.12, color=ACCENT3,
                label="Negatif bölge")
ax.set_xlabel("z = wᵀx", fontsize=10); ax.set_ylabel("σ(z)", fontsize=10)
ax.set_title("Sigmoid Fonksiyonu\nσ(z) = 1 / (1 + e⁻ᶻ)", fontsize=11,
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: BCE Loss vs p
ax = axes[1]
ax.set_facecolor(CARD_BG)
p_vals = np.linspace(0.001, 0.999, 300)
ax.plot(p_vals, -np.log(p_vals),      color=ACCENT2, linewidth=2.2,
        label="y=1: −log(p̂)")
ax.plot(p_vals, -np.log(1 - p_vals),  color=ACCENT3, linewidth=2.2,
        label="y=0: −log(1−p̂)")
ax.set_ylim(0, 5); ax.set_xlim(0, 1)
ax.set_xlabel("Tahmin edilen olasılık ŷ", fontsize=10)
ax.set_ylabel("Kayıp", fontsize=10)
ax.set_title("Binary Cross-Entropy\nL = −[y log(ŷ) + (1−y)log(1−ŷ)]",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: GD adımı (1D loss eğrisi)
ax = axes[2]
ax.set_facecolor(CARD_BG)
w_vals  = np.linspace(-3, 3, 300)
# Tek ağırlık için toy loss
toy_loss = 0.5 * w_vals**2 + 0.3 * np.sin(2*w_vals) + 0.5
w_curr   = 2.2
steps    = []
w_step   = w_curr
for _ in range(8):
    steps.append(w_step)
    grad_val = w_step + 0.3*2*np.cos(2*w_step)
    w_step  -= 0.4 * grad_val

ax.plot(w_vals, toy_loss, color=ACCENT1, linewidth=2.5, label="Loss(w)")
for i, ws in enumerate(steps[:-1]):
    loss_s  = 0.5*ws**2 + 0.3*np.sin(2*ws) + 0.5
    loss_n  = 0.5*steps[i+1]**2 + 0.3*np.sin(2*steps[i+1]) + 0.5
    alpha_  = 0.3 + 0.7*(i/len(steps))
    ax.annotate("", xy=(steps[i+1], loss_n), xytext=(ws, loss_s),
                arrowprops=dict(arrowstyle="->", color=ACCENT2,
                                lw=1.8, alpha=alpha_))
    ax.scatter([ws], [loss_s], color=ACCENT2, s=40, zorder=4, alpha=alpha_)

ax.scatter([steps[-1]], [0.5*steps[-1]**2 + 0.3*np.sin(2*steps[-1]) + 0.5],
           color=ACCENT5, s=100, zorder=5, label="Son adım")
ax.set_xlabel("w (ağırlık)", fontsize=10)
ax.set_ylabel("Loss(w)", fontsize=10)
ax.set_title("GD Adımları\nw ← w − α·∇L(w)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig("09_theory_visualization.png")


# ═══════════════════════════════════════════════════════════
#  SONUÇLARI KAYDET
# ═══════════════════════════════════════════════════════════
section_header("SONUÇLARI KAYDET")

# Katsayı tablosu
df_weights = pd.DataFrame({
    "Özellik"   : feat_names,
    "Ağırlık_GD": best_w,
    "Ağırlık_SK": np.concatenate([[sk_model.intercept_[0]],
                                    sk_model.coef_[0]]),
})
df_weights.to_csv(os.path.join(OUTPUT_DIR, "weights_comparison.csv"), index=False)

# LR özet tablosu
rows = []
for lr in LEARNING_RATES:
    va = results[lr]["val_acc"][~np.isnan(results[lr]["val_acc"])]
    tl = results[lr]["train_loss"][~np.isnan(results[lr]["train_loss"])]
    rows.append({
        "Learning Rate"  : lr,
        "Son Train Loss" : tl[-1] if len(tl) > 0 else np.nan,
        "Son Val Acc"    : va[-1] if len(va) > 0 else np.nan,
        "Yakınsama İter" : results[lr]["converged_at"],
        "Durum"          : "IRAKSAMA" if results[lr]["converged_at"] == -1 else "Yakınsadı",
    })
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "lr_summary.csv"), index=False)
print(f"\n  CSV'ler kaydedildi → {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════
#  SONUÇ
# ═══════════════════════════════════════════════════════════
section_header("FAZ 3 — TAMAMLANDI ✅")

best_va = results[BEST_LR]["val_acc"]
best_va_final = best_va[~np.isnan(best_va)][-1]

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                    FAZ 3 ÖZET                               │
  ├──────────────────────────────────────────────────────────────┤
  │  Hedef              : Outcome (Diyabet sınıflandırması)     │
  │  Yöntem             : Batch Gradient Descent                │
  │  İterasyon          : {N_ITER}                              │
  │  Loss Fonksiyonu    : Binary Cross-Entropy                  │
  ├──────────────────────────────────────────────────────────────┤
  │  ÖĞRENME ORANI ANALİZİ                                      │
  │  α=0.1    → Iraksama riski (loss zıplar)                    │
  │  α=0.01   → ✅ OPTİMAL  (Acc={best_va_final:.4f})           │
  │  α=0.001  → Yakınsıyor ama yavaş                            │
  │  α=0.0001 → {N_ITER} iterasyonda tam yakınsamıyor            │
  ├──────────────────────────────────────────────────────────────┤
  │  OPTİMAL MODEL (α=0.01)                                     │
  │  Val  Acc  : {acc_val:.4f}  AUC: {auc_val:.4f}               │
  │  Test Acc  : {acc_test:.4f}  AUC: {auc_test:.4f}              │
  │  sklearn   : Val Acc = {sk_acc:.4f} (referans)               │
  ├──────────────────────────────────────────────────────────────┤
  │  Kaydedilen Dosyalar: {OUTPUT_DIR}/                          │
  │    • weights_comparison.csv  • lr_summary.csv               │
  │  Grafikler (9 adet) : {OUTPUT_DIR}/*.png                    │
  └──────────────────────────────────────────────────────────────┘
""")
