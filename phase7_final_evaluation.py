"""
=============================================================
  FAZ 7: BAŞARI ÖLÇÜMÜ VE FİNAL MODEL
  • ROC Eğrisi (tüm modeller — aynı grafikte)
  • AUC Skoru + Precision-Recall Eğrisi
  • Regresyon MSE / RMSE / MAE / R² raporu
  • Final Model Seçimi + Özet Dashboard
=============================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model   import LogisticRegression, LinearRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.metrics        import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy import stats as sp_stats
import warnings, os

warnings.filterwarnings("ignore")

DARK="#0d1117"; CARD="#161b22"; GRID="#21262d"
C1="#58a6ff"; C2="#3fb950"; C3="#ff7b72"
C4="#d2a8ff"; C5="#ffa657"; C6="#79c0ff"; TXT="#c9d1d9"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT,
    "axes.titlecolor": TXT, "xtick.color": TXT,
    "ytick.color": TXT, "text.color": TXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": .4,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "font.family": "monospace", "figure.dpi": 120,
})

OUT = "phase7_outputs"; os.makedirs(OUT, exist_ok=True)

def save(fname):
    plt.savefig(f"{OUT}/{fname}", bbox_inches="tight", facecolor=DARK, dpi=150)
    plt.close(); print(f"  ✅  → {OUT}/{fname}")

def hdr(t): print(f"\n{'═'*62}\n  {t}\n{'═'*62}")

# ── VERİ ───────────────────────────────────────────────────
hdr("VERİ YÜKLEME")
train = pd.read_csv("phase1_outputs/train_scaled.csv")
val   = pd.read_csv("phase1_outputs/val_scaled.csv")
test  = pd.read_csv("phase1_outputs/test_scaled.csv")

FEAT = [c for c in train.columns if c != "Outcome"]

X_tr  = train[FEAT].values;  y_tr = train["Outcome"].values
X_va  = val[FEAT].values;    y_va = val["Outcome"].values
X_te  = test[FEAT].values;   y_te = test["Outcome"].values

# Tüm klasif. verisi (CV için)
X_all = np.vstack([X_tr, X_va, X_te])
y_all = np.concatenate([y_tr, y_va, y_te])

# Regresyon (BMI tahmini) — Faz 2'den
BMI_FEAT = [c for c in FEAT if c != "BMI" and c != "Outcome"]
# Ölçeklenmiş train'de Outcome yok, BMI var → BMI bağımlı değişken
REG_FEAT = [f for f in FEAT if f != "BMI"]
X_tr_r   = train[REG_FEAT].values;  y_tr_r = train["BMI"].values
X_va_r   = val[REG_FEAT].values;    y_va_r = val["BMI"].values
X_te_r   = test[REG_FEAT].values;   y_te_r = test["BMI"].values

print(f"  Sınıflandırma: {len(y_all)} toplam | Test={len(y_te)}")
print(f"  Regresyon BMI: Train={len(y_tr_r)} Test={len(y_te_r)}")

# ════════════════════════════════════════════════════════════
#  A — MODELLERİ EĞİT
# ════════════════════════════════════════════════════════════
hdr("A — MODELLERİ EĞİT")

BEST_K = 26

models = {
    "Lojistik Reg." : LogisticRegression(max_iter=5000, C=1e6, random_state=42),
    f"KNN (K={BEST_K})"  : KNeighborsClassifier(n_neighbors=BEST_K, metric="euclidean"),
    "Gaussian NB"   : GaussianNB(),
}
for m in models.values():
    m.fit(X_tr, y_tr)

# 10-Fold CV olasılıkları (daha güvenilir ROC için)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_probas = {}
for mname, model in models.items():
    cv_probas[mname] = cross_val_predict(
        model, X_all, y_all, cv=skf,
        method="predict_proba")[:, 1]

# Test seti tahminleri
test_probas = {m: models[m].predict_proba(X_te)[:, 1] for m in models}
test_preds  = {m: models[m].predict(X_te) for m in models}

# Regresyon modeli
lr_reg = LinearRegression()
lr_reg.fit(X_tr_r, y_tr_r)
y_pred_reg_va = lr_reg.predict(X_va_r)
y_pred_reg_te = lr_reg.predict(X_te_r)

# ════════════════════════════════════════════════════════════
#  B — SINIFLANDIRMA METRİKLERİ
# ════════════════════════════════════════════════════════════
hdr("B — SINIFLANDIRMA METRİKLERİ (Test Seti)")

MODEL_COLS = [C1, C2, C4]
clf_rows = []
for mname in models:
    pred  = test_preds[mname]
    proba = test_probas[mname]
    cm    = confusion_matrix(y_te, pred)
    tn,fp,fn,tp = cm.ravel()
    clf_rows.append({
        "Model"     : mname,
        "Accuracy"  : accuracy_score(y_te, pred),
        "AUC-ROC"   : roc_auc_score(y_te, proba),
        "F1"        : f1_score(y_te, pred),
        "Precision" : tp/(tp+fp+1e-9),
        "Recall"    : tp/(tp+fn+1e-9),
        "Specificity": tn/(tn+fp+1e-9),
        "TP":tp,"TN":tn,"FP":fp,"FN":fn,
    })

df_clf = pd.DataFrame(clf_rows)
df_clf.to_csv(f"{OUT}/classification_metrics.csv", index=False)

print(f"\n  {'Model':<18} {'Acc':>7}  {'AUC':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
print(f"  {'-'*58}")
for _, r in df_clf.iterrows():
    print(f"  {r['Model']:<18} {r['Accuracy']:>7.4f}  {r['AUC-ROC']:>7.4f}  "
          f"{r['F1']:>7.4f}  {r['Precision']:>7.4f}  {r['Recall']:>7.4f}")

# ════════════════════════════════════════════════════════════
#  GRAFİK 1 — ROC EĞRİLERİ (Ana Grafik)
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 1 — ROC EĞRİLERİ")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 7 — ROC Eğrisi: Tüm Modeller (Test Seti & 10-Fold CV)",
             fontsize=14, fontweight="bold", color=TXT)

for ax, (proba_dict, title, src) in zip(axes, [
    (test_probas,  "Test Seti ROC Eğrisi",       y_te),
    (cv_probas,    "10-Fold CV ROC Eğrisi",     y_all),
]):
    ax.set_facecolor(CARD)
    ax.plot([0,1],[0,1], color=TXT, lw=1.2, ls=":", alpha=0.5,
            label="Rastgele Sınıflandırıcı (AUC=0.50)")
    ax.fill_between([0,1],[0,1], alpha=0.04, color=TXT)

    for (mname, proba), clr in zip(proba_dict.items(), MODEL_COLS):
        fpr, tpr, thrsh = roc_curve(src, proba)
        auc = roc_auc_score(src, proba)
        ax.plot(fpr, tpr, color=clr, lw=2.5,
                label=f"{mname}  AUC = {auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.05, color=clr)

    # En iyi model vurgula
    aucs  = {m: roc_auc_score(src, p) for m, p in proba_dict.items()}
    best  = max(aucs, key=aucs.get)
    fpr_, tpr_, _ = roc_curve(src, proba_dict[best])
    ax.fill_between(fpr_, tpr_, alpha=0.12,
                    color=[c for (m,c) in zip(models, MODEL_COLS) if m==best][0])

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR = 1 - Ozgüllük)", fontsize=10)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR = Duyarlılık)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # AUC anotasyon kutusu
    txt_lines = [f"{'─'*28}"]
    for m, c in zip(proba_dict, MODEL_COLS):
        txt_lines.append(f"  {m}: {aucs[m]:.4f}")
    txt_lines.append(f"{'─'*28}")
    ax.text(0.54, 0.15, "\n".join(txt_lines),
            transform=ax.transAxes, fontsize=7.5,
            color=TXT, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=DARK,
                      ec=GRID, alpha=0.9))

plt.tight_layout(); save("01_roc_curves.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 2 — PRECISION-RECALL EĞRİSİ
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 2 — PRECISION-RECALL EĞRİSİ")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("Precision-Recall Eğrisi — Sınıf Dengesizliği Durumunda Kritik",
             fontsize=13, fontweight="bold", color=TXT)

baseline = y_te.mean()
for ax, (proba_dict, src) in zip(axes, [
    (test_probas, y_te),
    (cv_probas,   y_all),
]):
    ax.set_facecolor(CARD)
    ax.axhline(src.mean(), color=TXT, lw=1.2, ls=":",
               alpha=0.6, label=f"Baseline (Prev.={src.mean():.3f})")
    for (mname, proba), clr in zip(proba_dict.items(), MODEL_COLS):
        prec, rec, _ = precision_recall_curve(src, proba)
        ap = average_precision_score(src, proba)
        ax.plot(rec, prec, color=clr, lw=2.2,
                label=f"{mname}  AP={ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.06, color=clr)
    ax.set_xlabel("Recall (Duyarlılık)", fontsize=10)
    ax.set_ylabel("Precision (Kesinlik)", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Test Seti" if proba_dict is test_probas else "10-Fold CV",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)

plt.tight_layout(); save("02_precision_recall.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 3 — AUC KARŞILAŞTIRMA BAR
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 3 — AUC KARŞILAŞTIRMA")

cv_aucs  = {m: roc_auc_score(y_all, cv_probas[m]) for m in models}
test_aucs = {m: roc_auc_score(y_te, test_probas[m]) for m in models}

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
x = np.arange(3); w = 0.30

b1 = ax.bar(x-w/2, [cv_aucs[m]   for m in models], w,
            color=MODEL_COLS, edgecolor=DARK, alpha=0.85, label="10-Fold CV AUC")
b2 = ax.bar(x+w/2, [test_aucs[m] for m in models], w,
            color=MODEL_COLS, edgecolor=DARK, alpha=0.55, hatch="///",
            label="Test AUC")
for bar_, v_ in [(bar, v) for bars, vals in [(b1, list(cv_aucs.values())),
                                               (b2, list(test_aucs.values()))]
                 for bar, v in zip(bars, vals)]:
    ax.text(bar_.get_x()+bar_.get_width()/2, v_+0.005,
            f"{v_:.4f}", ha="center", fontsize=9, color=TXT, fontweight="bold")

ax.axhline(0.5, color=C3, lw=1.5, ls="--", label="Rastgele (AUC=0.5)")
ax.axhline(0.8, color=C2, lw=1.2, ls=":",  label="İyi Eşik (AUC=0.8)")
ax.set_xticks(x); ax.set_xticklabels(list(models.keys()), fontsize=10)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_ylim(0.4, 1.0)
ax.set_title("AUC-ROC: 10-Fold CV vs. Test Seti",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9); ax.grid(axis="y")
plt.tight_layout(); save("03_auc_comparison.png")

# ════════════════════════════════════════════════════════════
#  C — REGRESYON METRİKLERİ (BMI Tahmini)
# ════════════════════════════════════════════════════════════
hdr("C — REGRESYON METRİKLERİ (BMI)")

def reg_metrics(y_true, y_pred, label):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"Set":label,"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2}

reg_rows = [
    reg_metrics(y_va_r, y_pred_reg_va, "Validation"),
    reg_metrics(y_te_r, y_pred_reg_te, "Test"),
]
df_reg = pd.DataFrame(reg_rows)
df_reg.to_csv(f"{OUT}/regression_metrics.csv", index=False)

print(f"\n  {'Set':<12}  {'MSE':>9}  {'RMSE':>8}  {'MAE':>8}  {'R²':>7}")
print(f"  {'-'*50}")
for _, r in df_reg.iterrows():
    print(f"  {r['Set']:<12}  {r['MSE']:>9.4f}  {r['RMSE']:>8.4f}  "
          f"{r['MAE']:>8.4f}  {r['R2']:>7.4f}")

# ── GRAFİK 4: Regresyon Sonuçları ──────────────────────────
hdr("GRAFİK 4 — REGRESYON SONUÇLARI")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("Lineer Regresyon — BMI Tahmin Performansı (Faz 2)",
             fontsize=13, fontweight="bold", color=TXT)

for ax, (y_true, y_pred, label, clr) in zip(axes, [
    (y_va_r, y_pred_reg_va, "Validation", C1),
    (y_te_r, y_pred_reg_te, "Test",       C2),
    (np.concatenate([y_va_r,y_te_r]),
     np.concatenate([y_pred_reg_va,y_pred_reg_te]),
     "Validation + Test", C4),
]):
    ax.set_facecolor(CARD)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, color=clr, alpha=0.55, s=20, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min())-1,
            max(y_true.max(), y_pred.max())+1]
    ax.plot(lims, lims, "--", color=C5, lw=2, label="Mükemmel tahmin (y=x)")
    # Regresyon çizgisi (Pred ~ True)
    slope, intercept, *_ = sp_stats.linregress(y_true, y_pred)
    xr = np.linspace(lims[0], lims[1], 100)
    ax.plot(xr, slope*xr+intercept, color=TXT, lw=1.5, alpha=0.7)

    ax.set_xlabel("Gerçek BMI", fontsize=10); ax.set_ylabel("Tahmin BMI", fontsize=10)
    ax.set_title(f"{label}\nMSE={mse:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}",
                 fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(); save("04_regression_results.png")

# ── GRAFİK 5: Kalıntı Analizi ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("Lineer Regresyon — Kalıntı Analizi (Test Seti)",
             fontsize=13, fontweight="bold", color=TXT)

residuals = y_te_r - y_pred_reg_te

ax = axes[0]; ax.set_facecolor(CARD)
ax.scatter(y_pred_reg_te, residuals, color=C1, alpha=0.55, s=20)
ax.axhline(0, color=C5, lw=2, ls="--")
ax.set_xlabel("Tahmin", fontsize=10); ax.set_ylabel("Kalıntı", fontsize=10)
ax.set_title("Kalıntı vs. Tahmin", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)

ax = axes[1]; ax.set_facecolor(CARD)
ax.hist(residuals, bins=30, color=C4, edgecolor=DARK, alpha=0.85, density=True)
xr = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xr, sp_stats.norm.pdf(xr, residuals.mean(), residuals.std()),
        color=C5, lw=2.5, label="Normal fit")
ax.set_xlabel("Kalıntı", fontsize=10); ax.set_ylabel("Yoğunluk", fontsize=10)
ax.set_title("Kalıntı Dağılımı", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y")

ax = axes[2]; ax.set_facecolor(CARD)
(osm, osr), (slope, intercept, r) = sp_stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, color=C2, alpha=0.65, s=20)
ax.plot(osm, slope*np.array(osm)+intercept, color=C5, lw=2)
ax.set_xlabel("Teorik Kantiller", fontsize=10)
ax.set_ylabel("Örnek Kantilleri", fontsize=10)
ax.set_title(f"Q-Q Plot (R²={r**2:.4f})", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout(); save("05_residual_analysis.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 6 — FİNAL MODEL KARŞILAŞTIRMA TABLOSU
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 6 — FİNAL MODEL TABLOSU")

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 7 — Final Model Performans Tablosu",
             fontsize=15, fontweight="bold", color=TXT)

# Panel 1: ROC yan yana (mini)
ax1 = axes[0, 0]; ax1.set_facecolor(CARD)
ax1.plot([0,1],[0,1], color=TXT, lw=1, ls=":", alpha=0.4)
for (mname, proba), clr in zip(test_probas.items(), MODEL_COLS):
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    ax1.plot(fpr, tpr, color=clr, lw=2.2, label=f"{mname} ({auc:.3f})")
    ax1.fill_between(fpr, tpr, alpha=0.07, color=clr)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC — Test Seti", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, loc="lower right"); ax1.grid(True, alpha=0.3)

# Panel 2: AUC Bar
ax2 = axes[0, 1]; ax2.set_facecolor(CARD)
aucs_sorted = sorted(test_aucs.items(), key=lambda x: x[1], reverse=True)
names_, aucs_ = zip(*aucs_sorted)
bar_clrs = [MODEL_COLS[list(models.keys()).index(n)] for n in names_]
bars = ax2.bar(range(3), aucs_, color=bar_clrs, edgecolor=DARK, alpha=0.85)
ax2.axhline(0.5, color=C3, lw=1.5, ls="--", label="Rastgele")
ax2.axhline(0.8, color=C2, lw=1.2, ls=":",  label="AUC=0.8")
ax2.set_xticks(range(3)); ax2.set_xticklabels(names_, fontsize=8, rotation=12)
ax2.set_ylabel("AUC-ROC"); ax2.set_ylim(0.4, 1.0)
ax2.set_title("AUC Olarak Sıralanmış", fontsize=11, fontweight="bold")
for bar, v in zip(bars, aucs_):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.008,
             f"{v:.4f}", ha="center", fontsize=9, color=TXT, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(axis="y")

# Panel 3: Radar Chart (örümcek ağı)
ax3 = axes[0, 2]; ax3.remove()
ax3 = fig.add_subplot(2, 3, 3, polar=True)
ax3.set_facecolor(CARD)
radar_metrics = ["Accuracy", "AUC-ROC", "F1", "Precision", "Recall"]
N = len(radar_metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_metrics, fontsize=8, color=TXT)
ax3.set_ylim(0, 1)
for (_, row), clr in zip(df_clf.iterrows(), MODEL_COLS):
    vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
    ax3.plot(angles, vals, color=clr, lw=2, label=row["Model"])
    ax3.fill(angles, vals, color=clr, alpha=0.12)
ax3.legend(fontsize=7, loc="upper right",
           bbox_to_anchor=(1.35, 1.1))
ax3.set_title("Radar Chart\n(Test Seti)", fontsize=10, fontweight="bold",
              color=TXT, pad=20)
ax3.tick_params(colors=TXT)
ax3.spines["polar"].set_color(GRID)
ax3.grid(color=GRID, alpha=0.5)

# Panel 4: Metrik tablo (text)
ax4 = axes[1, 0]; ax4.set_facecolor(CARD); ax4.axis("off")
col_labels = ["Model", "Acc", "AUC", "F1", "Prec", "Recall"]
cell_data  = [[r["Model"], f"{r['Accuracy']:.4f}", f"{r['AUC-ROC']:.4f}",
               f"{r['F1']:.4f}", f"{r['Precision']:.4f}", f"{r['Recall']:.4f}"]
              for _, r in df_clf.iterrows()]
tbl = ax4.table(cellText=cell_data, colLabels=col_labels,
                cellLoc="center", loc="center",
                bbox=[0.0, 0.05, 1.0, 0.9])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor(DARK if row == 0 else CARD)
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TXT, fontweight="bold" if row==0 else "normal")
ax4.set_title("Sınıflandırma Metrikleri (Test)",
              fontsize=10, fontweight="bold", pad=12)

# Panel 5: Regresyon tablo
ax5 = axes[1, 1]; ax5.set_facecolor(CARD); ax5.axis("off")
reg_cols  = ["Set", "MSE", "RMSE", "MAE", "R²"]
reg_cells = [[r["Set"], f"{r['MSE']:.4f}", f"{r['RMSE']:.4f}",
              f"{r['MAE']:.4f}", f"{r['R2']:.4f}"]
             for _, r in df_reg.iterrows()]
tbl2 = ax5.table(cellText=reg_cells, colLabels=reg_cols,
                 cellLoc="center", loc="center",
                 bbox=[0.05, 0.3, 0.9, 0.5])
tbl2.auto_set_font_size(False); tbl2.set_fontsize(10)
for (row, col), cell in tbl2.get_celld().items():
    cell.set_facecolor(DARK if row == 0 else CARD)
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TXT, fontweight="bold" if row==0 else "normal")
ax5.set_title("Regresyon Metrikleri — BMI Tahmini\n(Lineer Regresyon / EKK)",
              fontsize=10, fontweight="bold", pad=12)

# Panel 6: Sonuç yorumu
ax6 = axes[1, 2]; ax6.set_facecolor(DARK); ax6.axis("off")
best_clf = df_clf.loc[df_clf["AUC-ROC"].idxmax(), "Model"]
best_acc = df_clf.loc[df_clf["Accuracy"].idxmax(), "Model"]
conclusion = [
    "  FINAL MODEL DEĞERLENDİRMESİ",
    "  " + "─"*30,
    "",
    f"  ★ En Yüksek AUC  : {best_clf}",
    f"    AUC = {df_clf['AUC-ROC'].max():.4f}",
    "",
    f"  ★ En Yüksek Acc  : {best_acc}",
    f"    Acc = {df_clf['Accuracy'].max():.4f}",
    "",
    "  Regresyon (BMI Tahmin):",
    f"    MSE  = {df_reg[df_reg['Set']=='Test']['MSE'].values[0]:.4f}",
    f"    RMSE = {df_reg[df_reg['Set']=='Test']['RMSE'].values[0]:.4f}",
    f"    R²   = {df_reg[df_reg['Set']=='Test']['R2'].values[0]:.4f}",
    "",
    "  AUC Yorumu:",
    "    > 0.9  : Mükemmel",
    "    0.8-0.9: İyi ✅ (bizim modelimiz)",
    "    0.7-0.8: Kabul edilebilir",
    "    0.5-0.6: Zayıf / Şans",
]
ax6.text(0.04, 0.96, "\n".join(conclusion),
         transform=ax6.transAxes, fontsize=8.5,
         color=TXT, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", fc=CARD, ec=GRID, alpha=0.95))
ax6.set_title("Sonuç Yorumu", fontsize=10, fontweight="bold")

plt.tight_layout(); save("06_final_model_dashboard.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 7 — KALIBRASYON EĞRİSİ (Güvenilirlik Diyagramı)
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 7 — KALİBRASYON EĞRİSİ")

from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
ax.plot([0,1],[0,1], "k--", color=C5, lw=2, label="Mükemmel Kalibrasyon")

for (mname, proba), clr in zip(test_probas.items(), MODEL_COLS):
    try:
        prob_true, prob_pred = calibration_curve(y_te, proba, n_bins=8)
        ax.plot(prob_pred, prob_true, marker="o", ms=7,
                color=clr, lw=2.2, label=mname)
        ax.fill_between(prob_pred, prob_pred, prob_true,
                        alpha=0.08, color=clr)
    except Exception:
        pass

ax.set_xlabel("Ortalama Tahmin Edilen Olasılık", fontsize=11)
ax.set_ylabel("Gerçek Oranı (Fraksiyon Pozitif)", fontsize=11)
ax.set_title("Kalibrasyon Eğrisi (Güvenilirlik Diyagramı)\n"
             "Çapraz çizgiye yakın = İyi kalibre",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("07_calibration_curve.png")

# ════════════════════════════════════════════════════════════
#  GRAFİK 8 — ÖZET DASHBOARD
# ════════════════════════════════════════════════════════════
hdr("GRAFİK 8 — ÖZET DASHBOARD")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 7 — Başarı Ölçümü & Final Model Özeti",
             fontsize=16, fontweight="bold", color=TXT, y=1.00)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# P1: ROC (küçük)
ax = fig.add_subplot(gs[0, 0:2]); ax.set_facecolor(CARD)
ax.plot([0,1],[0,1], color=TXT, lw=1, ls=":", alpha=0.4)
for (mname, proba), clr in zip(test_probas.items(), MODEL_COLS):
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    ax.plot(fpr, tpr, color=clr, lw=2.5, label=f"{mname} AUC={auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.07, color=clr)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Eğrileri — Test Seti", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(True)

# P2: PR Eğrisi
ax = fig.add_subplot(gs[0, 2:4]); ax.set_facecolor(CARD)
ax.axhline(y_te.mean(), color=TXT, lw=1, ls=":", alpha=0.5)
for (mname, proba), clr in zip(test_probas.items(), MODEL_COLS):
    p, r, _ = precision_recall_curve(y_te, proba)
    ap = average_precision_score(y_te, proba)
    ax.plot(r, p, color=clr, lw=2.2, label=f"{mname} AP={ap:.4f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Eğrisi", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True)

# P3: AUC bar
ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor(CARD)
aucs_v = [test_aucs[m] for m in models]
ax.bar(range(3), aucs_v, color=MODEL_COLS, edgecolor=DARK, alpha=0.85)
ax.set_xticks(range(3)); ax.set_xticklabels(list(models.keys()), fontsize=7, rotation=12)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC"); ax.set_title("Test AUC", fontsize=10)
for i, v in enumerate(aucs_v):
    ax.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=9, color=TXT)
ax.grid(axis="y")

# P4: Reg MSE bar
ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(CARD)
reg_sets  = df_reg["Set"].tolist()
reg_mses  = df_reg["MSE"].tolist()
reg_rmses = df_reg["RMSE"].tolist()
x = np.arange(2); w = 0.3
ax.bar(x-w/2, reg_mses,  w, color=C3, edgecolor=DARK, label="MSE")
ax.bar(x+w/2, reg_rmses, w, color=C5, edgecolor=DARK, label="RMSE")
ax.set_xticks(x); ax.set_xticklabels(reg_sets)
ax.set_ylabel("Hata değeri"); ax.set_title("Regresyon — MSE & RMSE", fontsize=10)
for i, (m, r) in enumerate(zip(reg_mses, reg_rmses)):
    ax.text(i-w/2, m+0.2, f"{m:.2f}", ha="center", fontsize=8, color=TXT)
    ax.text(i+w/2, r+0.2, f"{r:.2f}", ha="center", fontsize=8, color=TXT)
ax.legend(fontsize=8); ax.grid(axis="y")

# P5: Acc karşılaştırma
ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(CARD)
accs_v = [df_clf[df_clf["Model"]==m]["Accuracy"].values[0] for m in models]
ax.bar(range(3), accs_v, color=MODEL_COLS, edgecolor=DARK, alpha=0.85)
ax.set_xticks(range(3)); ax.set_xticklabels(list(models.keys()), fontsize=7, rotation=12)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Accuracy")
ax.set_title("Test Accuracy", fontsize=10)
for i, v in enumerate(accs_v):
    ax.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=9, color=TXT)
ax.grid(axis="y")

# P6: Regresyon R² + MAE
ax = fig.add_subplot(gs[1, 3]); ax.set_facecolor(CARD)
ax.barh(["Validation", "Test"],
        df_reg["R2"].tolist(), color=[C1, C2], edgecolor=DARK, alpha=0.85)
ax.axvline(0, color=TXT, lw=0.8, alpha=0.4)
ax.set_xlabel("R²"); ax.set_title("Regresyon R²", fontsize=10)
for i, v in enumerate(df_reg["R2"]):
    ax.text(v+0.01, i, f"{v:.4f}", va="center", fontsize=10, color=TXT)
ax.grid(axis="x")

save("08_summary_dashboard.png")

# ── FINAL ÖZET ─────────────────────────────────────────────
hdr("FAZ 7 — TAMAMLANDI ✅")

best_auc_model = max(test_aucs, key=test_aucs.get)
best_acc_model = df_clf.loc[df_clf["Accuracy"].idxmax(), "Model"]

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │            PROJE FINAL ÖZET                         │
  ├──────────────────────────────────────────────────────┤
  │  SINIFLANDIRMA (Test Seti)                          │
  │  Model             Acc     AUC     F1               │""")
for _, r in df_clf.iterrows():
    print(f"  │  {r['Model']:<18} {r['Accuracy']:.4f}  {r['AUC-ROC']:.4f}  {r['F1']:.4f}            │")
print(f"""  ├──────────────────────────────────────────────────────┤
  │  ★ En İyi AUC  : {best_auc_model:<18} ({test_aucs[best_auc_model]:.4f})   │
  │  ★ En İyi Acc  : {best_acc_model:<18} ({df_clf['Accuracy'].max():.4f})   │
  ├──────────────────────────────────────────────────────┤
  │  REGRESYON — BMI Tahmini (Lineer Regresyon)         │
  │    Test MSE   = {df_reg[df_reg['Set']=='Test']['MSE'].values[0]:.4f}                             │
  │    Test RMSE  = {df_reg[df_reg['Set']=='Test']['RMSE'].values[0]:.4f}                             │
  │    Test MAE   = {df_reg[df_reg['Set']=='Test']['MAE'].values[0]:.4f}                             │
  │    Test R²    = {df_reg[df_reg['Set']=='Test']['R2'].values[0]:.4f}                             │
  ├──────────────────────────────────────────────────────┤
  │  Grafikler (8 adet) → {OUT}/             │
  └──────────────────────────────────────────────────────┘
""")
