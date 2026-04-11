"""
=============================================================
  FAZ 6: RESAMPLING — Çapraz Geçerleme & Bootstrap
  • 10-Fold Cross Validation (LR, KNN, GNB)
  • Bootstrap (B=1000) — katsayı güven aralıkları
  • Bias-Corrected Accelerated (BCa) Bootstrap CI
=============================================================

Modül Amacı
-----------
  Bu modül, Faz 4 ve Faz 5'te elde edilen model performans
  tahminlerinin istatistiksel güvenilirliğini ölçer.

  İki temel resampling yöntemi uygulanır:
  1. **10-Fold Stratified Cross Validation:** Tüm veriyi 10 kata
     bölerek modellerin ortalama performansını ±1σ belirsizlikle
     raporlar. Tek bir hold-out test setine göre훨씬 daha kararlı
     bir tahmin üretir.
  2. **Bootstrap (B=1000):** Lojistik Regresyon katsayılarının
     örnekleme dağılımını parametrik olmayan yollarla (non-parametric)
     tahmin eder; iki farklı CI hesaplanır: Percentile ve BCa.

Teorik Arka Plan
----------------
  10-Fold Stratified Cross Validation:
    K=10 kat seçimi: Az sayıda fold → yüksek bias (az test verisi);
    Çok fold → yüksek varyans (test seti değişkenli).
    K=10, bias-variance açısından sektörün standart dengesidir.
    "Stratified" → her fold'da sınıf oranı (%34.9 pozitif) korunur;
    küçük fold'larda şans eseri dengesizlik önlenir.

  Bootstrap Güven Aralığı Yöntemleri:
  a) Percentile CI:
     CI = [θ*_(α/2), θ*_(1-α/2)]
     Boot. dağılımının doğrudan kantilleri kullanılır.
     Hızlı ve sezgisel, ancak skewed (çarpık) dağılımlarda bias var.

  b) BCa (Bias-Corrected & Accelerated) CI:
     z₀ = Φ⁻¹[P(θ* < θ̂)]  (bias düzeltme terimi)
     a = Σ(θ̄−θᵢ)³ / [6·(Σ(θ̄−θᵢ)²)^(3/2)]  (ivme parametresi)
     Skewed dağılımlarda ikinci derece doğru.
     BCa, parametrik varsayımsız en pürüzlü güven aralığıdır.

  Bootstrap Bias:
     θ̄* − θ̂ → Bootstrap katsayısının orijinalden ortalama sapması.
     Büyük bias → model belirli veri yapısına aşırı hassas.

  CV vs. Hold-Out Karşılaştırması:
     CV performansı ile hold-out testi arasındaki benzerlik,
     modelin kararlı genelleştiğini kanıtlar. Büyük uçurum →
     veri varyansı yüksek veya hold-out seti temsili değil.
     FINAL_REPORT.md §8.3'te LR için CV Acc ≈ Test Acc.

Neden Train+Val Bootstrap, Test Bootstrap Değil?
-------------------------------------------------
  Bootstrap katsayı tahmini yalnızca Train+Val verisinde yapılır.
  Test seti "gelecekteki görülmemiş veri"yi temsil eder.
  Bootstrap sürecinde test setini kullanmak, bu prensibi ihlal eder.
  CV ise train+val+test hepsini çapraz düzenlemede kullanır;
  ancak hiçbir doğrulama seti aynı anda hem train hem test olmaz.

Girdiler
--------
  phase1_outputs/train_scaled.csv
  phase1_outputs/val_scaled.csv
  phase1_outputs/test_scaled.csv
  phase1_outputs/train_raw.csv    ← Bootstrap için ölçeklenmemiş
  phase1_outputs/val_raw.csv

Çıktılar
--------
  phase6_outputs/cv_fold_results.csv    – Her fold sonuçları
  phase6_outputs/bootstrap_coefs.csv   – Bootstrap katsayıları
  phase6_outputs/*.png                  – 8 adet görselleştirme
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import warnings, os

from sklearn.linear_model   import LogisticRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics        import (accuracy_score, roc_auc_score,
                                    make_scorer, log_loss, f1_score)
from sklearn.preprocessing  import StandardScaler
from scipy.special          import expit as sigmoid_fn

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

OUT = "phase6_outputs"; os.makedirs(OUT, exist_ok=True)

def save(fname):
    """Figürü OUTPUT_DIR altına kaydeder ve kapatır."""
    plt.savefig(f"{OUT}/{fname}", bbox_inches="tight", facecolor=DARK, dpi=150)
    plt.close(); print(f"  ✅  → {OUT}/{fname}")

def hdr(t): print(f"\n{'═'*62}\n  {t}\n{'═'*62}")

# ── VERİ (ham — ölçeklendirmeyi CV içinde yapacağız) ───────
hdr("ADIM 0 — VERİ YÜKLEME")
train_sc = pd.read_csv("phase1_outputs/train_scaled.csv")
val_sc   = pd.read_csv("phase1_outputs/val_scaled.csv")
test_sc  = pd.read_csv("phase1_outputs/test_scaled.csv")

train_raw = pd.read_csv("phase1_outputs/train_raw.csv")
val_raw   = pd.read_csv("phase1_outputs/val_raw.csv")
test_raw  = pd.read_csv("phase1_outputs/test_raw.csv")

FEAT = [c for c in train_sc.columns if c != "Outcome"]

# CV için tüm veriyi birleştir (ölçeklenmiş): LR ve GNB ölçeklenmiş
# veride doğrudan çalışabilir. Notlar: X_all burada pipelinesiz
# kullanılmaktadır; veri varlıkla zaten Faz 1'de ölçeklenmiş.
all_sc  = pd.concat([train_sc, val_sc, test_sc], ignore_index=True)
X_all   = all_sc[FEAT].values
y_all   = all_sc["Outcome"].values

# Bootstrap için sadece train+val (test'i hiç görmeyiz)
# Katsayı güven aralıklarının test setine bağımlı olmaması
# istatistiksel geçerlilik için zorunludur.
all_tv   = pd.concat([train_raw, val_raw], ignore_index=True)
X_tv_raw = all_tv[FEAT].values
y_tv     = all_tv["Outcome"].values

print(f"  CV için: {len(y_all)} örnek | Bootstrap için: {len(y_tv)} örnek")

# ════════════════════════════════════════════════════════════
#  A — 10-FOLD STRATİFİED CROSS VALIDATION
# ════════════════════════════════════════════════════════════
hdr("A — 10-FOLD STRATİFİED CROSS VALIDATION")

def _safe_auc(y_true, y_prob):
    """
    AUC hesaplama — fold'da tek sınıf varsa 0.5 döndür.

    Parametreler
    ------------
    y_true : ndarray
        Gerçek etiketler.
    y_prob : ndarray
        Tahmin edilen olasılıklar.

    Döndürür
    --------
    float
        AUC-ROC; tek sınıf varsa 0.5 (anlamsız AUC yok).

    Notlar
    ------
    Küçük fold sayısında (K=10) stratify=True ile bu durum
    nadiren oluşur; ancak savunmacı kodlama için gereklidir.
    """
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.5

# StratifiedKFold: sınıf dağılımını her fold'da korur
# shuffle=True + random_state=42: tekrarlanabilir karıştırma
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scorers = {
    "accuracy" : make_scorer(accuracy_score),
    "auc"      : make_scorer(_safe_auc, needs_proba=True),
    "f1"       : make_scorer(f1_score),
    "neg_logloss": make_scorer(log_loss, needs_proba=True,
                               greater_is_better=False),
}

# Faz 5'ten gelen optimum K değeri.
# Val hatası minimum olan K=26 seçilmişti.
best_k = 26   # Faz 5'ten gelen optimum K

models_cv = {
    "Lojistik Reg." : LogisticRegression(max_iter=5000, C=1e6, random_state=42),
    f"KNN (K={best_k})" : KNeighborsClassifier(n_neighbors=best_k, metric="euclidean"),
    "Gaussian NB"   : GaussianNB(),
}

cv_results = {}
print(f"\n  {'Model':<18} {'Acc (μ±σ)':>16}  {'AUC (μ±σ)':>16}  {'F1 (μ±σ)':>16}")
print(f"  {'-'*68}")

for mname, model in models_cv.items():
    res = cross_validate(model, X_all, y_all,
                         cv=skf, scoring=scorers, return_train_score=True)
    cv_results[mname] = res
    acc_mu, acc_sd = res["test_accuracy"].mean(),    res["test_accuracy"].std()
    auc_mu, auc_sd = res["test_auc"].mean(),         res["test_auc"].std()
    f1_mu,  f1_sd  = res["test_f1"].mean(),          res["test_f1"].std()
    print(f"  {mname:<18} {acc_mu:.4f} ± {acc_sd:.4f}  "
          f"{auc_mu:.4f} ± {auc_sd:.4f}  "
          f"{f1_mu:.4f} ± {f1_sd:.4f}")

# ── GRAFİK 1: CV Fold Sonuçları ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("10-Fold Stratified Cross Validation — Her Modelin Fold Sonuçları",
             fontsize=14, fontweight="bold", color=TXT)

MODEL_COLS = [C1, C2, C4]
metrics_show = ["test_accuracy", "test_auc", "test_f1"]
metric_labels = ["Accuracy", "AUC-ROC", "F1"]

for ax, metric, mlabel in zip(axes, metrics_show, metric_labels):
    ax.set_facecolor(CARD)
    folds = np.arange(1, 11)
    for (mname, res), clr in zip(cv_results.items(), MODEL_COLS):
        vals = res[metric]
        mu   = vals.mean()
        ax.plot(folds, vals, marker="o", ms=6, color=clr,
                lw=1.8, alpha=0.85, label=f"{mname} (μ={mu:.3f})")
        # Ortalama çizgisi: fold boyunca tutarlılığı gösterir
        ax.axhline(mu, color=clr, lw=1, ls="--", alpha=0.5)
        ax.fill_between(folds, vals, mu, alpha=0.06, color=clr)
    ax.set_xlabel("Fold Numarası", fontsize=10)
    ax.set_ylabel(mlabel, fontsize=10)
    ax.set_title(f"{mlabel} — 10-Fold CV", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(list(folds)); ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout(); save("01_cv_fold_results.png")

# ── GRAFİK 2: CV Violin / Box Plot ─────────────────────────
# Violin plot: olasılık veya KDE dağılımını gösterir.
# Box plot: medyan, IQR ve outlier'ları gösterir.
# İkisinin birleşimi en kapsamlı performans belirsizlik görünümünü verir.
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("10-Fold CV — Dağılım (Box + Violin)",
             fontsize=13, fontweight="bold", color=TXT)

for ax, metric, mlabel in zip(axes, metrics_show, metric_labels):
    ax.set_facecolor(CARD)
    data_list = [cv_results[m][metric] for m in cv_results]
    vp = ax.violinplot(data_list, showmedians=True, showextrema=True)
    for pc, clr in zip(vp["bodies"], MODEL_COLS):
        pc.set_facecolor(clr); pc.set_alpha(0.55)
    vp["cmedians"].set_color(C5); vp["cmedians"].set_linewidth(2)
    for part in ["cmins","cmaxes","cbars"]:
        vp[part].set_color(TXT)
    # Box plot üstüne
    bp = ax.boxplot(data_list, patch_artist=False,
                    medianprops={"color": C5, "lw": 2},
                    whiskerprops={"color": TXT},
                    capprops={"color": TXT},
                    flierprops={"marker":"o","markersize":4,
                                "markerfacecolor":C4,"alpha":0.6})
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(list(cv_results.keys()), rotation=12, fontsize=8)
    ax.set_title(f"{mlabel}", fontsize=11, fontweight="bold")
    ax.set_ylabel(mlabel, fontsize=9); ax.grid(axis="y")
    # Ortalamalar konsol çıktısıyla tutarlı
    for i, (mname, res) in enumerate(cv_results.items(), 1):
        mu = res[metric].mean()
        ax.text(i, mu, f" μ={mu:.3f}", va="center",
                fontsize=7.5, color=MODEL_COLS[i-1], fontweight="bold")

plt.tight_layout(); save("02_cv_violin.png")

# ── GRAFİK 3: CV Özet Bar ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle("10-Fold CV — Ortalama ± 1σ (μ ± σ)", fontsize=13, fontweight="bold", color=TXT)

for ax, metric, mlabel in zip(axes, metrics_show, metric_labels):
    ax.set_facecolor(CARD)
    mus  = [cv_results[m][metric].mean() for m in cv_results]
    sds  = [cv_results[m][metric].std()  for m in cv_results]
    bars = ax.bar(range(3), mus, color=MODEL_COLS,
                  edgecolor=DARK, linewidth=1.2, alpha=0.85)
    # Hata çubukları: ±1σ → 10 fold boyunca performansın değişkenliği
    ax.errorbar(range(3), mus, yerr=sds, fmt="none",
                color=TXT, capsize=6, capthick=2, elinewidth=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(cv_results.keys()), rotation=12, fontsize=8)
    ax.set_title(mlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(mlabel, fontsize=9); ax.grid(axis="y")
    for bar, mu, sd in zip(bars, mus, sds):
        ax.text(bar.get_x()+bar.get_width()/2,
                mu + sd + 0.005,
                f"{mu:.3f}\n±{sd:.3f}",
                ha="center", fontsize=8, color=TXT, fontweight="bold")

plt.tight_layout(); save("03_cv_summary_bar.png")

# CV sonuçlarını kaydet
cv_df_rows = []
for mname, res in cv_results.items():
    for fold, (acc, auc, f1) in enumerate(zip(
            res["test_accuracy"], res["test_auc"], res["test_f1"]), 1):
        cv_df_rows.append({"Model": mname, "Fold": fold,
                           "Accuracy": acc, "AUC": auc, "F1": f1})
pd.DataFrame(cv_df_rows).to_csv(f"{OUT}/cv_fold_results.csv", index=False)
print(f"  CV CSV → {OUT}/cv_fold_results.csv")

# ════════════════════════════════════════════════════════════
#  B — BOOTSTRAP (B=1000) — Lojistik Regresyon Katsayıları
# ════════════════════════════════════════════════════════════
hdr("B — BOOTSTRAP (B=1000) — Katsayı Güven Aralıkları")

# Bootstrap prosedürü:
# 1. Train+Val'den yerine koyarak n adet örnek çek (n=|Train+Val|)
# 2. Her bootstrap örneğinde StandardScaler fit et (veri sızıntısı önlemi)
# 3. LR fit et → katsayıları kaydet
# 4. 1000 replikadan katsayı dağılımını oluştur → CI hesapla
# Not: Her bootstrap örnekleminde scaler yeniden fit ediliyor (önemli!).
# Aksi halde orijinal train'in ölçeğini replicaler'a dayatmış oluruz.
B            = 1000
rng          = np.random.default_rng(42)
n_tv         = len(y_tv)
FEAT_NAMES   = ["intercept"] + FEAT

# Her bootstrap replikasında ölçeklen + fit
boot_coefs   = np.zeros((B, len(FEAT_NAMES)))  # (1000, 9)
boot_accs    = np.zeros(B)

print(f"  {B} bootstrap örneklemesi çekiliyor...")
for b in range(B):
    # Yerine koyarak örnekleme: bazı örnekler birden fazla kez seçilir.
    # Ortalama uniqe örnek oranı: 1 − 1/e ≈ 63.2%
    idx     = rng.integers(0, n_tv, size=n_tv)   # yerine koyarak
    X_boot  = X_tv_raw[idx]
    y_boot  = y_tv[idx]

    # Her bootstrap örneğinde scaler ayrı fit edilir (veri sızıntısı önlemi).
    scaler  = StandardScaler()
    X_b_sc  = scaler.fit_transform(X_boot)

    lr_b = LogisticRegression(max_iter=2000, C=1e6, random_state=0)
    lr_b.fit(X_b_sc, y_boot)

    coefs   = np.concatenate([[lr_b.intercept_[0]], lr_b.coef_[0]])
    boot_coefs[b] = coefs
    boot_accs[b]  = accuracy_score(y_boot, lr_b.predict(X_b_sc))

print(f"  Tamamlandı.")

# Orijinal model katsayıları (tüm train_sc üzerinde)
lr_orig = LogisticRegression(max_iter=5000, C=1e6, random_state=42)
scaler_orig = StandardScaler()
X_tv_sc_orig = scaler_orig.fit_transform(X_tv_raw)
lr_orig.fit(X_tv_sc_orig, y_tv)
orig_coefs = np.concatenate([[lr_orig.intercept_[0]], lr_orig.coef_[0]])

# Percentile CI (%95)
# Bootstrap dağılımının doğrudan 2.5. ve 97.5. persentilleri
ci_lo_p   = np.percentile(boot_coefs, 2.5, axis=0)
ci_hi_p   = np.percentile(boot_coefs, 97.5, axis=0)

# BCa (Bias-Corrected Accelerated) CI
def bca_ci(boot_dist, orig_val, alpha=0.05):
    """
    Bias-Corrected Accelerated (BCa) Bootstrap Güven Aralığı hesaplar.

    Parametreler
    ------------
    boot_dist : ndarray, şekil (B,)
        Bootstrap replikalarından elde edilen istatistik dağılımı.
    orig_val : float
        Orijinal (tüm) veri üzerinde hesaplanan nokta tahmini.
    alpha : float, varsayılan 0.05
        Güven düzeyi: 1−α = %95 CI.

    Döndürür
    --------
    tuple (float, float)
        (CI alt sınırı, CI üst sınırı).

    Algoritma
    ---------
    1. Bias düzeltme terimi z₀:
       z₀ = Φ⁻¹[P(θ* < θ̂)]
       Boot. dağılımında orijinal değerin altında kalanların oranı.

    2. İvme parametresi a (jackknife yöntemiyle):
       L_i = θ̄_{jack} − θ_{jack,i}
       a = Σ(L³) / [6·(Σ(L²))^(3/2)]

    3. Düzeltilmiş kantiller:
       a₁ = Φ(z₀ + (z₀ + z_{α/2}) / (1 − a·(z₀ + z_{α/2})))
       a₂ = Φ(z₀ + (z₀ + z_{1−α/2}) / (1 − a·(z₀ + z_{1−α/2})))

    4. CI = [θ*_{a₁}, θ*_{a₂}]

    Notlar
    ------
    Jackknife için hesaplama maliyetini azaltmak amacıyla
    yalnızca B değerlerinin ilk 100'ü kullanılır (yaklaşım).
    """
    n = len(boot_dist)
    # Bias correction: z₀
    z0 = sp_stats.norm.ppf(np.mean(boot_dist < orig_val))
    # Acceleration: jackknife ile a parametresi
    jack = np.array([np.mean(np.delete(boot_dist, i)) for i in range(min(n, 100))])
    L    = np.mean(jack) - jack
    a    = np.sum(L**3) / (6 * (np.sum(L**2))**1.5 + 1e-15)
    # Adjusted quantiles
    z_al = sp_stats.norm.ppf(alpha/2)
    z_ah = sp_stats.norm.ppf(1 - alpha/2)
    a1   = sp_stats.norm.cdf(z0 + (z0 + z_al) / (1 - a*(z0 + z_al)))
    a2   = sp_stats.norm.cdf(z0 + (z0 + z_ah) / (1 - a*(z0 + z_ah)))
    return np.percentile(boot_dist, a1*100), np.percentile(boot_dist, a2*100)

ci_lo_bca = np.zeros(len(FEAT_NAMES))
ci_hi_bca = np.zeros(len(FEAT_NAMES))
for i in range(len(FEAT_NAMES)):
    lo, hi = bca_ci(boot_coefs[:, i], orig_coefs[i])
    ci_lo_bca[i], ci_hi_bca[i] = lo, hi

# Katsayı istatistikleri
boot_means = boot_coefs.mean(axis=0)
boot_stds  = boot_coefs.std(axis=0)
# Bootstrap bias: boot ortalaması orijinalden ne kadar sapıyor?
# Büyük bias → model parametresi örnekleme dağılımına duyarlı.
boot_bias  = boot_means - orig_coefs   # bootstrap bias

df_boot = pd.DataFrame({
    "Parametre"    : FEAT_NAMES,
    "Orig_Coef"    : orig_coefs,
    "Boot_Mean"    : boot_means,
    "Boot_Std"     : boot_stds,
    "Bootstrap_Bias": boot_bias,
    "CI95_Pct_Lo"  : ci_lo_p,
    "CI95_Pct_Hi"  : ci_hi_p,
    "CI95_BCa_Lo"  : ci_lo_bca,
    "CI95_BCa_Hi"  : ci_hi_bca,
    # Anlamlılık: CI'nın sıfır içermemesi → β≠0 (p<0.05 benzeri)
    "Anlamli_Pct"  : ~((ci_lo_p <= 0) & (ci_hi_p >= 0)),
    "Anlamli_BCa"  : ~((ci_lo_bca <= 0) & (ci_hi_bca >= 0)),
})
df_boot.to_csv(f"{OUT}/bootstrap_coefs.csv", index=False)

print(f"\n  {'Parametre':<28} {'Orig':>8}  {'Boot μ':>8}  {'Boot σ':>8}  "
      f"{'95%CI (Pct)':>20}  Sig")
print(f"  {'-'*85}")
for _, r in df_boot.iterrows():
    sig = "★" if r["Anlamli_Pct"] else " "
    print(f"  {r['Parametre']:<28} {r['Orig_Coef']:>8.4f}  "
          f"{r['Boot_Mean']:>8.4f}  {r['Boot_Std']:>8.4f}  "
          f"[{r['CI95_Pct_Lo']:>6.3f}, {r['CI95_Pct_Hi']:>6.3f}]  {sig}")

# ── GRAFİK 4: Katsayı Histogramları (Bootstrap Dağılımı) ───
hdr("GRAFİK 4 — KATSAYI DISTRİBÜSYONLARI (Bootstrap)")

# Bootstrap dağılımının normal görünmesi, Merkezi Limit Teoremi'nin
# büyük B için uygulanabildiğini gösterir. N(μ,σ) fit eğrisi bu
# normalliği sayısal olarak (görsel olarak) doğrular.
n_rows, n_cols = 3, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle(f"Bootstrap (B={B}) — Katsayı Dağılımları ve %95 Güven Aralıkları",
             fontsize=14, fontweight="bold", color=TXT)

axes_flat = axes.flatten()
colors_panel = [C1, C2, C3, C4, C5, C6, "#56d364", "#f47067", "#bc8cff"]

for i, (fname, ax, clr) in enumerate(zip(FEAT_NAMES, axes_flat, colors_panel)):
    ax.set_facecolor(CARD)
    vals = boot_coefs[:, i]
    orig = orig_coefs[i]
    ci_lo, ci_hi = ci_lo_p[i], ci_hi_p[i]

    # Histogram: B=1000 replikadan oluşan ampirik dağılım
    ax.hist(vals, bins=40, color=clr, edgecolor=DARK,
            linewidth=0.6, alpha=0.80, density=True)

    # Normal fit eğrisi: dağılımın normallik derecesini gösterir.
    # Teorik olarak B→∞ limitinde dağılım normal olur (CLT).
    mu_, sd_ = vals.mean(), vals.std()
    xr = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xr, sp_stats.norm.pdf(xr, mu_, sd_),
            color=C5, lw=2, label="N(μ,σ) fit")

    # Şeritler
    ax.axvline(orig,   color=C5,  lw=2.2, label=f"Orig={orig:.4f}")
    ax.axvline(ci_lo,  color=C3,  lw=1.8, ls="--", label=f"CI95=[{ci_lo:.3f},{ci_hi:.3f}]")
    ax.axvline(ci_hi,  color=C3,  lw=1.8, ls="--")
    ax.axvline(0,      color=TXT, lw=0.8, alpha=0.5)
    # CI gölgesi: güven aralığını doldurulan bölge ile vurgular
    ax.fill_between([ci_lo, ci_hi],
                    [0,0], [ax.get_ylim()[1]*1.5 if ax.get_ylim()[1] > 0 else 5],
                    alpha=0.08, color=C3, label="95% CI")
    ax.axvline(0, color=TXT, lw=0.8, alpha=0.4)

    # Anlamlılık: CI sıfırı içermiyorsa β anlamlı kabul edilir
    sig = "★ Anlamlı" if not (ci_lo <= 0 <= ci_hi) else "— Anlamsız"
    ax.set_title(f"{fname}\n{sig}", fontsize=9, pad=5,
                 color=C2 if "Anlamlı" in sig else C3, fontweight="bold")
    ax.set_xlabel("Katsayı Değeri", fontsize=8)
    ax.set_ylabel("Yoğunluk", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(axis="y")

plt.tight_layout(); save("04_bootstrap_histograms.png")

# ── GRAFİK 5: Forest Plot (Bootstrap CI) ───────────────────
hdr("GRAFİK 5 — BOOTSTRAP FOREST PLOT")

# Percentile CI vs BCa CI karşılaştırması:
# Simetrik dağılımlarda ikisi aynı sonucu verir.
# BCa, asimetrik veya yanlı (biased) katsayılarda daha güvenilir.
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("Bootstrap Güven Aralıkları — Percentile vs. BCa (%95)",
             fontsize=13, fontweight="bold", color=TXT)

feat_only = df_boot.iloc[1:]   # intercept hariç
y_pos     = range(len(feat_only))

for ax, (lo_col, hi_col, sig_col, title, clr_sig) in zip(axes, [
    ("CI95_Pct_Lo", "CI95_Pct_Hi", "Anlamli_Pct", "Percentile CI", C2),
    ("CI95_BCa_Lo", "CI95_BCa_Hi", "Anlamli_BCa", "BCa CI",        C4),
]):
    ax.set_facecolor(CARD)
    for i, (_, row) in enumerate(feat_only.iterrows()):
        clr = clr_sig if row[sig_col] else C3
        ax.plot([row[lo_col], row[hi_col]], [i, i],
                lw=3, color=clr, solid_capstyle="round")
        ax.scatter(row["Orig_Coef"], i, color=C5, s=70, zorder=5)
        ax.scatter(row["Boot_Mean"], i, color=clr, s=40, zorder=5,
                   marker="D")
    ax.axvline(0, color=C5, lw=1.5, ls="--", alpha=0.8, label="β=0")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feat_only["Parametre"], fontsize=9)
    ax.set_xlabel("Katsayı (95% Bootstrap CI)", fontsize=10)
    ax.set_title(f"{title}\n(★=Sıfır içermiyor → Anlamlı)",
                 fontsize=11, fontweight="bold", pad=10)
    import matplotlib.patches as mpa
    ax.legend(handles=[
        mpa.Patch(color=clr_sig, label="p<0.05 ★ Anlamlı"),
        mpa.Patch(color=C3, label="p≥0.05 — Anlamsız"),
    ], fontsize=8); ax.grid(axis="x")

plt.tight_layout(); save("05_bootstrap_forest.png")

# ── GRAFİK 6: Bootstrap Accuracy Dağılımı ──────────────────
hdr("GRAFİK 6 — BOOTSTRAP ACCURACY DAĞILIMI")

# Bootstrap accuracy dağılımı, modelin genel performans güvenilirliğini
# gösterir. Q-Q plot normallik testi olarak kullanılır.
# R² > 0.99 → dağılım normale yakın → CLT geçerli.
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("Bootstrap Accuracy Dağılımı — Performans Güvenilirliği",
             fontsize=13, fontweight="bold", color=TXT)

ax = axes[0]; ax.set_facecolor(CARD)
ax.hist(boot_accs, bins=40, color=C1, edgecolor=DARK,
        linewidth=0.6, alpha=0.85, density=True)
mu_acc = boot_accs.mean(); sd_acc = boot_accs.std()
xr = np.linspace(boot_accs.min(), boot_accs.max(), 300)
ax.plot(xr, sp_stats.norm.pdf(xr, mu_acc, sd_acc),
        color=C5, lw=2.5, label=f"N(μ={mu_acc:.4f}, σ={sd_acc:.4f})")
ci_acc_lo = np.percentile(boot_accs, 2.5)
ci_acc_hi = np.percentile(boot_accs, 97.5)
ax.axvline(mu_acc,    color=C2,  lw=2, label=f"μ={mu_acc:.4f}")
ax.axvline(ci_acc_lo, color=C3, lw=1.8, ls="--",
           label=f"95% CI=[{ci_acc_lo:.3f},{ci_acc_hi:.3f}]")
ax.axvline(ci_acc_hi, color=C3, lw=1.8, ls="--")
ax.fill_between([ci_acc_lo, ci_acc_hi], 0, 30, alpha=0.10, color=C3)
ax.set_xlabel("Accuracy (Bootstrap)", fontsize=10)
ax.set_ylabel("Yoğunluk", fontsize=10)
ax.set_title(f"Bootstrap Accuracy Histogramı\n(B={B})", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y")

# Q-Q plot: noktalar çizgi üzerinde ise dağılım normal.
# R²: 1'e ne kadar yakınsa o kadar ideal normal.
ax2 = axes[1]; ax2.set_facecolor(CARD)
(osm, osr), (slope, intercept, r) = sp_stats.probplot(boot_accs, dist="norm")
ax2.scatter(osm, osr, color=C1, alpha=0.6, s=12)
ax2.plot(osm, slope*np.array(osm)+intercept, color=C5,
         lw=2, label=f"R²={r**2:.4f}")
ax2.set_xlabel("Teorik Kantiller", fontsize=10)
ax2.set_ylabel("Örnek Kantilleri", fontsize=10)
ax2.set_title("Q-Q Plot — Bootstrap Accuracy Normal mi?",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout(); save("06_bootstrap_accuracy.png")

# ── GRAFİK 7: CV vs. Hold-Out Karşılaştırması ──────────────
hdr("GRAFİK 7 — CV vs. HOLD-OUT KARŞILAŞTIRMASI")

# CV ve hold-out sonuçları arasındaki benzerlik, cross-validation'ın
# gerçek performansa ne kadar iyi yaklaştığını gösterir.
# Büyük uçurum → ya CV seti temsili değil ya da test seti şansına bağlı.
# FINAL_REPORT.md §8.3: LR için CV ≈ Hold-Out → model kararlı.
X_te_sc = test_sc[FEAT].values; y_te = test_sc["Outcome"].values
holdout = {}
for mname, model in models_cv.items():
    model.fit(X_all, y_all)
    holdout[mname] = {
        "acc": accuracy_score(y_te, model.predict(X_te_sc)),
        "auc": roc_auc_score(y_te, model.predict_proba(X_te_sc)[:,1]),
    }

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle("10-Fold CV vs. Hold-Out Test — Karşılaştırma",
             fontsize=13, fontweight="bold", color=TXT)

for ax, metric, mlabel in zip(axes, ["acc","auc"], ["Accuracy", "AUC-ROC"]):
    ax.set_facecolor(CARD)
    cv_key  = "test_accuracy" if metric=="acc" else "test_auc"
    x = np.arange(3); w = 0.3
    cv_mus  = [cv_results[m][cv_key].mean()  for m in cv_results]
    cv_sds  = [cv_results[m][cv_key].std()   for m in cv_results]
    ho_vals = [holdout[m][metric] for m in holdout]

    b1 = ax.bar(x-w/2, cv_mus, w, color=MODEL_COLS, edgecolor=DARK,
                alpha=0.85, label="10-Fold CV (μ)")
    ax.errorbar(x-w/2, cv_mus, yerr=cv_sds, fmt="none",
                color=TXT, capsize=5, capthick=1.5, elinewidth=1.5)
    b2 = ax.bar(x+w/2, ho_vals, w, color=MODEL_COLS, edgecolor=DARK,
                alpha=0.5, hatch="///", label="Hold-Out Test")

    for xv, cv, ho in zip(x, cv_mus, ho_vals):
        ax.text(xv-w/2, cv+cv_sds[list(x).index(xv)]+0.005,
                f"{cv:.3f}", ha="center", fontsize=8, color=TXT)
        ax.text(xv+w/2, ho+0.005, f"{ho:.3f}",
                ha="center", fontsize=8, color=TXT)

    ax.set_xticks(x)
    ax.set_xticklabels(list(cv_results.keys()), fontsize=8, rotation=12)
    ax.set_ylabel(mlabel, fontsize=10)
    ax.set_title(f"{mlabel}: CV vs. Hold-Out", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y")
    ax.set_ylim(0, 1)

plt.tight_layout(); save("07_cv_vs_holdout.png")

# ── GRAFİK 8: ÖZET DASHBOARD ───────────────────────────────
hdr("GRAFİK 8 — ÖZET DASHBOARD")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle("FAZ 6 — Resampling Özet Dashboard: CV & Bootstrap",
             fontsize=16, fontweight="bold", color=TXT, y=1.00)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel 1: CV Accuracy Fold eğrileri
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(CARD)
folds = np.arange(1, 11)
for (mname, res), clr in zip(cv_results.items(), MODEL_COLS):
    ax1.plot(folds, res["test_accuracy"], marker="o", ms=5,
             color=clr, lw=1.8, label=mname)
    ax1.axhline(res["test_accuracy"].mean(), color=clr, lw=1, ls="--", alpha=0.5)
ax1.set_xlabel("Fold"); ax1.set_ylabel("Accuracy")
ax1.set_title("10-Fold CV Accuracy\n(Her fold sonucu)", fontsize=10, fontweight="bold")
ax1.set_xticks(list(folds)); ax1.legend(fontsize=7); ax1.grid(True)

# Panel 2: CV μ±σ bar
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(CARD)
mus = [cv_results[m]["test_accuracy"].mean() for m in cv_results]
sds = [cv_results[m]["test_accuracy"].std()  for m in cv_results]
bars = ax2.bar(range(3), mus, color=MODEL_COLS, edgecolor=DARK, alpha=0.85)
ax2.errorbar(range(3), mus, yerr=sds, fmt="none",
             color=TXT, capsize=6, capthick=2, elinewidth=2)
ax2.set_xticks(range(3))
ax2.set_xticklabels(list(cv_results.keys()), rotation=12, fontsize=8)
ax2.set_title("CV Accuracy μ ± σ", fontsize=10, fontweight="bold")
ax2.set_ylabel("Accuracy"); ax2.grid(axis="y")
for bar, mu, sd in zip(bars, mus, sds):
    ax2.text(bar.get_x()+bar.get_width()/2,
             mu+sd+0.005, f"{mu:.3f}\n±{sd:.3f}",
             ha="center", fontsize=8, color=TXT)

# Panel 3: Bootstrap Accuracy hist
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(CARD)
ax3.hist(boot_accs, bins=35, color=C1, edgecolor=DARK,
         linewidth=0.5, alpha=0.85, density=True)
ax3.axvline(mu_acc, color=C2, lw=2, label=f"μ={mu_acc:.4f}")
ax3.axvline(ci_acc_lo, color=C3, lw=1.8, ls="--")
ax3.axvline(ci_acc_hi, color=C3, lw=1.8, ls="--",
            label=f"95%CI=[{ci_acc_lo:.3f},{ci_acc_hi:.3f}]")
ax3.set_xlabel("Accuracy"); ax3.set_ylabel("Yoğunluk")
ax3.set_title(f"Bootstrap Acc. Dağılımı\n(B={B})", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(axis="y")

# Panel 4: Bootstrap Forest (top 5 özellik)
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(CARD)
top5 = feat_only.nlargest(5, "Orig_Coef")
y5   = range(len(top5))
for i, (_, row) in enumerate(top5.iterrows()):
    clr = C2 if row["Anlamli_Pct"] else C3
    ax4.plot([row["CI95_Pct_Lo"], row["CI95_Pct_Hi"]], [i, i],
             lw=3.5, color=clr, solid_capstyle="round")
    ax4.scatter(row["Orig_Coef"], i, color=C5, s=80, zorder=5)
ax4.axvline(0, color=C5, lw=1.5, ls="--", alpha=0.8)
ax4.set_yticks(list(y5))
ax4.set_yticklabels(top5["Parametre"], fontsize=9)
ax4.set_title("Bootstrap CI (Top-5 Katsayı)", fontsize=10, fontweight="bold")
ax4.set_xlabel("β Değeri"); ax4.grid(axis="x")

# Panel 5: CV AUC karşılaştırması
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(CARD)
mus_auc = [cv_results[m]["test_auc"].mean() for m in cv_results]
sds_auc = [cv_results[m]["test_auc"].std()  for m in cv_results]
bars5 = ax5.bar(range(3), mus_auc, color=MODEL_COLS, edgecolor=DARK, alpha=0.85)
ax5.errorbar(range(3), mus_auc, yerr=sds_auc,
             fmt="none", color=TXT, capsize=6, capthick=2, elinewidth=2)
ax5.set_xticks(range(3))
ax5.set_xticklabels(list(cv_results.keys()), rotation=12, fontsize=8)
ax5.set_title("CV AUC μ ± σ", fontsize=10, fontweight="bold")
ax5.set_ylabel("AUC-ROC"); ax5.grid(axis="y")
for bar, mu, sd in zip(bars5, mus_auc, sds_auc):
    ax5.text(bar.get_x()+bar.get_width()/2,
             mu+sd+0.003, f"{mu:.3f}\n±{sd:.3f}",
             ha="center", fontsize=8, color=TXT)

# Panel 6: CV vs Holdout scatter
# Noktalar y=x çizgisine yakınsa CV ile Hold-Out performansları örtüşüyor → kararlı model.
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(CARD)
cv_accs_all = [cv_results[m]["test_accuracy"].mean() for m in cv_results]
ho_accs_all = [holdout[m]["acc"] for m in holdout]
for (mname, cvv, hov), clr in zip(
        zip(list(cv_results.keys()), cv_accs_all, ho_accs_all), MODEL_COLS):
    ax6.scatter([cvv], [hov], color=clr, s=120, zorder=5,
                edgecolors=DARK, linewidths=1.5, label=mname)
    ax6.annotate(f"  {mname}", (cvv, hov), fontsize=7.5, color=clr)
mn = min(min(cv_accs_all), min(ho_accs_all)) - 0.02
mx = max(max(cv_accs_all), max(ho_accs_all)) + 0.02
ax6.plot([mn, mx], [mn, mx], "--", color=TXT, alpha=0.5, lw=1.5,
         label="CV = Hold-Out")
ax6.set_xlabel("10-Fold CV Accuracy", fontsize=9)
ax6.set_ylabel("Hold-Out Test Accuracy", fontsize=9)
ax6.set_title("CV vs. Hold-Out Uyumu", fontsize=10, fontweight="bold")
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

save("08_summary_dashboard.png")

# ── SONUÇ ──────────────────────────────────────────────────
hdr("FAZ 6 — TAMAMLANDI ✅")

for mname, res in cv_results.items():
    acc = res["test_accuracy"]
    auc = res["test_auc"]
    print(f"  {mname:<18}  Acc={acc.mean():.4f}±{acc.std():.4f}  "
          f"AUC={auc.mean():.4f}±{auc.std():.4f}")

sig_boot = df_boot[df_boot["Anlamli_Pct"] & (df_boot["Parametre"]!="intercept")]
print(f"\n  Bootstrap'ta CI sıfırı içermeyen (anlamlı) katsayılar:")
for _, r in sig_boot.iterrows():
    print(f"    ★ {r['Parametre']:<28} β={r['Orig_Coef']:.4f}  "
          f"95%CI=[{r['CI95_Pct_Lo']:.3f},{r['CI95_Pct_Hi']:.3f}]")

print(f"\n  Bootstrap Accuracy: μ={mu_acc:.4f}  σ={sd_acc:.4f}  "
      f"95%CI=[{ci_acc_lo:.4f},{ci_acc_hi:.4f}]")
print(f"  Grafikler (8 adet) → {OUT}/")
