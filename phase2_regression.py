"""
=============================================================
  FAZ 2: REGRESYON ANALİZİ — EKK vs. MLE
  Hedef Değişken: BMI
  Senaryo: Diğer özelliklerle BMI tahmini
=============================================================
  Adım 2A — EKK (Ordinary Least Squares) : sklearn
  Adım 2B — MLE (Maximum Likelihood Est.): el ile log-L
  Karşılaştırma: β_EKK ≈ β_MLE (normal hata varsayımı)
=============================================================

Modül Amacı
-----------
  Bu modül Faz 1'den gelen ham veri setlerini okuyarak
  BMI değerini diğer özelliklerden tahmin etmeyi deneyen
  bir doğrusal regresyon analizi gerçekleştirir.

  İki ayrı yöntem uygulanır:
  1. **EKK (Ordinary Least Squares):** Artık karelerin toplamını
     minimize eder; sklearn LinearRegression ve analitik Normal
     Denklem ile hesaplanır.
  2. **MLE (Maximum Likelihood Estimation):** Hata dağılımının
     N(0, σ²) olduğu varsayımı altında log-olasılık fonksiyonu
     scipy.optimize.minimize (L-BFGS-B) ile maksimize edilir.

  Temel soru: İki yöntemin ürettiği β katsayıları neden
  neredeyse aynıdır? (Cevap: FINAL_REPORT.md §3.2)

Teorik Arka Plan
----------------
  EKK çözümü (Normal Denklem):
    β̂_EKK = (XᵀX)⁻¹ Xᵀy

  Log-Likelihood maksimizasyonu:
    ln L(β,σ) = -n/2·ln(2πσ²) - 1/(2σ²)·‖y - Xβ‖²

  ∂ ln L / ∂β = 0  →  XᵀXβ = Xᵀy  (Normal Denklem!)
  ∴ β_EKK ≡ β_MLE (normal hata varsayımı altında matematiksel zorunluluk)

  Sayısal doğrulama (FINAL_REPORT.md §3.3):
    |β_EKK − β_MLE|_max ≈ 3.16 × 10⁻¹⁶  (makine hassasiyeti)

Girdiler
--------
  phase1_outputs/train_raw.csv  – Ham (ölçeklenmemiş) train seti
  phase1_outputs/val_raw.csv    – Ham validation seti
  phase1_outputs/test_raw.csv   – Ham test seti

Çıktılar
--------
  phase2_outputs/beta_comparison.csv  – β_EKK, β_MLE, mutlak farklar
  phase2_outputs/metrics.csv          – MSE, RMSE, MAE, R² metrikleri
  phase2_outputs/*.png                – 8 adet görselleştirme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  KOYU TEMA
#  Faz 1 ile aynı renk paleti kullanılır;
#  tüm fazlarda görsel tutarlılık sağlanır.
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

PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6,
           "#56d364", "#f47067", "#bc8cff", "#ffb86c"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "monospace",
    "figure.dpi":        120,
})

OUTPUT_DIR = "phase2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fname):
    """
    Mevcut Matplotlib figürünü OUTPUT_DIR altına kaydeder ve kapatır.

    Parametreler
    ------------
    fname : str
        Kaydedilecek dosya adı (örn. "01_beta_comparison.png").
    """
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight", facecolor=DARK_BG, dpi=150)
    plt.close()
    print(f"  ✅  → {path}")

def section_header(title):
    """Konsol çıktısına görsel ayırıcı başlık basar."""
    line = "═" * 62
    print(f"\n{line}\n  {title}\n{line}")


# ═══════════════════════════════════════════════════════════
#  ADIM 0 — VERİ YÜKLEME (Faz 1 çıktıları)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 0 — VERİ YÜKLEME")

# Faz 1'in ham (ölçeklenmemiş, ancak medyan doldurulmuş) setleri yüklenir.
# Faz 2, kendi ölçeklendirmesini yapacaktır çünkü hedef değişken BMI'dır
# ve StandardScaler yalnızca features'a değil, BMI target'a da uygulanır.
train = pd.read_csv("phase1_outputs/train_raw.csv")
val   = pd.read_csv("phase1_outputs/val_raw.csv")
test  = pd.read_csv("phase1_outputs/test_raw.csv")

# Hedef: BMI  |  Özellikler: BMI ve Outcome hariç her şey
TARGET    = "BMI"
FEATURES  = [c for c in train.columns if c not in [TARGET, "Outcome"]]

print(f"\n  Hedef değişken   : {TARGET}")
print(f"  Özellikler ({len(FEATURES)})    : {FEATURES}")
print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

X_train_raw = train[FEATURES].values
y_train     = train[TARGET].values
X_val_raw   = val[FEATURES].values
y_val       = val[TARGET].values
X_test_raw  = test[FEATURES].values
y_test      = test[TARGET].values

# ─── Regresyon için ölçeklendirme (Train'e fit) ───────────
# Hem özellikler hem de hedef (BMI) z-skor ölçeklendirilir.
# Veri sızıntısını önlemek için scaler yalnızca train'e fit edilir.
# (Faz 1'deki aynı ilke — FINAL_REPORT.md §2.3)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_val_s = scaler_X.transform(X_val_raw)
X_test_s= scaler_X.transform(X_test_raw)

# y de ölçeklenir: standardize uzayda hesaplanan β katsayıları
# yorumlanabilir olmakla birlikte, metrikler orijinal BMI birimine
# çevrilerek raporlanır (FINAL_REPORT.md §3.5).
y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).ravel()

n, p = X_train.shape   # n=örnekler (536), p=özellik sayısı (7)

# Bias (intercept) ekle → [1 | X], boyut: (n, p+1) = (536, 8)
# Bu sütun, modelin bir sabit terim (β₀) öğrenmesini sağlar.
X_aug = np.hstack([np.ones((n, 1)), X_train])   # (n, p+1)
param_names = ["intercept"] + FEATURES


# ═══════════════════════════════════════════════════════════
#  ADIM 1 — EKK ANALİTİK ÇÖZÜM (Normal Denklem)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 1 — EKK ANALİTİK ÇÖZÜM (Normal Denklem)")

# Normal Denklem: β̂_EKK = (XᵀX)⁻¹ Xᵀy
# Bu formül, kapalı biçim (closed-form) çözümdür.
# Büyük veri setlerinde matris tersi hesaplaması pahalıdır;
# bu nedenle sklearn, iteratif yöntemler kullanır.

# XᵀX → (p+1, p+1) = (8, 8) boyutunda bilgi matrisi
XtX     = X_aug.T @ X_aug   # boyut: (p+1, p+1)
# Xᵀy → (p+1,) boyutunda hedef vektörü projeksiyonu
Xty     = X_aug.T @ y_train_s  # boyut: (p+1,)
# linalg.solve: tersini almak yerine lineer sistemi çözer (sayısal kararlılık)
beta_analytic = np.linalg.solve(XtX, Xty)

print("\n  β_EKK (Analitik Normal Denklem):")
for name, val_ in zip(param_names, beta_analytic):
    print(f"    {name:<30} {val_:>10.6f}")


# ═══════════════════════════════════════════════════════════
#  ADIM 2A — EKK (sklearn LinearRegression)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2A — EKK (sklearn LinearRegression)")

# sklearn, iç olarak SVD tabanlı çözüm kullanır (sayısal kararlılık).
# Bu yöntem Normal Denklem ile eşdeğer sonuç üretir.
ols_model = LinearRegression()
ols_model.fit(X_train, y_train_s)

# intercept ve coef_ birleştirilerek tek bir β vektörü oluşturulur.
beta_ols = np.concatenate([[ols_model.intercept_], ols_model.coef_])

print("\n  β_EKK (sklearn):")
for name, b in zip(param_names, beta_ols):
    print(f"    {name:<30} {b:>10.6f}")

# Tahminler (ölçeklenmiş uzayda)
y_pred_train_ols = ols_model.predict(X_train)
y_pred_val_ols   = ols_model.predict(X_val_s)
y_pred_test_ols  = ols_model.predict(X_test_s)

# Ölçeği geri al: z-skor uzayından orijinal BMI (kg/m²) birimine.
# Metrikler bu orijinal ölçekte hesaplanır ve raporlanır.
y_pred_val_orig_ols  = scaler_y.inverse_transform(y_pred_val_ols.reshape(-1,1)).ravel()
y_pred_test_orig_ols = scaler_y.inverse_transform(y_pred_test_ols.reshape(-1,1)).ravel()


# ═══════════════════════════════════════════════════════════
#  ADIM 2B — MLE (Log-Likelihood Maksimizasyonu)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2B — MLE (Log-Likelihood Maksimizasyonu)")

"""
Model: y_i = X_i β + ε_i ,  ε_i ~ N(0, σ²)

Log-Likelihood:
  L(β,σ) = -n/2·ln(2π) - n/2·ln(σ²) - 1/(2σ²)·Σ(y_i - X_i β)²

Maximizasyon ↔ negatif log-likelihood minimizasyonu
Parametre vektörü θ = [β₀, β₁, ..., βₚ, ln(σ)]  (σ>0 için log-parametre)
"""

def neg_log_likelihood(theta):
    """
    Negatif log-likelihood fonksiyonu (minimize edilecek).

    Model: y ~ N(Xβ, σ²I)

    Parametreler
    ------------
    theta : ndarray, şekil (p+2,)
        Son eleman ln(σ)'dır; σ>0 kısıtı doğal olarak sağlanır.
        İlk p+1 eleman β katsayı vektörüdür.

    Döndürür
    --------
    float
        Negatif log-likelihood değeri (L-BFGS-B bu değeri minimize eder).

    Matematiksel Formül
    -------------------
    NLL = n/2·ln(2π) + n·ln(σ) + 1/(2σ²)·‖y - Xβ‖²
    """
    beta  = theta[:-1]                     # β katsayıları: (p+1,)
    log_s = theta[-1]                      # ln(σ): σ>0 garantisi
    sigma = np.exp(log_s)                  # σ: hata standart sapması
    # Artık vektörü: (n,) — boyut: (536,)
    resid = y_train_s - X_aug @ beta      # (n,)
    # NLL = n/2·ln(2π) + n·ln(σ) + 1/(2σ²)·‖resid‖²
    nll   = (n/2) * np.log(2*np.pi)       \
          + n * log_s                      \
          + (1/(2*sigma**2)) * np.sum(resid**2)
    return nll

def neg_log_likelihood_gradient(theta):
    """
    NLL fonksiyonunun analitik gradyanı.

    Parametreler
    ------------
    theta : ndarray, şekil (p+2,)
        [β vektörü | ln(σ)]

    Döndürür
    --------
    ndarray, şekil (p+2,)
        Gradyan vektörü [∂NLL/∂β | ∂NLL/∂(ln σ)]

    Matematiksel Formüller
    ----------------------
    ∂NLL/∂β        = -1/σ² · Xᵀ·resid
    ∂NLL/∂(ln σ)   = n - Σresid²/σ²
    """
    beta  = theta[:-1]
    log_s = theta[-1]
    sigma = np.exp(log_s)
    resid = y_train_s - X_aug @ beta

    # ∂NLL/∂β = -1/σ² · Xᵀ·resid
    # X_aug.T → (p+1, n), resid → (n,) → sonuç: (p+1,)
    grad_beta  = -(1/sigma**2) * (X_aug.T @ resid)

    # ∂NLL/∂(ln σ) = n - Σresid²/σ²
    # Bu türev, zincir kuralı ile ∂NLL/∂σ · dσ/d(ln σ) = σ·∂NLL/∂σ'dan gelir.
    grad_logs  = n - np.sum(resid**2) / sigma**2

    return np.concatenate([grad_beta, [grad_logs]])

# Başlangıç noktası: EKK çözümü + σ=1
# EKK çözümü MLE'ye çok yakın olduğundan, optimizasyonun
# global minimumu daha az iterasyonda bulması beklenir.
theta0      = np.concatenate([beta_analytic, [0.0]])   # ln(1)=0
print(f"\n  Optimizasyon (L-BFGS-B) başlıyor...")
print(f"  Parametre sayısı: {len(theta0)} (β×{p+1} + ln_σ×1)")

result = minimize(
    neg_log_likelihood,
    theta0,
    jac=neg_log_likelihood_gradient,
    method="L-BFGS-B",
    options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10}
)

print(f"  Optimizasyon durumu : {'✅ BAŞARILI' if result.success else '❌ BAŞARISIZ'}")
print(f"  İterasyon sayısı    : {result.nit}")
print(f"  Final NLL değeri    : {result.fun:.6f}")

beta_mle   = result.x[:-1]        # β tahminleri: (p+1,)
sigma_mle  = np.exp(result.x[-1]) # σ tahmini: yeniden üstel dönüşüm

print(f"\n  σ_MLE (hata std. sapması) = {sigma_mle:.6f}")
print("\n  β_MLE:")
for name, b in zip(param_names, beta_mle):
    print(f"    {name:<30} {b:>10.6f}")

# MLE tahminleri: X_aug @ beta_mle ile elde edilir.
# X_aug boyutu: (n, p+1) = (536, 8)
# beta_mle boyutu: (p+1,) = (8,)
# Sonuç: (n,) = (536,)
y_pred_train_mle = X_aug @ beta_mle

# Val ve Test için de bias sütunu eklenir (X_aug yapısı tekrarlanır).
X_val_aug  = np.hstack([np.ones((len(X_val_s),  1)), X_val_s])   # (116, 8)
X_test_aug = np.hstack([np.ones((len(X_test_s), 1)), X_test_s])  # (116, 8)
y_pred_val_mle   = X_val_aug  @ beta_mle
y_pred_test_mle  = X_test_aug @ beta_mle

# Ölçeği geri al: orijinal BMI birimine dönüştür.
y_pred_val_orig_mle  = scaler_y.inverse_transform(y_pred_val_mle.reshape(-1,1)).ravel()
y_pred_test_orig_mle = scaler_y.inverse_transform(y_pred_test_mle.reshape(-1,1)).ravel()


# ═══════════════════════════════════════════════════════════
#  ADIM 3 — KATSAYI KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════
section_header("ADIM 3 — KATSAYI KARŞILAŞTIRMASI")

# Teorik beklenti: β_EKK = β_MLE (normal hata varsayımı altında).
# Bu, ∂ ln L / ∂β = 0 ifadesinin Normal Denklem'e eşdeğer olmasından
# kaynaklanır. FINAL_REPORT.md §3.2'de analitik kanıt yer almaktadır.
#
# Sayısal fark makine hassasiyeti (~2.22e-16) mertebesinde olmalıdır.
# FINAL_REPORT.md §3.3: max|β_EKK − β_MLE| ≈ 3.16 × 10⁻¹⁶ ✅
diff     = np.abs(beta_ols - beta_mle)
rel_diff = np.where(np.abs(beta_ols) > 1e-9,
                    diff / np.abs(beta_ols) * 100, 0.0)

df_compare = pd.DataFrame({
    "Parametre"        : param_names,
    "β_EKK (sklearn)"  : beta_ols,
    "β_MLE (optim.)"   : beta_mle,
    "β_Analitik (NEQ)" : beta_analytic,
    "Mutlak Fark"      : diff,
    "Göreli Fark (%)."  : rel_diff,
})

print(f"\n{df_compare.to_string(index=False, float_format=lambda x: f'{x:>11.7f}')}")
print(f"\n  Maks. mutlak fark  : {diff.max():.2e}")
print(f"  Ort. mutlak fark   : {diff.mean():.2e}")
print(f"  EKK ↔ MLE uyumu    : {'✅ MÜKEMMEL (fark < 1e-4)' if diff.max() < 1e-4 else '⚠️ FARK VAR'}")


# ═══════════════════════════════════════════════════════════
#  ADIM 4 — METRİKLER
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4 — MODEL METRİKLERİ")

def metrics(y_true, y_pred, label):
    """
    Regresyon performans metriklerini hesaplar.

    Parametreler
    ------------
    y_true : ndarray
        Gerçek BMI değerleri (orijinal ölçek, kg/m²).
    y_pred : ndarray
        Tahmin edilen BMI değerleri (orijinal ölçek).
    label : str
        Model ve set adı (örn. "EKK — Validation").

    Döndürür
    --------
    dict
        {"Model": label, "MSE": ..., "RMSE": ..., "MAE": ..., "R²": ...}

    Notlar
    ------
    - RMSE ve MAE orijinal BMI birimiyle (kg/m²) yorumlanır.
    - R²=0 → model ortalama tahmini kadar iyi;
      R²=1 → mükemmel tahmin. R²=0.431 (Test) → FINAL_REPORT.md §3.5.
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"Model": label, "MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

rows = [
    metrics(y_val,  y_pred_val_orig_ols,  "EKK — Validation"),
    metrics(y_test, y_pred_test_orig_ols, "EKK — Test"),
    metrics(y_val,  y_pred_val_orig_mle,  "MLE — Validation"),
    metrics(y_test, y_pred_test_orig_mle, "MLE — Test"),
]
df_metrics = pd.DataFrame(rows)
print(f"\n{df_metrics.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")

df_compare.to_csv(os.path.join(OUTPUT_DIR, "beta_comparison.csv"), index=False)
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"),         index=False)
print(f"\n  CSV'ler kaydedildi → {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 1 — KATSAYI KARŞILAŞTIRMA (grouped bar)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 1 — KATSAYI KARŞILAŞTIRMA")

# Bu grafik β_EKK ile β_MLE katsayılarının neredeyse üst üste
# bindiğini görsel olarak kanıtlar. Log-ölçekli sağ panel,
# farkların makine hassasiyeti mertebesinde (<1e-15) olduğunu gösterir.
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 2 — β Katsayıları: EKK vs. MLE Karşılaştırması",
             fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)

x     = np.arange(len(param_names))
width = 0.3

# Panel A: değerler
ax = axes[0]
ax.set_facecolor(CARD_BG)
b1 = ax.bar(x - width/2, beta_ols, width, label="β_EKK",
            color=ACCENT1, edgecolor=DARK_BG, linewidth=1.2, alpha=0.85)
b2 = ax.bar(x + width/2, beta_mle, width, label="β_MLE",
            color=ACCENT3, edgecolor=DARK_BG, linewidth=1.2, alpha=0.85)
ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(param_names, rotation=35, ha="right", fontsize=8)
ax.set_title("β Değerleri (EKK vs MLE)", fontsize=11, pad=10)
ax.set_ylabel("Katsayı Değeri", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y")

# Panel B: mutlak fark (log scale)
# Fark neredeyse sıfır (makine hassasiyeti) → iki çubuk üst üste biner.
# FINAL_REPORT.md §3.3'te raporlanan 3.16e-16 değeri bu grafiğin kaynağıdır.
ax2 = axes[1]
ax2.set_facecolor(CARD_BG)
colors_diff = [ACCENT2 if d < 1e-6 else ACCENT5 if d < 1e-4 else ACCENT3 for d in diff]
bars = ax2.bar(x, diff, color=colors_diff, edgecolor=DARK_BG, linewidth=1.2)
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(param_names, rotation=35, ha="right", fontsize=8)
ax2.set_title("|β_EKK − β_MLE| (log ölçeği)", fontsize=11, pad=10)
ax2.set_ylabel("Mutlak Fark (log)", fontsize=10)
ax2.axhline(1e-6, color=ACCENT2, linewidth=1.5, linestyle="--",
            alpha=0.8, label="1e-6 eşiği")
ax2.axhline(1e-4, color=ACCENT5, linewidth=1.5, linestyle="--",
            alpha=0.8, label="1e-4 eşiği")
ax2.legend(fontsize=8)
ax2.grid(axis="y")

plt.tight_layout()
save_fig("01_beta_comparison.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 2 — LOG-LİKELİHOOD YÜZEYI (2D slice: β₀ vs β₁)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 2 — LOG-LİKELİHOOD YÜZEYI")

# Log-likelihood yüzeyi iki boyutlu bir dilim şeklinde görselleştirilir:
# β₀ (intercept) ve β₁ (ilk özellik — Pregnancies) eksenleri sabit tutulurken
# diğer β'lar MLE değerlerinde sabit bırakılır.
# EKK ve MLE nokta tahminlerinin çakıştığı görülür → teorik kanıt.
b0_range = np.linspace(beta_mle[0] - 0.5, beta_mle[0] + 0.5, 80)
b1_range = np.linspace(beta_mle[1] - 1.0, beta_mle[1] + 1.0, 80)
B0, B1   = np.meshgrid(b0_range, b1_range)

# Diğer β'ları MLE değerinde sabitle
LL_grid = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        beta_tmp = beta_mle.copy()
        beta_tmp[0] = B0[i, j]
        beta_tmp[1] = B1[i, j]
        theta_tmp = np.concatenate([beta_tmp, [np.log(sigma_mle)]])
        # neg_log_likelihood'in negatifi = Log-Likelihood
        LL_grid[i, j] = -neg_log_likelihood(theta_tmp)  # Log-Likelihood (negatif NLL)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Log-Likelihood Yüzeyi (β₀ × β₁ Dilimi, Diğerleri Sabit)",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

# Contour
ax = axes[0]
ax.set_facecolor(CARD_BG)
cf = ax.contourf(B0, B1, LL_grid, levels=40, cmap="plasma")
cs = ax.contour(B0,  B1, LL_grid, levels=12,
                colors=TEXT_COLOR, alpha=0.3, linewidths=0.6)
ax.plot(beta_mle[0], beta_mle[1], "o",
        color=ACCENT2, markersize=10, markeredgecolor=DARK_BG,
        markeredgewidth=2, label="MLE Optimumu", zorder=5)
ax.plot(beta_ols[0], beta_ols[1], "^",
        color=ACCENT3, markersize=10, markeredgecolor=DARK_BG,
        markeredgewidth=2, label="EKK Çözümü", zorder=5)
cbar = plt.colorbar(cf, ax=ax)
cbar.set_label("Log-Likelihood", color=TEXT_COLOR, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax.set_xlabel("β₀ (intercept)", fontsize=10)
ax.set_ylabel(f"β₁ ({FEATURES[0]})", fontsize=10)
ax.set_title("Kontur Haritası", fontsize=11, pad=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 3D yüzey (imshow + smooth)
ax2 = axes[1]
ax2.set_facecolor(CARD_BG)
im = ax2.imshow(LL_grid, aspect="auto",
                extent=[b0_range.min(), b0_range.max(),
                        b1_range.min(), b1_range.max()],
                origin="lower", cmap="inferno", interpolation="bilinear")
ax2.plot(beta_mle[0], beta_mle[1], "o",
         color=ACCENT2, markersize=10, label="MLE",
         zorder=5, markeredgecolor=DARK_BG, markeredgewidth=2)
ax2.plot(beta_ols[0], beta_ols[1], "^",
         color=ACCENT3, markersize=10, label="EKK",
         zorder=5, markeredgecolor=DARK_BG, markeredgewidth=2)
cbar2 = plt.colorbar(im, ax=ax2)
cbar2.set_label("Log-Likelihood", color=TEXT_COLOR, fontsize=9)
cbar2.ax.yaxis.set_tick_params(color=TEXT_COLOR)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax2.set_xlabel("β₀ (intercept)", fontsize=10)
ax2.set_ylabel(f"β₁ ({FEATURES[0]})", fontsize=10)
ax2.set_title("Isı Haritası", fontsize=11, pad=10)
ax2.legend(fontsize=9)

plt.tight_layout()
save_fig("02_loglikelihood_surface.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 3 — TAHMİN vs GERÇEK (val + test)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 3 — TAHMİN vs GERÇEK")

# Mükemmel tahmin ekseni boyunca noktaların toplanması gerekir.
# Dağılma, modelin hata payını gösterir.
# Orijinal BMI ölçeğinde görselleştirilir (FINAL_REPORT.md §3.5).
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Tahmin vs. Gerçek BMI Değerleri (Orijinal Ölçek)",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

for ax, y_true, y_pred_ols, y_pred_mle, title in [
        (axes[0], y_val,  y_pred_val_orig_ols,  y_pred_val_orig_mle,  "Validation Seti"),
        (axes[1], y_test, y_pred_test_orig_ols, y_pred_test_orig_mle, "Test Seti"), ]:
    ax.set_facecolor(CARD_BG)
    mn = min(y_true.min(), y_pred_ols.min(), y_pred_mle.min()) - 2
    mx = max(y_true.max(), y_pred_ols.max(), y_pred_mle.max()) + 2
    ax.plot([mn, mx], [mn, mx], "--", color=TEXT_COLOR,
            linewidth=1.5, alpha=0.6, label="Mükemmel tahmin")
    ax.scatter(y_true, y_pred_ols, color=ACCENT1, alpha=0.65,
               s=30, edgecolors="none", label="EKK")
    ax.scatter(y_true, y_pred_mle, color=ACCENT3, alpha=0.4,
               s=30, edgecolors="none", marker="^", label="MLE")
    r2_ols = r2_score(y_true, y_pred_ols)
    r2_mle = r2_score(y_true, y_pred_mle)
    ax.text(0.04, 0.94, f"R²_EKK = {r2_ols:.4f}\nR²_MLE = {r2_mle:.4f}",
            transform=ax.transAxes, fontsize=9,
            color=TEXT_COLOR, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                      edgecolor=GRID_COLOR, alpha=0.8))
    ax.set_xlabel("Gerçek BMI", fontsize=10)
    ax.set_ylabel("Tahmin Edilen BMI", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)

plt.tight_layout()
save_fig("03_predicted_vs_actual.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 4 — KALİNTI (RESIDUAL) ANALİZİ
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 4 — KALİNTI ANALİZİ")

# Kalıntı analizi, MLE'nin temel varsayımını kontrol eder:
# ε_i ~ N(0, σ²) — kalıntılar sıfır ortalama ve sabit varyansla
# normal dağılmalıdır. Q-Q plot normallik derecesini R² ile sayısal olarak ölçer.
resid_train_ols = y_train_s    - y_pred_train_ols
resid_train_mle = y_train_s    - y_pred_train_mle
resid_val_ols   = y_val_s      - y_pred_val_ols
resid_val_mle   = y_val_s      - y_pred_val_mle

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 2 — Kalıntı Analizi: EKK ve MLE",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# Satır 1: EKK kalıntıları
titles_row1 = [
    ("EKK — Kalıntı Histogramı (Train)", resid_train_ols, ACCENT1),
    ("EKK — Kalıntı Dağılımı (Train)",   resid_train_ols, ACCENT1),
    ("EKK — Q-Q Plot (Train)",            resid_train_ols, ACCENT1),
    ("EKK — Kalıntı vs Tahmin (val)",    None,            ACCENT1),
]

for col, (title, resid, color) in enumerate(titles_row1):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(CARD_BG)
    if col == 0:   # Histogram: normalliği histogram + N(μ,σ) eğrisi ile göster
        ax.hist(resid, bins=30, color=color, edgecolor=DARK_BG,
                linewidth=0.8, alpha=0.85, density=True)
        xr = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()),
                color=ACCENT5, linewidth=2, label="N(μ,σ)")
        ax.set_xlabel("Kalıntı"); ax.set_ylabel("Yoğunluk")
        ax.legend(fontsize=8)
    elif col == 1:  # Kalıntı zaman serisi: rastgele yapı olmalı (homokedestedisite)
        ax.scatter(range(len(resid)), resid, color=color, alpha=0.5, s=15)
        ax.axhline(0, color=ACCENT3, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Gözlem İndeksi"); ax.set_ylabel("Kalıntı")
    elif col == 2:  # Q-Q plot: noktaların çizgi üzerinde olması normallik gösterir
        (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
        ax.scatter(osm, osr, color=color, alpha=0.6, s=12)
        ax.plot(osm, slope*np.array(osm)+intercept,
                color=ACCENT5, linewidth=2, label=f"R²={r**2:.4f}")
        ax.set_xlabel("Teorik Kantiller"); ax.set_ylabel("Örnek Kantilleri")
        ax.legend(fontsize=8)
    elif col == 3:  # Kalıntı vs. tahmin: funnel shape yoksa homokedestedisite var
        ax.scatter(y_pred_val_ols, resid_val_ols, color=color, alpha=0.6, s=18)
        ax.axhline(0, color=ACCENT3, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Tahmin (EKK)"); ax.set_ylabel("Kalıntı")
    ax.set_title(title, fontsize=9, pad=6)
    ax.grid(True, alpha=0.3)

# Satır 2: MLE kalıntıları (EKK ile neredeyse özdeş — teorik tutarlılık kanıtı)
titles_row2 = [
    ("MLE — Kalıntı Histogramı (Train)", resid_train_mle, ACCENT3),
    ("MLE — Kalıntı Dağılımı (Train)",   resid_train_mle, ACCENT3),
    ("MLE — Q-Q Plot (Train)",            resid_train_mle, ACCENT3),
    ("MLE — Kalıntı vs Tahmin (val)",    None,            ACCENT3),
]

for col, (title, resid, color) in enumerate(titles_row2):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor(CARD_BG)
    if col == 0:
        ax.hist(resid, bins=30, color=color, edgecolor=DARK_BG,
                linewidth=0.8, alpha=0.85, density=True)
        xr = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()),
                color=ACCENT5, linewidth=2, label="N(μ,σ)")
        ax.set_xlabel("Kalıntı"); ax.set_ylabel("Yoğunluk")
        ax.legend(fontsize=8)
    elif col == 1:
        ax.scatter(range(len(resid)), resid, color=color, alpha=0.5, s=15)
        ax.axhline(0, color=ACCENT1, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Gözlem İndeksi"); ax.set_ylabel("Kalıntı")
    elif col == 2:
        (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
        ax.scatter(osm, osr, color=color, alpha=0.6, s=12)
        ax.plot(osm, slope*np.array(osm)+intercept,
                color=ACCENT5, linewidth=2, label=f"R²={r**2:.4f}")
        ax.set_xlabel("Teorik Kantiller"); ax.set_ylabel("Örnek Kantilleri")
        ax.legend(fontsize=8)
    elif col == 3:
        ax.scatter(y_pred_val_mle, resid_val_mle, color=color, alpha=0.6, s=18)
        ax.axhline(0, color=ACCENT1, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Tahmin (MLE)"); ax.set_ylabel("Kalıntı")
    ax.set_title(title, fontsize=9, pad=6)
    ax.grid(True, alpha=0.3)

save_fig("04_residual_analysis.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 5 — ÖZELLİK ÖNEMİ (katsayı büyüklükleri)
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 5 — ÖZELLİK ÖNEMİ (Katsayı Büyüklükleri)")

# Standardize edilmiş katsayılar doğrudan karşılaştırılabilir:
# Daha büyük |β| → o özellik BMI tahmininde daha baskındır.
# FINAL_REPORT.md §3.4: SkinThickness (β≈+0.508) birinci sıradadır.
# ÖNEMLİ: SkinThickness'ın yüksek katsayısı, Faz 1'deki yüksek
# medyan impütasyon oranından (%29.6) etkilenmiş olabilir.

# intercept hariç
betas_ols_feat = beta_ols[1:]
betas_mle_feat = beta_mle[1:]
# |β|'ya göre büyükten küçüğe sırala
feat_order     = np.argsort(np.abs(betas_ols_feat))[::-1]
labels_sorted  = [FEATURES[i] for i in feat_order]
ols_sorted     = betas_ols_feat[feat_order]
mle_sorted     = betas_mle_feat[feat_order]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Özellik Önemi: Standardize Edilmiş Katsayılar",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

for ax, betas, label, color, title in [
        (axes[0], ols_sorted, "β_EKK", ACCENT1, "EKK Katsayıları (büyüklüğe göre)"),
        (axes[1], mle_sorted, "β_MLE", ACCENT3, "MLE Katsayıları (büyüklüğe göre)"),
]:
    ax.set_facecolor(CARD_BG)
    bar_colors = [ACCENT2 if b > 0 else ACCENT3 for b in betas]
    bars = ax.barh(range(len(labels_sorted)), betas,
                  color=bar_colors, edgecolor=DARK_BG, linewidth=1, alpha=0.85)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=10)
    ax.set_xlabel("Katsayı Değeri (z-skorda)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    for bar, val in zip(bars, betas):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}",
                ha="left" if val >= 0 else "right",
                va="center", fontsize=7.5, color=TEXT_COLOR)
    p_pos = mpatches.Patch(color=ACCENT2, label="Pozitif etki")
    p_neg = mpatches.Patch(color=ACCENT3, label="Negatif etki")
    ax.legend(handles=[p_pos, p_neg], fontsize=8)
    ax.grid(axis="x")

plt.tight_layout()
save_fig("05_feature_importance.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 6 — METRİKLER DASHBOARD
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 6 — METRİKLER DASHBOARD")

metric_values = {
    "RMSE": [
        np.sqrt(mean_squared_error(y_val,  y_pred_val_orig_ols)),
        np.sqrt(mean_squared_error(y_test, y_pred_test_orig_ols)),
        np.sqrt(mean_squared_error(y_val,  y_pred_val_orig_mle)),
        np.sqrt(mean_squared_error(y_test, y_pred_test_orig_mle)),
    ],
    "MAE": [
        mean_absolute_error(y_val,  y_pred_val_orig_ols),
        mean_absolute_error(y_test, y_pred_test_orig_ols),
        mean_absolute_error(y_val,  y_pred_val_orig_mle),
        mean_absolute_error(y_test, y_pred_test_orig_mle),
    ],
    "R²": [
        r2_score(y_val,  y_pred_val_orig_ols),
        r2_score(y_test, y_pred_test_orig_ols),
        r2_score(y_val,  y_pred_val_orig_mle),
        r2_score(y_test, y_pred_test_orig_mle),
    ],
}
bar_labels = ["EKK-Val", "EKK-Test", "MLE-Val", "MLE-Test"]
bar_colors = [ACCENT1, ACCENT6, ACCENT3, "#f47067"]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Model Performans Metrikleri — EKK vs. MLE (Orijinal Ölçek)",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

for ax, (metric, vals) in zip(axes, metric_values.items()):
    ax.set_facecolor(CARD_BG)
    bars = ax.bar(bar_labels, vals, color=bar_colors,
                  edgecolor=DARK_BG, linewidth=1.2, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax.set_title(metric, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    if metric == "R²":
        ax.set_ylim(0, 1)

plt.tight_layout()
save_fig("06_metrics_dashboard.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 7 — TEORİK KARŞILAŞTIRMA: NEDEN β_EKK ≈ β_MLE?
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 7 — NEDEN EKK ≈ MLE? (Teorik Görselleştirme)")

# Bu grafik, normal hata varsayımı altında EKK=MLE eşitliğini
# sezgisel olarak açıklar. Hata dağılımı N(0,σ²) ise
# log-likelihood maksimizasyonu, artık kare minimizasyonuyla eşdeğerdir.
# Formül: lnP(ε) ∝ -(ε²/2σ²) → maximize et = (ε²) minimize et
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Normal Hata Varsayımı Altında EKK = MLE (Teorik Kanıt)",
             fontsize=13, fontweight="bold", color=TEXT_COLOR)

# Panel 1: Normal olasılık yoğunluğu
ax = axes[0]
ax.set_facecolor(CARD_BG)
eps = np.linspace(-4, 4, 300)
ax.plot(eps, stats.norm.pdf(eps, 0, 1),
        color=ACCENT1, linewidth=2.5, label="N(0,1)")
for x_pt, clr in [(-1, ACCENT2), (0, ACCENT5), (1.5, ACCENT3)]:
    yp = stats.norm.pdf(x_pt, 0, 1)
    ax.plot([x_pt, x_pt], [0, yp], "--", color=clr, linewidth=1.5, alpha=0.7)
    ax.scatter([x_pt], [yp], color=clr, s=60, zorder=5)
    ax.text(x_pt, yp + 0.01, f"ε={x_pt}", ha="center",
            fontsize=8, color=clr)
ax.fill_between(eps, stats.norm.pdf(eps, 0, 1), alpha=0.12, color=ACCENT1)
ax.set_xlabel("Hata Terimi ε", fontsize=10)
ax.set_ylabel("Olasılık Yoğunluğu f(ε)", fontsize=10)
ax.set_title("Hata Dağılımı Varsayımı: ε ~ N(0, σ²)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.05,
        "MLE: β* = arg max Σ log f(εᵢ)\nEKK: β* = arg min Σ εᵢ²\n→ Normal dist. altında EŞİT!",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK_BG,
                  edgecolor=ACCENT4, alpha=0.9))

# Panel 2: Optimizasyon yörüngesi (NLL değişimi)
# β₀'ın farklı değerleri için NLL'yi hesaplar; minimum EKK ile çakışır.
ax2 = axes[1]
ax2.set_facecolor(CARD_BG)
# β₀ parametresinin farklı değerleri için NLL hesapla
b0_sweep = np.linspace(beta_mle[0] - 0.8, beta_mle[0] + 0.8, 200)
nll_vals  = []
for b0_val in b0_sweep:
    bt         = beta_mle.copy(); bt[0] = b0_val
    theta_tmp  = np.concatenate([bt, [np.log(sigma_mle)]])
    nll_vals.append(neg_log_likelihood(theta_tmp))

ax2.plot(b0_sweep, nll_vals, color=ACCENT1, linewidth=2.5, label="NLL(β₀)")
ax2.axvline(beta_mle[0], color=ACCENT2, linewidth=2,
            linestyle="--", label=f"MLE opt: β₀={beta_mle[0]:.4f}")
ax2.axvline(beta_ols[0], color=ACCENT3, linewidth=2,
            linestyle=":",  label=f"EKK opt: β₀={beta_ols[0]:.4f}")
ax2.scatter([beta_mle[0]], [min(nll_vals)], color=ACCENT2, s=100, zorder=5)
ax2.set_xlabel("β₀ (intercept)", fontsize=10)
ax2.set_ylabel("Negatif Log-Likelihood", fontsize=10)
ax2.set_title("NLL Fonksiyonu — β₀ Boyutu (Diğerleri Sabit)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig("07_theory_comparison.png")


# ═══════════════════════════════════════════════════════════
#  GRAFİK 8 — ÖZET DASHBOARD
# ═══════════════════════════════════════════════════════════
section_header("GRAFİK 8 — ÖZET DASHBOARD")

fig = plt.figure(figsize=(22, 13))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 2 — EKK vs. MLE ÖZET DASHBOARD (BMI Regresyonu)",
             fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.00)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# Panel 1 — Katsayı scatter
# β_EKK ve β_MLE'nin y=x doğrusu üzerinde toplanması mükemmel uyumu kanıtlar.
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD_BG)
ax1.scatter(beta_ols[1:], beta_mle[1:], color=ACCENT1,
            s=80, edgecolors=DARK_BG, linewidths=1.2, zorder=5)
mn_b = min(beta_ols[1:].min(), beta_mle[1:].min()) - 0.05
mx_b = max(beta_ols[1:].max(), beta_mle[1:].max()) + 0.05
ax1.plot([mn_b, mx_b], [mn_b, mx_b], "--",
         color=ACCENT5, linewidth=1.5, label="β_EKK = β_MLE")
for i, feat in enumerate(FEATURES):
    ax1.annotate(feat, (beta_ols[i+1], beta_mle[i+1]),
                 fontsize=6, color=TEXT_COLOR, ha="left",
                 textcoords="offset points", xytext=(4, 2))
ax1.set_xlabel("β_EKK", fontsize=9); ax1.set_ylabel("β_MLE", fontsize=9)
ax1.set_title("β_EKK vs β_MLE\n(Mükemmel Uyum Beklenir)", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Panel 2 — R² karşılaştırması
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_BG)
r2_vals  = [r2_score(y_val, y_pred_val_orig_ols),
            r2_score(y_test,y_pred_test_orig_ols),
            r2_score(y_val, y_pred_val_orig_mle),
            r2_score(y_test,y_pred_test_orig_mle)]
ax2.bar(["EKK-Val","EKK-Test","MLE-Val","MLE-Test"],
        r2_vals, color=[ACCENT1,ACCENT6,ACCENT3,"#f47067"],
        edgecolor=DARK_BG, linewidth=1)
ax2.set_ylim(0, 1); ax2.set_ylabel("R² Skoru"); ax2.grid(axis="y")
ax2.set_title("R² — Val & Test Setleri", fontsize=10)
for i, v in enumerate(r2_vals):
    ax2.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8.5,
             fontweight="bold", color=TEXT_COLOR)

# Panel 3 — Mutlak fark bar
# Log ölçeğinde tüm farkların makine hassasiyeti mertebesinde olduğu görülür.
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(CARD_BG)
ax3.bar(range(len(param_names)), diff, color=ACCENT4,
        edgecolor=DARK_BG, linewidth=1)
ax3.set_yscale("log")
ax3.set_xticks(range(len(param_names)))
ax3.set_xticklabels(param_names, rotation=40, ha="right", fontsize=7)
ax3.set_title("|β_EKK − β_MLE| (log ölçeği)", fontsize=10)
ax3.set_ylabel("Mutlak Fark (log)"); ax3.grid(axis="y")

# Panel 4 — Kalıntı histogram (her iki model)
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(CARD_BG)
ax4.hist(resid_val_ols, bins=20, color=ACCENT1, alpha=0.7,
         edgecolor=DARK_BG, label="EKK-Val", density=True)
ax4.hist(resid_val_mle, bins=20, color=ACCENT3, alpha=0.5,
         edgecolor=DARK_BG, label="MLE-Val", density=True)
ax4.set_xlabel("Kalıntı (z-skor)"); ax4.set_ylabel("Yoğunluk")
ax4.set_title("Kalıntı Histogramları (Val)", fontsize=10)
ax4.legend(fontsize=8); ax4.grid(axis="y")

# Panel 5 — Tahmin vs Gerçek (test)
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(CARD_BG)
mn_t = min(y_test.min(), y_pred_test_orig_ols.min()) - 1
mx_t = max(y_test.max(), y_pred_test_orig_ols.max()) + 1
ax5.plot([mn_t, mx_t], [mn_t, mx_t], "--", color=TEXT_COLOR,
         linewidth=1.5, alpha=0.6)
ax5.scatter(y_test, y_pred_test_orig_ols, color=ACCENT1,
            alpha=0.6, s=25, label="EKK")
ax5.scatter(y_test, y_pred_test_orig_mle, color=ACCENT3,
            alpha=0.4, s=25, marker="^", label="MLE")
ax5.set_xlabel("Gerçek BMI"); ax5.set_ylabel("Tahmin BMI")
ax5.set_title("Tahmin vs. Gerçek (Test)", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# Panel 6 — RMSE & MAE
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD_BG)
x_m = np.arange(2)
rmse_ols = [np.sqrt(mean_squared_error(y_val,y_pred_val_orig_ols)),
             np.sqrt(mean_squared_error(y_test,y_pred_test_orig_ols))]
rmse_mle = [np.sqrt(mean_squared_error(y_val,y_pred_val_orig_mle)),
             np.sqrt(mean_squared_error(y_test,y_pred_test_orig_mle))]
ax6.bar(x_m - 0.2, rmse_ols, 0.35, color=ACCENT1, label="EKK RMSE",
        edgecolor=DARK_BG)
ax6.bar(x_m + 0.2, rmse_mle, 0.35, color=ACCENT3, label="MLE RMSE",
        edgecolor=DARK_BG)
ax6.set_xticks(x_m); ax6.set_xticklabels(["Validation", "Test"])
ax6.set_ylabel("RMSE (orijinal ölçek)"); ax6.grid(axis="y")
ax6.set_title("RMSE Karşılaştırması", fontsize=10)
ax6.legend(fontsize=8)
for i, (v1, v2) in enumerate(zip(rmse_ols, rmse_mle)):
    ax6.text(i-0.2, v1+0.05, f"{v1:.3f}", ha="center",
             fontsize=8, color=TEXT_COLOR, fontweight="bold")
    ax6.text(i+0.2, v2+0.05, f"{v2:.3f}", ha="center",
             fontsize=8, color=TEXT_COLOR, fontweight="bold")

save_fig("08_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════
#  SONUÇ
# ═══════════════════════════════════════════════════════════
section_header("FAZ 2 — TAMAMLANDI ✅")

r2_ols_test = r2_score(y_test, y_pred_test_orig_ols)
r2_mle_test = r2_score(y_test, y_pred_test_orig_mle)
rmse_ols_test = np.sqrt(mean_squared_error(y_test, y_pred_test_orig_ols))
rmse_mle_test = np.sqrt(mean_squared_error(y_test, y_pred_test_orig_mle))

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                    FAZ 2 ÖZET                               │
  ├──────────────────────────────────────────────────────────────┤
  │  Hedef Değişken     : BMI                                   │
  │  Özellik Sayısı     : {p} + 1 (intercept)                   │
  │  Train Boyutu       : {n}                                   │
  ├──────────────────────────────────────────────────────────────┤
  │  KATSAYI KARŞILAŞTIRMASI                                    │
  │  Maks. mutlak fark  : {diff.max():.2e}                     │
  │  EKK ↔ MLE uyumu   : {'✅ MÜKEMMEL' if diff.max() < 1e-4 else '⚠️ KONTROL ET'}                        │
  ├──────────────────────────────────────────────────────────────┤
  │  TEST SETİ SONUÇLARI                                        │
  │  EKK → R²={r2_ols_test:.4f}  RMSE={rmse_ols_test:.4f}        │
  │  MLE → R²={r2_mle_test:.4f}  RMSE={rmse_mle_test:.4f}        │
  ├──────────────────────────────────────────────────────────────┤
  │  TEORİK DURUM                                               │
  │  Normal hata varsayımı altında:                             │
  │    log L(β) = -1/(2σ²)·Σresid² + const                     │
  │    → MLE maksimizasyonu = EKK minimizasyonu                  │
  │  ∴ β_EKK = β_MLE  (analitik olarak kanıtlanmış)            │
  ├──────────────────────────────────────────────────────────────┤
  │  Kaydedilen Dosyalar: {OUTPUT_DIR}/                          │
  │    • beta_comparison.csv  • metrics.csv                    │
  │  Grafikler (8 adet) : {OUTPUT_DIR}/*.png                    │
  └──────────────────────────────────────────────────────────────┘
""")
