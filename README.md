# Pima Indians Diabetes — Machine Learning Pipeline

> **End-to-end makine öğrenmesi projesi:** Pima Indians Diabetes veri seti üzerinde veri ön işlemeden final model değerlendirmesine kadar 7 fazlı kapsamlı bir ML pipeline'ı.

**Hazırlayan:** Eren Demirbaş - Ulaş Efe Sakarya | **Tarih:** Nisan 2026 | **Ders:** Makine Öğrenmesi ve Fizikteki Uygulamaları Proje Ödevi

---

## İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti](#-veri-seti)
- [Proje Yapısı](#-proje-yapısı)
- [Pipeline Fazları](#-pipeline-fazları)
- [Temel Sonuçlar](#-temel-sonuçlar)
- [Model Karşılaştırması](#-model-karşılaştırması)
- [Kurulum ve Kullanım](#-kurulum-ve-kullanım)
- [Gereksinimler](#-gereksinimler)

---

## Proje Hakkında

Bu proje, Pima Indian kökenli kadınlara ait klinik ölçümlerden **diyabet teşhisi yapmayı** amaçlamaktadır. Proje; veri kalitesi sorunlarının çözümünden istatistiksel model değerlendirmesine kadar uzanan sistematik bir makine öğrenmesi iş akışını kapsamaktadır.

**Çözülen Problem:** Binary sınıflandırma — bir hastanın 8 klinik ölçüme dayanarak diyabetli olup olmadığını tahmin etmek.

**Ana Bulgular:**
- Glikoz düzeyi, **tüm yöntemlerde** (korelasyon, gradient descent, lojistik regresyon, bootstrap) tutarlı biçimde en güçlü diyabet prediktörü olarak öne çıkmaktadır
- Lojistik Regresyon **AUC = 0.835** ile en yüksek ayırt edicilik gücüne sahiptir
- KNN (K=26) **Accuracy = 0.767** ve **F1 = 0.630** ile en yüksek doğruluğu sağlamaktadır
- EKK ve MLE'nin matematiksel denkliği sayısal olarak kanıtlanmıştır (fark: 3.16 × 10⁻¹⁶)

---

## Veri Seti

**Kaynak:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — Ulusal Diyabet ve Sindirim ve Böbrek Hastalıkları Enstitüsü (NIDDK)

| Özellik | Açıklama | Birim |
|---|---|---|
| `Pregnancies` | Gebelik sayısı | — |
| `Glucose` | Plazma glikoz konsantrasyonu | mg/dL |
| `BloodPressure` | Diyastolik kan basıncı | mmHg |
| `SkinThickness` | Triseps cilt kalınlığı | mm |
| `Insulin` | 2 saatlik serum insülini | µU/mL |
| `BMI` | Vücut Kitle İndeksi | kg/m² |
| `DiabetesPedigreeFunction` | Diyabet soy ağacı skoru | — |
| `Age` | Yaş | yıl |
| **`Outcome`** | **Diyabet tanısı (hedef)** | **0 / 1** |

**Boyut:** 768 örnek × 9 değişken | **Sınıf Dağılımı:** %65.1 Negatif / %34.9 Pozitif

---

## Proje Yapısı

```
proje_odev/
│
├── diabetes.csv                    # Ham veri seti
│
├── phase1_preprocessing.py         # Veri ön işleme
├── phase2_regression.py            # OLS & MLE regresyon analizi
├── phase3_gradient_descent.py      # Sıfırdan gradient descent
├── phase4_classification.py        # Sınıflandırma modelleri
├── phase5_bias_variance.py         # Bias-Variance analizi
├── phase6_resampling.py            # Cross-validation & bootstrap
├── phase7_final_evaluation.py      # Final ROC/AUC değerlendirmesi
│
├── phase1_outputs/                 # (9 grafik + 6 CSV dosyası)
├── phase2_outputs/                 # (8 grafik)
├── phase3_outputs/                 # (9 grafik)
├── phase4_outputs/                 # (7 grafik)
├── phase5_outputs/                 # (6 grafik)
├── phase6_outputs/                 # (8 grafik)
├── phase7_outputs/                 # (8 grafik + 2 CSV)
│
├── phase1_report.md                # Faz 1 detaylı raporu
├── phase2_report.md                # Faz 2 detaylı raporu
├── phase3_report.md                # Faz 3 detaylı raporu
├── phase4_report.md                # Faz 4 detaylı raporu
├── phase5_report.md                # Faz 5 detaylı raporu
├── phase6_report.md                # Faz 6 detaylı raporu
├── phase7_report.md                # Faz 7 detaylı raporu
│
└── FINAL_REPORT.md                 # Tüm fazları kapsayan nihai rapor
```

---

## Pipeline Fazları

### Faz 1 — Veri Ön İşleme (`phase1_preprocessing.py`)

> Biyolojik olarak imkânsız sıfır değerlerinin tespiti ve işlenmesi, veri bölme ve ölçeklendirme.

- **Sıfır değer impütasyonu:** 652 adet geçersiz değer medyan ile dolduruldu
- **Bölme stratejisi:** Stratified 70/15/15 (Train/Val/Test) — sınıf dengesi korundu
- **Ölçeklendirme:** `StandardScaler` yalnızca Train setine fit edildi → veri sızıntısı kontrolü

| Değişken | Sıfır Oranı | Medyan İmpütasyon |
|---|---|---|
| Insulin | **%48.7** ⚠️ | 125.0 |
| SkinThickness | **%29.6** ⚠️ | 29.0 |
| BloodPressure | %4.6 | 72.0 |
| BMI | %1.4 | 32.0 |
| Glucose | %0.7 | 122.0 |

---

### Faz 2 — Regresyon Analizi: OLS & MLE (`phase2_regression.py`)

> BMI tahmin etmek için Ordinary Least Squares ve Maximum Likelihood Estimation yöntemlerinin teorik ve sayısal karşılaştırması.

**Temel Bulgu:** Normal hata varsayımı altında β_OLS ≡ β_MLE — maksimum mutlak fark: **3.16 × 10⁻¹⁶** (makine hassasiyetinde)

```
Matematiksel kanıt:
∂ln𝐿/∂β = 0  ⟺  XᵀXβ = Xᵀy  (Normal Denklem)
```

**Test Performansı:** MSE = 31.85, RMSE = 5.64, R² = 0.431

---

### Faz 3 — Gradient Descent (`phase3_gradient_descent.py`)

> Binary Cross-Entropy kaybıyla Batch Gradient Descent'in sıfırdan Python implementasyonu.

| Öğrenme Oranı (α) | Son Kayıp | Val Accuracy | Durum |
|---|---|---|---|
| 0.1 | 0.4610 | 0.7759 | Hızlı yakınsama |
| **0.01** | **0.4610** | **0.7759** | **Optimal** |
| 0.001 | 0.4928 | 0.7500 | Yavaş |
| 0.0001 | 0.6299 | 0.7328 | Yakınsamadı |

**El yapımı GD vs. sklearn:** Birebir özdeş sonuç (%100 uyum) — implementasyonun doğruluğu kanıtlandı.

---

### Faz 4 — Sınıflandırma ve İstatistiksel Yorum (`phase4_classification.py`)

> Lojistik Regresyon, KNN ve Gaussian Naive Bayes modellerinin Z-istatistiği, p-değeri ve Odds Oranı ile istatistiksel yorumu.

**Lojistik Regresyon — Anlamlı değişkenler:**

| Değişken | β | Z | p-değeri | Odds Oranı |
|---|---|---|---|---|
| **Glucose** | +1.164 | +8.06 | **<0.001** | 3.203 |
| **BMI** | +0.660 | +4.44 | **<0.001** | 1.934 |
| **Pregnancies** | +0.325 | +2.52 | **0.012** | 1.383 |
| **Age** | +0.269 | +2.02 | **0.044** | 1.308 |
| BloodPressure | −0.084 | −0.66 | 0.507 | 0.919 |

---

### Faz 5 — Bias-Variance Trade-off (`phase5_bias_variance.py`)

> KNN modelinde K=1'den K=50'ye Bias² ve Varyans ayrışımı, bootstrap ile hesaplama.

```
Beklenen Test Hatası = Bias² + Varyans + ε (gürültü)
```

| K | Train Err | Val Err | Bölge |
|---|---|---|---|
| 1 | **0.000** | 0.310 | ⬆️ Overfit |
| **26** | 0.211 | **0.224** | **Optimum** |
| 50 | 0.248 | 0.233 | ⬇️ Underfit |

---

### Faz 6 — Resampling (`phase6_resampling.py`)

> 10-Fold Stratified Cross-Validation ve B=1000 Bootstrap ile katsayı güven aralıkları.

**10-Fold CV Sonuçları:**

| Model | CV Accuracy | CV F1 | Stabilite (σ) |
|---|---|---|---|
| Lojistik Reg. | 0.7682 ± 0.027 | 0.6297 ± 0.044 | — |
| **KNN (K=26)** | 0.7630 ± **0.017** | 0.6024 ± 0.043 | **En stabil** |
| GNB | 0.7447 ± 0.048 | 0.6196 ± 0.067 | En değişken |

**Bootstrap (B=1000) — %95 Güven Aralıkları:**
- Glucose: [0.920, 1.457] — sıfırı **içermiyor** 
- BMI: [0.406, 0.928] — sıfırı **içermiyor** 

---

### Faz 7 — Final Model Değerlendirmesi (`phase7_final_evaluation.py`)

> ROC/AUC eğrileri, Precision-Recall analizi, kalibrasyon eğrisi ve nihai model karşılaştırması.

**Test Seti AUC-ROC:**

| Model | AUC-ROC |
|---|---|
| **Lojistik Regresyon** | **0.835** ★ |
| KNN (K=26) | 0.827 |
| Gaussian NB | 0.779 |
| Rastgele tahmin | 0.500 |

---

## Model Karşılaştırması

| Kriter | Lojistik Reg. | KNN (K=26) | Gaussian NB |
|---|---|---|---|
| **Accuracy** | 0.707 | **0.767** | 0.716 |
| **AUC-ROC** | **0.835** | 0.827 | 0.779 |
| **F1** | 0.564 | **0.630** | 0.621 |
| **Recall** | 0.550 | 0.575 | **0.675** |
| **Yorumlanabilirlik** | **Yüksek** | Düşük | Orta |
| **Kalibrasyon** | **İyi** | Orta | Orta |
| **CV Stabilite (σ)** | 0.027 | **0.017** | 0.048 |

### Kullanım Senaryosuna Göre Model Seçimi

| Senaryo | Önerilen Model | Gerekçe |
|---|---|---|
| Klinik karar destek | **Lojistik Regresyon** | En yüksek AUC, yorumlanabilir β, iyi kalibrasyon |
| Tarama programı (Accuracy) | **KNN (K=26)** | En yüksek accuracy ve F1, en stabil CV |
| Hasta kaçırmama önceliği | **Gaussian NB** | En yüksek Recall (0.675) |

---

## Kurulum ve Kullanım

### Gereksinimler

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Pipeline'ı Çalıştırma

Her faz bağımsız olarak çalıştırılabilir. Fazlar sırayla uygulanmalıdır:

```bash
# Faz 1: Veri ön işleme (önce çalıştırılmalı — diğer fazlar bu çıktıları kullanır)
python phase1_preprocessing.py

# Faz 2: OLS ve MLE regresyon karşılaştırması
python phase2_regression.py

# Faz 3: Gradient descent implementasyonu
python phase3_gradient_descent.py

# Faz 4: Sınıflandırma ve istatistiksel analiz
python phase4_classification.py

# Faz 5: Bias-variance trade-off analizi
python phase5_bias_variance.py

# Faz 6: Cross-validation ve bootstrap
python phase6_resampling.py

# Faz 7: Final ROC/AUC değerlendirmesi
python phase7_final_evaluation.py
```

> **Not:** Her script kendi `phaseX_outputs/` dizinine grafikleri ve CSV dosyalarını otomatik olarak kaydeder.

### Hızlı Başlangıç

```bash
# Tüm pipeline'ı sırayla çalıştır
for i in 1 2 3 4 5 6 7; do
    python phase${i}_*.py
done
```

---

## Gereksinimler

| Kütüphane | Sürüm (önerilen) | Kullanım |
|---|---|---|
| `numpy` | ≥ 1.23 | Matris işlemleri, OLS/MLE hesaplama |
| `pandas` | ≥ 1.5 | Veri yükleme ve manipülasyon |
| `matplotlib` | ≥ 3.6 | Grafik üretimi |
| `seaborn` | ≥ 0.12 | İstatistiksel görselleştirme |
| `scikit-learn` | ≥ 1.1 | ML modelleri, cross-validation |
| `scipy` | ≥ 1.9 | İstatistiksel testler |

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## Raporlar

Her faz için ayrıntılı rapor mevcuttur:

| Faz | Rapor | İçerik |
|---|---|---|
| 1 | [phase1_report.md](phase1_report.md) | EDA, impütasyon, bölme, ölçeklendirme |
| 2 | [phase2_report.md](phase2_report.md) | OLS vs MLE teorisi ve sayısal kanıt |
| 3 | [phase3_report.md](phase3_report.md) | GD implementasyonu, öğrenme oranı analizi |
| 4 | [phase4_report.md](phase4_report.md) | Sınıflandırma, Z-istatistikleri, Odds Oranları |
| 5 | [phase5_report.md](phase5_report.md) | Bias-Variance ayrışımı, karar sınırı görselleştirme |
| 6 | [phase6_report.md](phase6_report.md) | 10-Fold CV ve Bootstrap güven aralıkları |
| 7 | [phase7_report.md](phase7_report.md) | ROC/AUC, kalibrasyon, final karşılaştırma |
| **Final** | [**FINAL_REPORT.md**](FINAL_REPORT.md) | **Tüm fazları kapsayan kapsamlı rapor** |

---

## Üretilen Çıktılar

| Faz | Script | Grafik Sayısı | CSV Sayısı |
|---|---|---|---|
| 1 | `phase1_preprocessing.py` | 9 | 6 |
| 2 | `phase2_regression.py` | 8 | — |
| 3 | `phase3_gradient_descent.py` | 9 | — |
| 4 | `phase4_classification.py` | 7 | — |
| 5 | `phase5_bias_variance.py` | 6 | — |
| 6 | `phase6_resampling.py` | 8 | — |
| 7 | `phase7_final_evaluation.py` | 8 | 2 |
| **TOPLAM** | **7 script** | **55 grafik** | **8 CSV** |

---

## Sınırlılıklar

1. **Küçük veri seti:** 768 örnek — büyük veri setlerinde model performansı artabilir
2. **Cinsiyete özgüllük:** Yalnızca kadın hastalar — bulgular genellenemez
3. **Yüksek impütasyon:** Insulin (%48.7) ve SkinThickness (%29.6) için medyan doldurma uygulandı
4. **Doğrusal regresyon kısıtı:** BMI için R²=0.43 — ensemble yöntemler iyileştirme sağlayabilir
5. **Sınıf dengesizliği:** %35 pozitif oran — SMOTE veya `class_weight='balanced'` uygulanabilir

---

*Pima Indians Diabetes Dataset — NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)*
