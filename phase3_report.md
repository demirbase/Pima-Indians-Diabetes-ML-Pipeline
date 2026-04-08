# FAZ 3 — Gradient Descent (Gradyan Alçalma) Raporu
**Görev:** Diyabet ikili sınıflandırması (Outcome = 0/1)  
**Yöntem:** Batch Gradient Descent (sıfırdan, el ile)  
**Script:** `phase3_gradient_descent.py` | **Çıktı:** `phase3_outputs/`

---

## 1. Teorik Arka Plan

### Lojistik Regresyon Modeli

$$\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}$$

### Kayıp Fonksiyonu — Binary Cross-Entropy (BCE)

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

### Gradyan

$$\nabla_\mathbf{w}\mathcal{L} = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$$

### Güncelleme Kuralı

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \nabla_\mathbf{w}\mathcal{L}(\mathbf{w}_t)$$

![09_theory_visualization.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/09_theory_visualization.png)

---

## 2. Uygulama Detayları

| Parametre | Değer |
|---|---|
| Algoritma | Batch Gradient Descent |
| Ağırlık başlangıcı | N(0, 0.01) — küçük rastgele değerler |
| Maksimum iterasyon | 5.000 |
| Yakınsama toleransı | \|ΔLoss\| < 1e-9 |
| Test edilen α değerleri | **0.1 · 0.01 · 0.001 · 0.0001** |

---

## 3. Öğrenme Oranı Analizi

### 3A — Loss vs. İterasyon (4 LR)

![01_loss_vs_iteration.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_loss_vs_iteration.png)

| α | Son Train Loss | Val Accuracy | Yakınsama | Durum |
|---|---|---|---|---|
| **0.1** | 0.4610 | 0.7759 | iter 1122 | ✅ Hızlı yakınsama |
| **0.01** | 0.4610 | **0.7759** | iter 5000 | ✅ Optimal bölge |
| **0.001** | 0.4928 | 0.7500 | iter 5000 | ⚠️ Yavaş |
| **0.0001** | 0.6299 | 0.7328 | iter 5000 | ❌ Tam yakınsamadı |

> [!NOTE]
> **α = 0.1** bu veri setinde ıraksamamadı — ancak gerçek derin ağlarda bu değer tipik olarak patlama yaratır. Burada stabil çünkü veri ölçeklendirilmiş (StandardScaler) ve model basit (lojistik regresyon).
>
> **α = 0.0001** → 5000 iterasyonda loss hâlâ yüksek (0.63), yakınsamak için ~50.000 iterasyon gerekir.

### 3B — Tüm LR Karşılaştırması

![02_all_lr_comparison.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_all_lr_comparison.png)

### 3C — Validation Accuracy Eğrileri

![03_val_accuracy.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_val_accuracy.png)

> [!TIP]
> Turuncu yatay kesik çizgi = sklearn referans değeri (0.7759). El ile yazılan GD modeli (α=0.01) sklearn ile **tamamen aynı doğruluğa** ulaştı — implementasyonun doğruluğunu kanıtlar.

---

## 4. Gradyan Normu Analizi

![04_gradient_norm.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_gradient_norm.png)

‖∇L‖ → 0 yakınsaması, algoritmanın bir durağan noktaya (minimum) ulaştığını kanıtlar.  
α=0.01'de gradyan normu monoton olarak azalarak **yakınsamayı** doğrular.

---

## 5. Optimal Model (α = 0.01) Değerlendirmesi

### 5A — Performans Metrikleri

| Metrik | Validation | Test |
|---|---|---|
| **Accuracy** | **0.7759** | **0.7155** |
| **AUC-ROC** | **0.8354** | **0.8342** |
| Log-Loss | 0.4826 | 0.4774 |
| sklearn referans (Val Acc) | 0.7759 | — |

### 5B — Öğrenilen Ağırlıklar

| Özellik | Ağırlık (GD) | Yorum |
|---|---|---|
| **Glucose** | **+1.1374** | En güçlü pozitif prediktör |
| **BMI** | **+0.6286** | İkinci güçlü |
| Pregnancies | +0.3165 | Orta düzey |
| Age | +0.2669 | Orta düzey |
| DiabetesPedigreeFunction | +0.2178 | Zayıf-orta |
| SkinThickness | +0.0719 | Zayıf |
| BloodPressure | −0.0701 | **Negatif** — hafif koruyucu etki |
| Insulin | −0.0065 | Neredeyse nötr |
| intercept | −0.8635 | Taban bias |

> [!IMPORTANT]
> **Glucose (w=1.137)** açık ara en güçlü prediktör. Bu Faz 1'deki korelasyon analizinde (r=0.493) bulduğumuzla **tutarlı**. BMI ise Faz 2'de regresyon hedefiydi, burada bağımsız değişken olarak **+0.629** ağırlık aldı.

![07_feature_weights.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_feature_weights.png)

### 5C — Karışıklık Matrisi

![05_confusion_matrix.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_confusion_matrix.png)

### 5D — ROC Eğrisi (GD vs. sklearn)

![06_roc_curve.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_roc_curve.png)

**AUC = 0.8342** (Test) — Rastgele (0.5) çok üzerinde. El ile yazılan GD, sklearn ile özdeş AUC üretmektedir.

---

## 6. LR Analiz Özet Dashboard

![08_lr_analysis_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/08_lr_analysis_dashboard.png)

---

## 7. Kritik Durum Değerlendirmesi

> [!WARNING]
> **α = 0.1 — Neden ıraksamadı?**  
> Veriler StandardScaler ile ölçeklendirildi. Ölçeklendirilmemiş verilerle α=0.1 kesinlikle ıraksar. Bu nedenle **Faz 1 preprocessing kritikti**.

> [!CAUTION]
> **α = 0.0001 — Neden yavaş?**  
> Her iterasyondaki güncelleme `Δw ≈ 10⁻⁴ × gradyan`. 5000 iterasyonda loss 0.63'de kaldı. Pratik eğitimde bu LR tercih edilmez.

---

## 8. Üretilen Dosyalar

| Dosya | Açıklama |
|---|---|
| `weights_comparison.csv` | GD vs. sklearn ağırlıkları |
| `lr_summary.csv` | Her LR için son loss, acc, yakınsama |
| `01_loss_vs_iteration.png` | 4 panel Loss vs. İterasyon |
| `02_all_lr_comparison.png` | Karşılaştırmalı Train/Val Loss |
| `03_val_accuracy.png` | Val Accuracy eğrileri |
| `04_gradient_norm.png` | ‖∇L‖ yakınsama kanıtı |
| `05_confusion_matrix.png` | Val ve Test karışıklık matrisleri |
| `06_roc_curve.png` | ROC — GD vs. sklearn |
| `07_feature_weights.png` | Özellik ağırlıkları bar grafiği |
| `08_lr_analysis_dashboard.png` | Özet dashboard |
| `09_theory_visualization.png` | Sigmoid + BCE + GD adımı |

---

## 9. Fazlar Arası Tutarlılık

| Gözlem | Faz 1 | Faz 2 | **Faz 3** |
|---|---|---|---|
| En güçlü prediktör | Glucose (r=0.493) | — | **Glucose (w=1.137)** ✅ |
| İkinci güçlü | BMI (r=0.312) | SkinThickness (β=0.508) | **BMI (w=0.629)** ✅ |
| Model performansı | — | R²=0.43 | **AUC=0.83** ✅ |
