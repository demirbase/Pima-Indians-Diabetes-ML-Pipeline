# FAZ 2 — Regresyon Analizi: EKK vs. MLE Raporu
**Görev:** BMI değerini diğer özelliklerle tahmin et  
**Yöntemler:** EKK (Ordinary Least Squares) · MLE (Maximum Likelihood Estimation)  
**Script:** `phase2_regression.py` | **Çıktı:** `phase2_outputs/`

---

## 1. Problem Tanımı

| Özellik | Değer |
|---|---|
| **Hedef değişken** | `BMI` (Vücut Kitle İndeksi) |
| **Bağımsız değişkenler** | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, Age (7 özellik) |
| **Train seti** | 536 örnek |
| **Val / Test** | 116 / 116 örnek |
| **Ölçeklendirme** | StandardScaler (Train'e fit, ayrı scaler_X & scaler_y) |

---

## 2. Teorik Arka Plan

### 2A — EKK (Ordinary Least Squares)

Hedef: **artık karelerin toplamını minimize et**

$$\hat{\beta}_{EKK} = \arg\min_\beta \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top\beta)^2 = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

Analitik (kapalı form) çözüm → **Normal Denklem**.

### 2B — MLE (Maximum Likelihood Estimation)

Hata dağılımı varsayımı: $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$

Log-olabilirlik fonksiyonu:

$$\ln \mathcal{L}(\beta, \sigma) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{x}_i^\top\beta)^2$$

**Maksimizasyon ↔ Negatif log-L minimizasyonu** (scipy L-BFGS-B ile)

> [!IMPORTANT]
> Normal hata varsayımı altında log-L fonksiyonunu β'ya göre maksimize etmek, artık kareleri minimize etmekle **matematiksel olarak özdeştir**. Bu nedenle **β_EKK = β_MLE** beklenir.

---

## 3. Katsayı Karşılaştırması

![01_beta_comparison.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_beta_comparison.png)

| Parametre | β_EKK | β_MLE | Mutlak Fark |
|---|---|---|---|
| intercept | −0.0000006 | −0.0000006 | ~0 |
| Pregnancies | 0.0601459 | 0.0601459 | **3.5e-16** |
| Glucose | 0.0398108 | 0.0398108 | **5.6e-17** |
| BloodPressure | 0.1943737 | 0.1943737 | **0.0** |
| SkinThickness | 0.5079540 | 0.5079540 | **0.0** |
| Insulin | 0.0649959 | 0.0649959 | **0.0** |
| DiabetesPedigreeFunction | 0.0843231 | 0.0843231 | **0.0** |
| Age | −0.1330381 | −0.1330381 | **0.0** |

> [!NOTE]
> **Maksimum mutlak fark: 3.16 × 10⁻¹⁶** — Bu sayısal anlamda sıfırdır (float64 makine hassasiyeti ~2.2e-16). EKK ve MLE **tamamen özdeş** katsayılar üretmiştir.

---

## 4. Log-Likelihood Yüzeyi

````carousel
![Log-Likelihood Yüzeyi (Kontur + Isı Haritası)](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_loglikelihood_surface.png)
<!-- slide -->
![Teorik Karşılaştırma — Neden EKK = MLE?](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_theory_comparison.png)
````

**Gözlem:** Log-likelihood yüzeyinin maksimumu (🟢 yeşil nokta = MLE) ile EKK çözümü (🔴 kırmızı üçgen = EKK) aynı noktaya düşmektedir. Sağdaki NLL eğrisi de tek bir minimum göstermektedir.

---

## 5. Özellik Önemi (Katsayılar)

![05_feature_importance.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_feature_importance.png)

| Sıra | Özellik | β (z-skor) | Etki Yönü |
|---|---|---|---|
| 1 | **SkinThickness** | +0.5080 | Pozitif — En güçlü prediktör |
| 2 | **BloodPressure** | +0.1944 | Pozitif |
| 3 | **Age** | −0.1330 | Negatif — Yaş BMI'yi düşürüyor |
| 4 | DiabetesPedigreeFunction | +0.0843 | Pozitif |
| 5 | Insulin | +0.0650 | Pozitif |
| 6 | Pregnancies | +0.0601 | Pozitif |
| 7 | Glucose | +0.0398 | Pozitif — En zayıf |

> [!TIP]
> **SkinThickness → BMI'nin en güçlü prediktörü.** Bu beklenen bir sonuçtur; cilt kıvrım kalınlığı vücut yağ oranıyla doğrudan ilişkilidir ve BMI ile güçlü korelasyon gösterir.

---

## 6. Model Performansı

### 6A — Sayısal Metrikler

| Model | Set | MSE | RMSE | MAE | R² |
|---|---|---|---|---|---|
| EKK | Validation | 34.38 | 5.86 | 4.38 | 0.3064 |
| EKK | **Test** | 31.85 | **5.64** | 4.42 | **0.4314** |
| MLE | Validation | 34.38 | 5.86 | 4.38 | 0.3064 |
| MLE | **Test** | 31.85 | **5.64** | 4.42 | **0.4314** |

> [!NOTE]
> EKK ve MLE metrikleri birebir aynıdır — katsayıların sayısal olarak özdeş olduğunu doğrular.

**R² Yorumu:** Test setinde R² = 0.43. Bu, modelin BMI varyansının yaklaşık %43'ünü açıklayabildiği anlamına gelir. Doğrusal modelin sınırlılığı açıktır; Faz 3'te KNN ile karşılaştırılacak.

### 6B — Tahmin vs. Gerçek

![03_predicted_vs_actual.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_predicted_vs_actual.png)

![06_metrics_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_metrics_dashboard.png)

---

## 7. Kalıntı Analizi

![04_residual_analysis.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_residual_analysis.png)

| Kontrol | Sonuç | Yorum |
|---|---|---|
| Histogram şekli | Yaklaşık çan eğrisi | ✅ Normal dağılım varsayımı makul |
| Q-Q plot | Hafif kuyruk sapmaları | ⚠️ Hafif heavy-tail; transformasyon düşünülebilir |
| Kalıntı vs. Tahmin | Rastgele dağılım | ✅ Heteroskedastisite yok |
| Kalıntı zaman | Yapısal örüntü yok | ✅ Otokorelasyon yok |

---

## 8. Özet Dashboard

![08_summary_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/08_summary_dashboard.png)

---

## 9. Üretilen Dosyalar

| Dosya | Tür | Açıklama |
|---|---|---|
| `beta_comparison.csv` | CSV | EKK / MLE / Analitik β karşılaştırması |
| `metrics.csv` | CSV | Val & Test MSE, RMSE, MAE, R² |
| `01_beta_comparison.png` | Grafik | Katsayı değerleri + mutlak fark (log) |
| `02_loglikelihood_surface.png` | Grafik | Log-L kontur + ısı haritası |
| `03_predicted_vs_actual.png` | Grafik | Tahmin vs. Gerçek BMI |
| `04_residual_analysis.png` | Grafik | 8 panel kalıntı analizi |
| `05_feature_importance.png` | Grafik | Standardize katsayı büyüklükleri |
| `06_metrics_dashboard.png` | Grafik | RMSE / MAE / R² bar grafikleri |
| `07_theory_comparison.png` | Grafik | Normal hata + NLL eğrisi |
| `08_summary_dashboard.png` | Grafik | Özet dashboard (6 panel) |

---

## 10. Kritik Sonuçlar ve Faz 3 İçin Notlar

> [!IMPORTANT]
> **β_EKK = β_MLE** (fark: 3.16 × 10⁻¹⁶ — makine hassasiyeti sınırında). Bu teorik beklentiyle tam uyumludur.

**EKK = MLE Olmasının Kanıtı:**

$$\frac{\partial \ln\mathcal{L}}{\partial\beta} = \frac{1}{\sigma^2}\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\beta) = 0 \iff \mathbf{X}^\top\mathbf{X}\beta = \mathbf{X}^\top\mathbf{y}$$

Bu, Normal Denklem ile aynıdır — kanıt tamamdır.

**Faz 3 için öneriler:**
- Doğrusal model R² = 0.43 ile orta düzey performans gösteriyor
- SkinThickness yüksek katsayısı (0.508) Faz 1'deki %29.6 medyan doldurma oranından etkilenmiş olabilir — **dikkatli yorumlanmalı**
- BMI tahmini için KNN'in (özellikle SkinThickness'ın rolüyle) daha iyi sonuç vermesi beklenir
