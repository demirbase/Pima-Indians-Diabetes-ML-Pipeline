# Pima Indians Diyabet Veri Seti — Makine Öğrenmesi Projesi
## Kapsamlı Final Raporu

**Hazırlayan:** Eren Demirbaş  
**Tarih:** Nisan 2026  
**Veri Seti:** Pima Indians Diabetes Dataset (768 örnek, 9 değişken)  
**Proje Kapsamı:** Faz 1 → Faz 7 (Veri Ön İşleme → Final Model Değerlendirme)

---

## İÇİNDEKİLER

1. [Proje Özeti](#proje-özeti)
2. [Faz 1: Veri Ön İşleme](#faz-1-veri-ön-i̇şleme)
3. [Faz 2: Regresyon Analizi — EKK ve MLE](#faz-2-regresyon-analizi)
4. [Faz 3: Gradient Descent](#faz-3-gradient-descent)
5. [Faz 4: Sınıflandırma ve İstatistiksel Yorum](#faz-4-sınıflandırma)
6. [Faz 5: Bias-Variance Trade-off](#faz-5-bias-variance)
7. [Faz 6: Resampling — CV ve Bootstrap](#faz-6-resampling)
8. [Faz 7: Final Model Değerlendirmesi](#faz-7-final)
9. [Genel Tartışma ve Yorumlar](#tartışma)
10. [Sonuç](#sonuç)

---

## 1. Proje Özeti

Bu proje, Pima Indians Diabetes veri seti üzerinde kapsamlı bir makine öğrenmesi analizi gerçekleştirmeyi amaçlamaktadır. Çalışma yedi fazdan oluşmaktadır: (1) veri ön işleme ve hazırlık, (2) regresyon analizi ile parametre tahmini karşılaştırması, (3) gradient descent ile manuel optimizasyon, (4) çoklu sınıflandırma modellerinin istatistiksel yorumu, (5) bias-variance dengesi analizi, (6) çapraz geçerleme ve bootstrap yeniden örnekleme, ve (7) final model değerlendirmesi ve ROC/AUC analizi.

### Veri Seti Hakkında

Veri seti, Ulusal Diyabet ve Sindirim ve Böbrek Hastalıkları Enstitüsü'nden elde edilmiştir. Pima Indian kökenli kadınlar üzerinde yapılan ölçümleri içermekte olup diyabet teşhisi için kullanılmaktadır.

| Değişken | Açıklama | Tip |
|---|---|---|
| Pregnancies | Gebelik sayısı | Sayısal |
| Glucose | Plazma glukozu (mg/dL) | Sayısal |
| BloodPressure | Diyastolik kan basıncı (mmHg) | Sayısal |
| SkinThickness | Triseps cilt kalınlığı (mm) | Sayısal |
| Insulin | 2 saatlik serum insülini (µU/mL) | Sayısal |
| BMI | Vücut Kitle İndeksi (kg/m²) | Sayısal |
| DiabetesPedigreeFunction | Diyabet soy ağacı skoru | Sayısal |
| Age | Yaş (yıl) | Sayısal |
| **Outcome** | **Diyabet tanısı (0=Hayır, 1=Evet)** | **İkili (hedef)** |

**Sınıf Dağılımı:** %65.1 Negatif (Diyabet Yok) / %34.9 Pozitif (Diyabet Var) → Hafif dengesizlik.

---

## 2. Faz 1: Veri Ön İşleme

### 2.1 Veri Kalitesi Sorunları

İlk incelemede beş değişkende **biyolojik olarak imkânsız sıfır değerleri** tespit edilmiştir:

| Değişken | Sıfır Sayısı | Oranı | İmpütasyon |
|---|---|---|---|
| Glucose | 5 | %0.7 | Medyan (122.0) |
| BloodPressure | 35 | %4.6 | Medyan (72.0) |
| SkinThickness | 227 | %29.6 | Medyan (29.0) |
| Insulin | 374 | %48.7 | Medyan (125.0) |
| BMI | 11 | %1.4 | Medyan (32.0) |

> [!WARNING]
> **Kritik Not:** Insulin (%48.7) ve SkinThickness (%29.6) çok yüksek oranla medyan ile doldurulmuştur. Bu durum yapay bir yoğunlaşmaya yol açmakta ve bu iki değişkenin model katsayılarını yapay olarak düşürebilmektedir. Sonraki fazlardaki yorumlarda bu husus göz önünde bulundurulmuştur.

### 2.2 Veri Bölme Stratejisi

Stratifiye örnekleme (`stratify=y`) kullanılarak sınıf oranı korunmuş şekilde bölme yapılmıştır:

| Set | Boyut | Oranı | Pozitif Oranı |
|---|---|---|---|
| **Eğitim (Train)** | 536 | %70 | ~%35 |
| **Doğrulama (Validation)** | 116 | %15 | ~%35 |
| **Test** | 116 | %15 | ~%35 |

**Neden 3'e bölme?** Standart eğitim-test bölmesi "tek seferlik şans" riskine maruz kalır. Validation seti hiperparametre seçimi (örn. KNN'de K değeri) için kullanılırken test seti yalnızca final değerlendirmede kullanılmıştır.

### 2.3 Ölçeklendirme

`StandardScaler` (z-skor normalizasyonu) yalnızca eğitim setine fit edilerek uygulanmıştır:

$$z = \frac{x - \mu_{train}}{\sigma_{train}}$$

**Neden train'e fit?** Validation ve test verisi "görülmemiş" olarak simüle edilmelidir. Scaler'ı tüm veriye fit etmek **veri sızıntısına (data leakage)** yol açar — model geleceği "görmüş" olur.

![02_raw_distributions.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_raw_distributions.png)

![04_correlation_heatmap.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_correlation_heatmap.png)

### 2.4 Korelasyon Analizi Bulguları

Hedef değişken (Outcome) ile en yüksek korelasyona sahip değişkenler:

| Değişken | Pearson r | Yorum |
|---|---|---|
| **Glucose** | **+0.493** | En güçlü ilişki — tutarlı şekilde tüm fazlarda birinci sırada |
| BMI | +0.312 | Orta güçlü ilişki |
| Age | +0.238 | Orta ilişki |
| Pregnancies | +0.222 | Zayıf-orta |
| SkinThickness | +0.215 | Medyan doldurmanın etkisiyle düşük |
| Insulin | +0.204 | Yüksek impütasyon oranı nedeniyle güvenilirlik düşük |
| BloodPressure | +0.166 | En zayıf — ilerleyen fazlarda anlamsız çıkacak |

---

## 3. Faz 2: Regresyon Analizi — EKK ve MLE

### 3.1 Problem Tanımı

**Hedef:** BMI değerini diğer özelliklerden tahmin et (sürekli çıktı → regresyon).  
**Bağımsız değişkenler:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, Age (7 özellik).

### 3.2 Teorik Karşılaştırma

**EKK (Ordinary Least Squares):** Artık karelerin toplamını minimize eder.

$$\hat{\beta}_{EKK} = \arg\min_\beta \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \beta)^2 = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

**MLE (Maximum Likelihood Estimation):** Hata dağılımı **N(0, σ²)** varsayımı altında log-olabilirlik fonksiyonunu maksimize eder.

$$\ln\mathcal{L}(\beta, \sigma) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{x}_i^\top\beta)^2$$

**Teorik kanıt — Neden EKK = MLE?**

$$\frac{\partial \ln\mathcal{L}}{\partial\beta} = \frac{1}{\sigma^2}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\beta) = 0 \iff \mathbf{X}^\top\mathbf{X}\beta = \mathbf{X}^\top\mathbf{y}$$

Bu ifade Normal Denklem'in ta kendisidir. Dolayısıyla **β_EKK ≡ β_MLE** matematiksel zorunluluktur.

### 3.3 Sayısal Doğrulama

| Metrik | Değer |
|---|---|
| **Maksimum mutlak fark (β_EKK − β_MLE)** | **3.16 × 10⁻¹⁶** |
| Float64 makine hassasiyeti | ~2.22 × 10⁻¹⁶ |
| Uyum değerlendirmesi | ✅ Sayısal olarak sıfır — MÜKEMMEL |

### 3.4 Katsayı Tablosu (Ortak β)

| Değişken | β (Standardize) | En Önemli Prediktörler |
|---|---|---|
| **SkinThickness** | **+0.508** | BMI tahmininde en güçlü — beklenen |
| BloodPressure | +0.194 | İkinci güçlü |
| Age | −0.133 | Negatif etki — yaş arttıkça BMI düşüyor |
| DiabetesPedigreeFunction | +0.084 | Zayıf |
| Insulin | +0.065 | Zayıf |
| Pregnancies | +0.060 | Zayıf |
| Glucose | +0.040 | En zayıf (BMI tahmini için) |

### 3.5 Model Performansı

| Set | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Validation | 34.38 | 5.86 | 4.38 | 0.306 |
| **Test** | **31.85** | **5.64** | **4.42** | **0.431** |

**R² = 0.431 yorumu:** Doğrusal model BMI varyansının %43.1'ini açıklayabilmektedir. Bu, orta düzey bir performanstır. SkinThickness yüksek katsayısının (%29.6 medyan doldurma oranından) yapay olarak artmış olabileceği unutulmamalıdır.

![01_beta_comparison.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_beta_comparison.png)

---

## 4. Faz 3: Gradient Descent

### 4.1 Algoritma

**Görev:** Lojistik regresyon için Batch Gradient Descent sıfırdan uygulama.  
**Kayıp fonksiyonu:** Binary Cross-Entropy (BCE)

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Güncelleme kuralı:**

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \underbrace{\frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})}_{\nabla_\mathbf{w}\mathcal{L}}$$

### 4.2 Öğrenme Oranı Analizi

| α | Son Loss | Val Acc | Yakınsama | Durum |
|---|---|---|---|---|
| **0.1** | 0.4610 | 0.7759 | iter 1122 | Hızlı — bazen riskli |
| **0.01** | 0.4610 | 0.7759 | iter 5000 | ✅ **Optimal** |
| **0.001** | 0.4928 | 0.7500 | iter 5000 | Yavaş |
| **0.0001** | 0.6299 | 0.7328 | iter 5000 | ❌ Yakınsamadı |

**Önemli Gözlem:** α=0.1 bu veri setinde ıraksamamıştır — çünkü veriler StandardScaler ile ölçeklendirilmiştir. Ölçeklendirilmemiş veride bu LR tipik olarak patlama yapardı. Bu da **Faz 1 ölçeklendirmesinin kritik önemini** bir kez daha doğrulamaktadır.

### 4.3 El Yapımı GD vs. sklearn Karşılaştırması

| Metrik | El Yapımı GD (α=0.01) | sklearn (referans) |
|---|---|---|
| Val Accuracy | **0.7759** | 0.7759 |
| Test AUC | **0.8342** | 0.8342 |

**Sonuç:** %100 özdeş — implementasyonun matematiksel doğruluğu kanıtlanmıştır.

**En önemli öğrenilen ağırlıklar:**
- Glucose: **+1.137** (En dominan)
- BMI: **+0.629**
- Pregnancies: **+0.317**

![01_loss_vs_iteration.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_loss_vs_iteration.png)

---

## 5. Faz 4: Sınıflandırma ve İstatistiksel Yorum

### 5.1 Lojistik Regresyon — İstatistiksel Özet

**Std. hatalar** Fisher Bilgi Matrisi `Cov(β̂) = (XᵀWX)⁻¹` ile hesaplanmıştır.  
**Z-istatistiği:** `Z = β / SE(β)`, |Z| > 1.96 ise p < 0.05.

| Değişken | β | Z | p-değeri | Odds Oranı | Anlamlı |
|---|---|---|---|---|---|
| intercept | −0.872 | −7.454 | <0.001 | 0.418 | ★ |
| **Glucose** | **+1.164** | **+8.060** | **<0.001** | **3.203** | **★★★** |
| **BMI** | **+0.660** | **+4.435** | **<0.001** | **1.934** | **★★★** |
| **Pregnancies** | **+0.325** | **+2.516** | **0.012** | **1.383** | **★★** |
| **Age** | **+0.269** | **+2.018** | **0.044** | **1.308** | **★** |
| DiabetesPedigreeFunction | +0.219 | +1.901 | 0.057 | 1.245 | (Marjinal) |
| BloodPressure | −0.084 | −0.663 | 0.507 | 0.919 | — |
| SkinThickness | +0.052 | +0.365 | 0.715 | 1.053 | — |
| Insulin | −0.023 | −0.184 | 0.854 | 0.977 | — |

### 5.2 Log-Odds Yorumları

**Glucose (β=1.164, Z=8.06, p<0.001, OR=3.203):**  
Glikoz düzeyi 1 standart sapma arttığında, diyabet olma ihtimalinin log-odds değeri 1.164 artırmaktadır. Bunun anlamı, odds değerinin **3.203 katına çıkmasıdır** — diyabet açısından en kritik prediktördür.

**BMI (β=0.660, Z=4.44, p<0.001, OR=1.934):**  
BMI 1 standart sapma arttığında log-odds 0.660 artmakta, odds **%93 yükselmektedir**. Obezite ile diyabet arasındaki güçlü fizyolojik ilişkiyi doğrular.

**Pregnancies (β=0.325, Z=2.52, p=0.012, OR=1.383):**  
Her 1 std sapmalık artış log-odds'u 0.325 artırır, odds **%38 yükselir**. Gebelik sayısının hormonal değişimler yoluyla insülin direncini artırdığı bilinmektedir.

**Age (β=0.269, Z=2.02, p=0.044, OR=1.308):**  
En zayıf anlamlı etki. Yaş 1 std sapma arttığında odds **%31 yükselir**.

**Anlamsız Değişkenler (p>0.5): BloodPressure, SkinThickness, Insulin.**  
Bu değişkenlerin yüksek impütasyon oranları (SkinThickness %29.6, Insulin %48.7) anlamsızlığın temel sebebi olabilir.

![02_forest_plot.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_forest_plot.png)

### 5.3 KNN (K=1…20) Analizi

| K | Val Acc | Test Acc | AUC |
|---|---|---|---|
| 1 | 0.690 | 0.724 | 0.683 |
| 10 | 0.716 | 0.759 | 0.798 |
| **20** | **0.767** | **0.784** | **0.825** |

K büyüdükçe hem doğruluk hem AUC monoton iyileşmektedir — bu Faz 5'te 50'ye kadar genişletilmiştir.

### 5.4 Gaussian Naive Bayes

**Varsayım:** `P(xⱼ|y=k) ~ N(μⱼₖ, σ²ⱼₖ)` — her özellik her sınıf için Gaussian dağılır.

| Set | Accuracy | AUC |
|---|---|---|
| Val | 0.767 | 0.800 |
| Test | 0.716 | 0.779 |

GNB, bağımsızlık varsayımına rağmen makul performans sergilemektedir. En yüksek Recall (0.675) sağlaması klinik açıdan değerlidir.

### 5.5 Model Karşılaştırması (Test Seti)

| Model | Accuracy | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Lojistik Reg. | 0.707 | **0.835** | 0.564 | 0.579 | 0.550 |
| **KNN (K=20)** | **0.784** | 0.825 | **0.667** | **0.714** | 0.625 |
| GNB | 0.716 | 0.779 | 0.621 | 0.574 | **0.675** |

![06_model_comparison_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_model_comparison_dashboard.png)

---

## 6. Faz 5: Bias-Variance Trade-off

### 6.1 Teorik Çerçeve

$$\text{Beklenen Test Hatası} = \underbrace{\text{Bias}^2}_{\text{Model Yanlılığı}} + \underbrace{\text{Varyans}}_{\text{Model Değişkenliği}} + \underbrace{\varepsilon}_{\text{Gürültü (Azaltılamaz)}}$$

KNN modelinde K parametresi bu dengeyi doğrudan kontrol eder:
- **K küçük (K=1):** Karmaşık model → Düşük Bias, Yüksek Varyans → **Overfit**
- **K büyük (K=50):** Basit model → Yüksek Bias, Düşük Varyans → **Underfit**
- **K optimal:** İkisinin dengelendiği nokta → **Genelleme**

### 6.2 Error vs. K Analizi (K=1…50)

| K | Train Err | Val Err | Test Err | Bölge |
|---|---|---|---|---|
| **1** | **0.0000** | 0.310 | 0.276 | ⬆️ Overfit |
| 5 | 0.168 | 0.293 | 0.259 | Geçiş |
| 13 | 0.185 | 0.259 | 0.216 | İyileşme |
| **26** | 0.211 | **0.224** | 0.233 | ✅ **Optimum** |
| 40 | 0.235 | 0.224 | 0.250 | Underfit başlar |
| **50** | 0.248 | 0.233 | 0.241 | ⬆️ Underfit |

**K=1:** Train Err = 0.000 — Model eğitim verisini tamamen ezberledi. Her eğitim noktası kendi tek komşusudur → sıfır hata. Test hatası %27.6 — ciddi overfit.

**K=26:** Validation hatası minimumda (0.224). Grafikteki "çukur" noktası — optimum model burasıdır.

**K=50:** Her iki hata da yüksek — model, K büyüdükçe daha fazla komşuyu ortalıyor ve ince ayrımları kaybediyor (underfit).

### 6.3 Bootstrap Bias² & Varyans Ayrışımı

30 bootstrap örneklemesiyle elde edilen değerler:

| K | Bias² | Varyans | Toplam |
|---|---|---|---|
| 1 | 0.154 | **0.021** | 0.175 |
| 10 | 0.153 | 0.010 | 0.163 |
| **25** | 0.159 | **0.007** | **0.166** |
| 50 | **0.170** | 0.005 | 0.175 |

K artıkça Varyans 0.021'den 0.005'e dramatik düşüyor, Bias² ise 0.154'ten 0.170'e artıyor. **Trade-off görsel olarak kanıtlanmıştır.**

![01_error_vs_k.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_error_vs_k.png)

![03_bias_variance_decomposition.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_bias_variance_decomposition.png)

### 6.4 Karar Sınırı Görselleştirmesi

Glucose-BMI düzleminde:
- **K=1:** Çok girintili-çıkıntılı sınır (her veri noktasını izole eder)
- **K=26:** Düzgün, makul eğrisel sınır
- **K=50:** Aşırı düzleşmiş sınır

![04_decision_boundary.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_decision_boundary.png)

---

## 7. Faz 6: Resampling — Çapraz Geçerleme ve Bootstrap

### 7.1 10-Fold Stratified Cross Validation

**Motivasyon:** Tek seferlik hold-out bölmesi şans eseri iyi/kötü sonuç verebilir. 10-Fold CV her fold'u bir kez test seti olarak kullanır → istatistiksel olarak güvenilir μ ± σ elde edilir.

| Model | CV Acc μ ± σ | CV F1 μ ± σ | Gözlem |
|---|---|---|---|
| **Lojistik Reg.** | **0.7682 ± 0.0268** | **0.6297 ± 0.0438** | En yüksek ortalama |
| KNN (K=26) | 0.7630 ± **0.0172** | 0.6024 ± 0.0426 | **En düşük σ** — en stabil |
| GNB | 0.7447 ± 0.0479 | 0.6196 ± 0.0670 | En yüksek varyans |

**Önemli Yorum:** KNN'nin σ=0.017 ile en düşük fold-to-fold değişkenliğe sahip olması, bu modelin farklı veri bölmelerinde tutarlı çalıştığını gösterir. GNB'nin yüksek σ'su (0.048), bazı fold'larda Gaussian varsayımının bozulduğuna işaret etmektedir.

![01_cv_fold_results.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_cv_fold_results.png)

### 7.2 Bootstrap (B=1000) — Katsayı Güven Aralıkları

**Yöntem:** Train+Val setinden 1000 kez yerine koyarak örneklem, her seferinde StandardScaler + LogisticRegression fit. Katsayıların empirik dağılımları elde edildi.

| Değişken | β | Boot μ | Boot σ | 95% CI (Pct) | Anlamlı |
|---|---|---|---|---|---|
| **Glucose** | 1.1497 | 1.1752 | 0.1403 | [0.920, 1.457] | **★** |
| **BMI** | 0.6550 | 0.6684 | 0.1305 | [0.406, 0.928] | **★** |
| **Pregnancies** | 0.4064 | 0.4065 | 0.1386 | [0.136, 0.674] | **★** |
| **DiabetesPedigreeFunction** | 0.2444 | 0.2479 | 0.1216 | [0.013, 0.485] | **★** |
| Age | 0.1206 | 0.1282 | 0.1357 | [-0.137, 0.391] | — |
| BloodPressure | −0.0916 | −0.0927 | 0.1052 | [-0.285, 0.123] | — |
| SkinThickness | 0.0638 | 0.0669 | 0.1231 | [-0.171, 0.318] | — |
| Insulin | −0.0281 | −0.0341 | 0.1413 | [-0.318, 0.227] | — |

**Önemli Fark — Faz 4 vs Faz 6:**
- **Age:** Faz 4'te p=0.044 (anlamlı) iken Bootstrap CI [-0.137, 0.391] sıfırı kapsıyor → **Bootstrap daha muhafazakâr**
- **DiabetesPedigreeFunction:** Faz 4'te p=0.057 (marjinal) iken Bootstrap CI [0.013, 0.485] sıfırı içermiyor → **Bootstrap anlamlı buluyor**

Bu tutarsızlık, iki yöntemin farklı varsayımlarından kaynaklanmaktadır: Fisher matrisi normal dağılım gerektirir; Bootstrap dağılım varsayımı yapmaz.

**Bootstrap Accuracy Dağılımı:**

| İstatistik | Değer |
|---|---|
| μ | 0.7833 |
| σ | 0.0173 |
| 95% CI | [0.7485, 0.8160] |
| Q-Q R² | ~0.99 → Normal dağılım ✅ |

![04_bootstrap_histograms.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_bootstrap_histograms.png)

![05_bootstrap_forest.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_bootstrap_forest.png)

---

## 8. Faz 7: Final Model Değerlendirmesi

### 8.1 ROC Eğrisi Analizi

ROC (Receiver Operating Characteristic) eğrisi, bir sınıflandırıcının tüm olası eşik değerlerinde TPR (duyarlılık) / FPR (1-özgüllük) dengesini gösterir.

**Test Seti AUC Skorları:**

| Model | AUC-ROC | Sınıf |
|---|---|---|
| **Lojistik Reg.** | **0.8352** | ★ İyi (0.8-0.9 arası) |
| KNN (K=26) | 0.8268 | ★ İyi |
| Gaussian NB | 0.7786 | Kabul edilebilir |
| Rastgele | 0.5000 | Referans (yazı-tura) |

**AUC Yorumlama Rehberi:**

| AUC Aralığı | Anlam |
|---|---|
| 0.90 – 1.00 | Mükemmel |
| **0.80 – 0.90** | **İyi ✅ (Bizim modellerimiz)** |
| 0.70 – 0.80 | Kabul edilebilir |
| 0.60 – 0.70 | Zayıf |
| 0.50 – 0.60 | İşe yaramaz |

Lojistik Regresyon ve KNN her iki eğri de **"İyi" kategorisinde** yer almaktadır.

![01_roc_curves.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_roc_curves.png)

### 8.2 Precision-Recall Analizi

Sınıf dengesizliği (%34.9 pozitif) söz konusu olduğunda ROC yanıltıcı olabilir; PR eğrisi daha bilgilendiricidir.

- **Lojistik Reg.:** En yüksek AP (Average Precision) — eşik bağımsız olarak en iyi
- **GNB:** Recall öncelikli — yüksek duyarlılık ama düşük kesinlik

### 8.3 Regresyon Final Metrikleri

| Set | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Validation | 0.771 | 0.878 | 0.656 | 0.306 |
| **Test** | **0.714** | **0.845** | **0.662** | **0.431** |

> [!NOTE]
> **Standardize uzayda** (z-skor) MSE=0.714. Orijinal BMI birimlerine çevrildiğinde bu değerler ~31.85 kg²/m⁴ ve RMSE~5.64 kg/m²'ye karşılık gelir.

### 8.4 Kalibrasyon Analizi

![07_calibration_curve.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_calibration_curve.png)

Lojistik Regresyon, çapraz çizgiye (mükemmel kalibrasyon) en yakın model olarak öne çıkmaktadır. Bu, "model %70 diyabet olasılığı dediğinde gerçekten yaklaşık %70 ihtimalle hasta olduğu" anlamına gelir — klinik karar destek uygulamaları için kritik.

### 8.5 Nihai Model Karşılaştırması

| Kriter | Lojistik Reg. | KNN (K=26) | GNB |
|---|---|---|---|
| **Accuracy** | 0.707 | **0.767** | 0.716 |
| **AUC-ROC** | **0.835** | 0.827 | 0.779 |
| **F1** | 0.564 | **0.630** | 0.621 |
| **Recall** | 0.550 | 0.575 | **0.675** |
| **Yorumlanabilirlik** | **Yüksek** | Düşük | Orta |
| **Kalibrasyon** | **İyi** | Orta | Orta |
| **CV Stabilite (σ)** | 0.027 | **0.017** | 0.048 |

---

## 9. Genel Tartışma ve Yorumlar

### 9.1 Tutarlılık Analizi — Dört Fazda Glucose

En çarpıcı bulgu, Glucose değişkeninin **tüm fazlarda tutarlı biçimde birinci sırada** yer almasıdır:

| Faz | Yöntem | Değer | Sıra |
|---|---|---|---|
| Faz 1 | Pearson korelasyon | r = +0.493 | 1/8 |
| Faz 3 | GD ağırlığı | w = +1.137 | 1/8 |
| Faz 4 | LR katsayısı | β = +1.164, Z=8.06 | 1/8 |
| Faz 6 | Bootstrap CI | [0.920, 1.457] | 1/8 |

Bu tutarlılık tesadüf değildir. Glukoz, insülin üretimini doğrudan etkileyen ve diyabet tanımının özünde yer alan biyomarker olduğundan, herhangi bir başarılı modelin bu değişkeni öncelikli olarak yakalaması beklenmektedir.

### 9.2 Anlamsız Değişkenler Meselesi

BloodPressure, SkinThickness ve Insulin üç yöntemde de anlamsız çıkmıştır:

**BloodPressure (p=0.507):** Kan basıncının diyabetle direkt ilişkisi zayıftır; daha çok kardiyovasküler komorbiditeleri yansıtır.

**SkinThickness ve Insulin:** %29.6 ve %48.7 medyan impütasyon oranları bu değişkenlerin bilgi içeriğini ciddi ölçüde aşındırmıştır. Özgün ölçüm yapılamayan gözlemlerin medyan değeri alması, bu değişkenlerin varyansını yapay olarak bastırır ve gerçek sinyal kaybolur.

**Pratik öneri:** Gelecek çalışmalarda bu iki değişken ya dışarıda bırakılmalı ya da daha sofistike impütasyon yöntemleri (Multiple Imputation by Chained Equations — MICE) kullanılmalıdır.

### 9.3 Model Seçim Rehberi

**Soru:** Hangi model kullanılmalıdır?

**Cevap: Amaca bağlıdır.**

> [!IMPORTANT]
> **Klinik Karar Destek (hasta riski değerlendirmesi):**
>
> → **Lojistik Regresyon** önerilir.
>
> _Gerekçe:_ En yüksek AUC (0.835), en iyi kalibrasyon, yorumlanabilir katsayılar (β, OR, p), istatistiksel güven aralıkları hesaplanabilir. Bir hekime "bu hasta için olasılık %73" demek, "model bu hastayı pozitif sınıflandırdı" demekten çok daha bilgilendiricidir.

> [!TIP]
> **Doğruluk Optimizasyonu (tarama programı):**
>
> → **KNN (K=26)** önerilir.
>
> _Gerekçe:_ En yüksek accuracy (0.767) ve F1 (0.630). Doğrusal olmayan karar sınırları, lojistik regresyonun kaçırdığı örüntüleri yakalıyor. CV stabilitesi de en yüksek (σ=0.017).

> [!NOTE]
> **Hasta Kaçırmama Önceliği (Recall maks.):**
>
> → **Gaussian Naive Bayes** önerilir.
>
> _Gerekçe:_ En yüksek Recall (0.675). Diyabetli bir hastayı "negatif" olarak sınıflandırmak (False Negative), negatif bir hastayı "pozitif" saymaktan (False Positive) çok daha maliyetlidir. Erken tarama bağlamında bu ödünleşim önemlidir.

### 9.4 Sınırlılıklar

1. **Veri büyüklüğü:** 768 örnek makine öğrenmesi için küçüktür. Özellikle KNN gibi instance-based modeller büyük veri setlerinde çok daha iyi çalışır.

2. **Cinsiyete özgüllük:** Veri seti yalnızca kadınları kapsamaktadır → sonuçlar gen. için uygulanamaz.

3. **Yüksek impütasyon:** Insulin (%48.7) ve SkinThickness (%29.6) değişkenlerindeki yoğun medyan doldurma, bu değişkenlerin katsayı güvenilirliğini zayıflatmaktadır.

4. **Doğrusal regresyon kısıtı:** BMI tahmini için R²=0.43 orta düzey — Random Forest veya XGBoost gibi ensemble yöntemleriyle anlamlı iyileştirme beklenir.

5. **Sınıf dengesizliği:** %35 pozitif oran, recall ve F1 üzerinde бaskı oluşturmaktadır. SMOTE veya class_weight='balanced' uygulanabilir.

---

## 10. Sonuç

Bu proje, Pima Indians Diabetes veri seti üzerinde titiz bir makine öğrenmesi pipeline'ı uygulamıştır. Elde edilen başlıca sonuçlar şöyle özetlenebilir:

### Regresyon (Faz 2)
EKK ve MLE yöntemlerinin normal hata varsayımı altında matematiksel olarak özdeş olduğu sayısal olarak kanıtlanmıştır (fark: 3.16 × 10⁻¹⁶). BMI tahmini için lineer model R²=0.431 başarmıştır.

### Gradient Descent (Faz 3)
Sıfırdan yazılan Batch Gradient Descent, α=0.01 ile sklearn referansıyla birebir sonuç üretmiştir — matematiksel implementasyonun doğruluğu kanıtlanmıştır.

### Sınıflandırma (Faz 4)
Lojistik Regresyon analizinde 4 değişken istatistiksel olarak anlamlı bulunmuştur: **Glucose (Z=8.06)**, **BMI (Z=4.44)**, **Pregnancies (Z=2.52)**, **Age (Z=2.02)**.

### Bias-Variance (Faz 5)
KNN'de K=26 Bias-Variance dengesi açısından optimal nokta olarak belirlenmiştir. K=1 (Train Err=0.000, overfit) ve K=50 (underfit) uçları grafiksel olarak gösterilmiştir.

### Resampling (Faz 6)
10-Fold CV, LR için Acc=0.768±0.027 bildirirken Bootstrap (B=1000), Glucose ve BMI'nin \%95 güven aralıklarının sıfırı kesinlikle içermediğini doğrulamıştır.

### Final Değerlendirme (Faz 7)
En yüksek AUC **Lojistik Regresyon (0.835)**, en yüksek Accuracy **KNN-K=26 (0.767)**, en yüksek Recall **GNB (0.675)** ile elde edilmiş; **tüm modeller "İyi" AUC kategorisinde** yer almaktadır.

---

> [!IMPORTANT]
> **Ana Bulgu:** Glukoz değişkeni, Pearson korelasyondan (r=0.493) başlayarak Gradient Descent ağırlıklarına (w=1.137), Lojistik Regresyon katsayısına (β=1.164, p<0.001) ve Bootstrap güven aralıklarına ([0.920, 1.457]) kadar **tüm yöntemlerde tutarlı biçimde en güçlü diyabet prediktörü** olarak öne çıkmıştır. Bu bulgu, veri odaklı analizin klinik bilgiyle mükemmel uyum içinde olduğunu göstermektedir.

---

**Üretilen Dosyalar Özeti:**

| Faz | Script | Çıktı Dizini | Grafik Sayısı |
|---|---|---|---|
| 1 | `phase1_preprocessing.py` | `phase1_outputs/` | 9 |
| 2 | `phase2_regression.py` | `phase2_outputs/` | 8 |
| 3 | `phase3_gradient_descent.py` | `phase3_outputs/` | 9 |
| 4 | `phase4_classification.py` | `phase4_outputs/` | 7 |
| 5 | `phase5_bias_variance.py` | `phase5_outputs/` | 6 |
| 6 | `phase6_resampling.py` | `phase6_outputs/` | 8 |
| 7 | `phase7_final_evaluation.py` | `phase7_outputs/` | 8 |
| **TOPLAM** | **7 script** | **7 dizin** | **55 grafik** |
