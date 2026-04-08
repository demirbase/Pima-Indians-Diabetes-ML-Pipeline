# FAZ 6 — Resampling: Çapraz Geçerleme & Bootstrap Raporu

**Script:** `phase6_resampling.py` | **Çıktı:** `phase6_outputs/`

---

## A — 10-Fold Stratified Cross Validation

### Neden CV?
Tek seferlik bölme (hold-out) **şans eseri iyi/kötü sonuç** verebilir. 10-Fold CV, veriyi 10 farklı şekilde eğitim/test olarak böler — her fold bir kez test seti olur. Sonuç **istatistiksel olarak güvenilir bir ortalama ve standart sapma** sağlar.

### CV Sonuçları (Accuracy & F1)

| Model | CV Acc μ ± σ | CV F1 μ ± σ | Yorum |
|---|---|---|---|
| **Lojistik Reg.** | **0.7682 ± 0.0268** | **0.6297 ± 0.0438** | En tutarlı |
| KNN (K=26) | 0.7630 ± 0.0172 | 0.6024 ± 0.0426 | **En düşük σ** (stabil) |
| Gaussian NB | 0.7447 ± 0.0479 | 0.6196 ± 0.0670 | En yüksek varyans |

> [!IMPORTANT]
> **KNN (K=26) en düşük standart sapmaya (σ=0.017)** sahip — fold'lar arası tutarlılık en yüksek bu modelde. GNB'nin yüksek σ'su (0.048) fold'a göre değişkenlik gösterdiğini işaret eder.

````carousel
![10-Fold CV Fold Sonuçları](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_cv_fold_results.png)
<!-- slide -->
![CV Violin Plot](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_cv_violin.png)
<!-- slide -->
![CV Özet Bar](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_cv_summary_bar.png)
````

---

## B — Bootstrap (B = 1000): Lojistik Regresyon Katsayıları

### Yöntem
1. Train+Val setinden **yerine koyarak** 1000 kez örneklem çek (n=652)
2. Her örneklemde StandardScaler + LogisticRegression fit et
3. Katsayıların **empirik dağılımını** elde et
4. **%95 Percentile CI** ve **BCa (Bias-Corrected Accelerated) CI** hesapla

### Katsayı Güven Aralıkları

| Parametre | β (orijinal) | Boot μ | Boot σ | 95% CI (Pct) | Anlamlı |
|---|---|---|---|---|---|
| **Glucose** | **1.1497** | 1.1752 | 0.1403 | [0.920, 1.457] | **★** |
| **BMI** | **0.6550** | 0.6684 | 0.1305 | [0.406, 0.928] | **★** |
| **Pregnancies** | **0.4064** | 0.4065 | 0.1386 | [0.136, 0.674] | **★** |
| **DiabetesPedigreeFunction** | **0.2444** | 0.2479 | 0.1216 | [0.013, 0.485] | **★** |
| Age | 0.1206 | 0.1282 | 0.1357 | [-0.137, 0.391] | — |
| BloodPressure | -0.0916 | -0.0927 | 0.1052 | [-0.285, 0.123] | — |
| SkinThickness | 0.0638 | 0.0669 | 0.1231 | [-0.171, 0.318] | — |
| Insulin | -0.0281 | -0.0341 | 0.1413 | [-0.318, 0.227] | — |

> [!IMPORTANT]
> **CI sıfırı içermeyen (★) = Bootstrap ile istatistiksel anlamlılık.**
>
> Faz 4'te (Fisher matrisi) 4 anlamlı değişken vardı (Glucose, BMI, Pregnancies, Age).
> Bootstrap'ta Age CI'ı sıfırı kapsıyor [-0.137, 0.391] → **Bootstrap age'i daha muhafazakâr değerlendiriyor**.
> Buna karşılık **DiabetesPedigreeFunction [0.013, 0.485]** marginally anlamlı çıktı — Faz 4'te p=0.057 idi.

### Katsayı Dağılım Histogramları

![04_bootstrap_histograms.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_bootstrap_histograms.png)

Her histogram **B=1000 katsayı tahmininin dağılımını** göstermektedir. Glucose ve BMI çan eğrisi oluşturmakta, düzgün simetrik — güvenilir tahminler.

### Bootstrap Forest Plot (Percentile vs. BCa)

![05_bootstrap_forest.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_bootstrap_forest.png)

---

## C — Bootstrap Accuracy Dağılımı

![06_bootstrap_accuracy.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_bootstrap_accuracy.png)

| İstatistik | Değer |
|---|---|
| Bootstrap Accuracy μ | **0.7833** |
| Bootstrap Accuracy σ | 0.0173 |
| 95% CI (Percentile) | [0.7485, 0.8160] |
| Q-Q R² | ~0.99 → Normal dağılıma yakın ✅ |

> [!NOTE]
> Bootstrap accuracy'nin normal dağılıma yakın olması (Q-Q R²≈0.99), **Merkezi Limit Teoremi'nin işlediğini** gösterir. Bu, güven aralıklarının güvenilir olduğunu doğrular.

---

## D — CV vs. Hold-Out Karşılaştırması

![07_cv_vs_holdout.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_cv_vs_holdout.png)

| Model | CV Acc μ | Hold-Out Acc | Fark |
|---|---|---|---|
| Lojistik Reg. | 0.7682 | ~0.707 | CV daha iyimser |
| KNN (K=26) | 0.7630 | ~0.784 | Hold-out daha iyi |
| GNB | 0.7447 | ~0.716 | Benzer |

> [!TIP]
> Hold-out test setinin CV'den daha **iyi ya da daha kötü** çıkması normaldir — hold-out tek bir bölme, CV 10 farklı bölmenin ortalaması. Fark büyük değilse (**< ~5pp**) model güvenilirdir.

---

## E — Özet Dashboard

![08_summary_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/08_summary_dashboard.png)

---

## F — Tüm Fazlarla Tutarlılık Tablosu

| Değişken | Faz 4 LR (p) | Faz 4 Bootstrap | Faz 6 Bootstrap | Tutarlı? |
|---|---|---|---|---|
| Glucose | p<0.001 ★ | — | [0.920, 1.457] ★ | ✅ |
| BMI | p<0.001 ★ | — | [0.406, 0.928] ★ | ✅ |
| Pregnancies | p=0.012 ★ | — | [0.136, 0.674] ★ | ✅ |
| DiabetesPedigreeFunction | p=0.057 (marginal) | — | [0.013, 0.485] ★ | ✅ (Bootstrap daha kuvvetli) |
| Age | p=0.044 ★ | — | [-0.137, 0.391] — | ⚠️ (Bootstrap daha muhafazakâr) |
| Insulin | p=0.854 | — | [-0.318, 0.227] — | ✅ |
