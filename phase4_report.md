# FAZ 4 — Sınıflandırma ve İstatistiksel Yorum Raporu

**Hedef:** Outcome (Diyabet = 1 / Yok = 0)
**Modeller:** Lojistik Regresyon · KNN · Gaussian Naive Bayes
**Script:** `phase4_classification.py` | **Çıktı:** `phase4_outputs/`

---

## A — Lojistik Regresyon: İstatistiksel Özet

Std. hatalar Fisher Bilgi Matrisi ile hesaplandı: `Cov(beta) = (X'WX)^{-1}`
Z-istatistiği: `Z = beta / SE(beta)` → |Z| > 1.96 ise p < 0.05

### A1 — Katsayı Tablosu

| Parametre | beta | SE | Z | p-degeri | Anlamlı | Odds Orani |
|---|---|---|---|---|---|---|
| intercept | -0.8717 | 0.1170 | -7.454 | 0.0000 | **★** | 0.418 |
| **Glucose** | **+1.1640** | 0.1444 | **+8.060** | **0.0000** | **★** | **3.203** |
| **BMI** | **+0.6595** | 0.1488 | **+4.435** | **0.0000** | **★** | **1.934** |
| **Pregnancies** | **+0.3245** | 0.1290 | **+2.516** | **0.0119** | **★** | **1.383** |
| **Age** | **+0.2685** | 0.1330 | **+2.018** | **0.0436** | **★** | **1.308** |
| DiabetesPedigreeFunction | +0.2190 | 0.1152 | +1.901 | 0.0573 | — | 1.245 |
| BloodPressure | -0.0841 | 0.1268 | -0.663 | 0.5071 | — | 0.919 |
| SkinThickness | +0.0518 | 0.1419 | +0.365 | 0.7152 | — | 1.053 |
| Insulin | -0.0231 | 0.1255 | -0.184 | 0.8544 | — | 0.977 |

> [!IMPORTANT]
> **p < 0.05 olan anlamlı degiskenler: Glucose, BMI, Pregnancies, Age**
> BloodPressure, SkinThickness, Insulin istatistiksel olarak anlamsızdır (p > 0.5).

![01_logistic_stats.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_logistic_stats.png)

### A2 — Guven Araligi Forest Plot

![02_forest_plot.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_forest_plot.png)

### A3 — Log-Odds Yorumları (z-skor uzayında)

**Glucose** (beta = 1.164, Z = 8.06, p < 0.001):
Glukoz degeri 1 standart sapma arttığında, diyabet olma ihtimalinin log-odds degeri **1.164 artis gosterir**.
Odds orani: e^1.164 = **3.203** → Diyabet olma sansi 3.2x carpilir.

**BMI** (beta = 0.660, Z = 4.44, p < 0.001):
BMI 1 std sapma arttığında log-odds **0.660 artar**.
Odds orani: **1.934** → ~%93 sans artisi.

**Pregnancies** (beta = 0.325, Z = 2.52, p = 0.012):
1 std sapma artis → log-odds **0.325 artar**.
Odds orani: **1.383** → ~%38 artis.

**Age** (beta = 0.269, Z = 2.02, p = 0.044):
Yas 1 std sapma arttığında log-odds **0.269 artar**.
Odds orani: **1.308** → En zayif anlamli etki.

> [!NOTE]
> **DiabetesPedigreeFunction** (p = 0.057) esığın cok az uzerinde — marjinal anlamlilik. Daha buyuk bir calismada anlamli cikabilir.

---

## B — KNN (K = 1…20, Oklid Mesafesi)

![03_knn_k_analysis.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_knn_k_analysis.png)

### B1 — K Sonucları

| K | Val Acc | Test Acc | AUC |
|---|---|---|---|
| 1 | 0.6897 | 0.7241 | 0.683 |
| 5 | 0.7069 | 0.7414 | 0.763 |
| 10 | 0.7155 | 0.7586 | 0.798 |
| 15 | 0.7500 | 0.7845 | 0.822 |
| **20** | **0.7672** | **0.7845** | **0.825** ◀ En iyi |

> [!TIP]
> **Best K = 20.** K=1 → overfit (Train≈1.0, Test=0.72). K=20 → bias-variance dengesi.

### B2 — Bias-Variance Dengesi

![04_knn_bias_variance.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_knn_bias_variance.png)

---

## C — Gaussian Naive Bayes

Varsayım: Her ozellik, her sinif icin Gaussian dagilir: `P(xj|y=k) ~ N(mujk, sigma^2_jk)`

| Set | Accuracy | AUC-ROC |
|---|---|---|
| Validation | 0.7672 | 0.8000 |
| Test | 0.7155 | 0.7786 |

![05_gnb_distributions.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_gnb_distributions.png)

---

## D — Model Karşılaştırması (Test Seti)

| Model | Accuracy | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Lojistik Reg. | 0.7069 | **0.8352** | 0.564 | 0.579 | 0.550 |
| **KNN (K=20)** | **0.7845** | 0.8253 | **0.667** | **0.714** | 0.625 |
| Gaussian NB | 0.7155 | 0.7786 | 0.621 | 0.574 | **0.675** |

````carousel
![Model Karşılaştırma Dashboard](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_model_comparison_dashboard.png)
<!-- slide -->
![Karışıklık Matrisleri](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_confusion_matrices.png)
````

| Kriter | En İyi |
|---|---|
| **Accuracy** | KNN (K=20) = 0.784 |
| **AUC-ROC** | Lojistik Reg. = 0.835 |
| **Recall** | GNB = 0.675 |
| **F1** | KNN = 0.667 |

> [!IMPORTANT]
> Klinik bağlamda Recall kritiktir (FN = hasta kacirmak tehlikeli). **GNB en yuksek Recall (0.675).** Genel performans icin **KNN (K=20)** en dengeli model.

---

## E — Fazlar Arası Bütünleşik Tablo

| Değişken | Faz 1 (r) | Faz 3 GD (w) | Faz 4 LR (beta) | p | Anlamlı |
|---|---|---|---|---|---|
| Glucose | +0.493 | +1.137 | **+1.164** | <0.001 | ★★★ |
| BMI | +0.312 | +0.629 | **+0.660** | <0.001 | ★★★ |
| Pregnancies | +0.222 | +0.317 | **+0.325** | 0.012 | ★★ |
| Age | +0.238 | +0.267 | **+0.269** | 0.044 | ★ |
| BloodPressure | +0.166 | -0.070 | -0.084 | 0.507 | — |
| SkinThickness | +0.215 | +0.072 | +0.052 | 0.715 | — |
| Insulin | +0.204 | -0.007 | -0.023 | 0.854 | — |

Tüm fazlarda **Glucose tutarlı olarak en güçlü prediktör** çıkmaktadır.
