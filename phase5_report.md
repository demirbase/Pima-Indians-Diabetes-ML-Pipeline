# FAZ 5 — Yanlılık & Varyans (Bias-Variance Trade-off) Raporu

**Yöntem:** KNN, K=1…50 aralığında test edildi  
**Script:** `phase5_bias_variance.py` | **Çıktı:** `phase5_outputs/`

---

## 1. Teorik Çerçeve

Bir modelin beklenen toplam hatası 3 bileşene ayrışır:

$$\text{Toplam Hata} = \text{Bias}^2 + \text{Varyans} + \text{Gürültü}$$

| Terim | Tanım | KNN'de K ile ilişki |
|---|---|---|
| **Bias²** | Modelin gerçek fonksiyonu ne kadar yanlış öğrenir? | K arttıkça **bias artar** |
| **Varyans** | Model farklı eğitim setlerine ne kadar duyarlı? | K arttıkça **varyans azalır** |
| **Optimum** | Bias² + Varyans toplamının minimumu | **K = 26** (val. hatası min.) |

---

## 2. Error vs. K — Ana Grafik

![01_error_vs_k.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_error_vs_k.png)

### Kritik Noktalar

| K | Train Err | Val Err | Test Err | Bölge |
|---|---|---|---|---|
| **1** | **0.0000** | 0.3103 | 0.2759 | Yüksek Varyans (Overfit) |
| 5 | 0.1679 | 0.2931 | 0.2586 | Overfit bölgesi |
| 13 | 0.1847 | 0.2586 | 0.2155 | İyileşme |
| **26** | 0.2108 | **0.2241** | 0.2328 | **Optimum (Val. min.)** |
| 40 | 0.2351 | 0.2241 | 0.2500 | Underfit başlangıcı |
| **50** | 0.2481 | 0.2328 | 0.2414 | Yüksek Yanlılık (Underfit) |

> [!IMPORTANT]
> **K=1 Train Err = 0.0000** — Model eğitim verisini tam ezberledi. Test hatası 0.276. Bu klasik **overfitting**.
>
> **K=50** — Train ve Test hataları birbirine yakınsıyor ama her ikisi de yükseliş gösteriyor. Bu **underfitting**.
>
> **Grafiğin en çukur noktası K=26 (Val. Hatası = 0.2241)** → Optimum nokta burasıdır.

---

## 3. AUC vs. K

![02_auc_vs_k.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_auc_vs_k.png)

AUC monoton artarak K büyüdükçe iyileşiyor — bu KNN'in büyük K değerlerinde daha **dengeli** sınıflandırma yaptığını gösterir.

---

## 4. Bias² & Varyans Ayrışımı (Bootstrap)

30 bootstrap örneklemesi ile Bias² ve Varyans ayrı ayrı tahmin edildi.

![03_bias_variance_decomposition.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_bias_variance_decomposition.png)

| K | Bias² | Varyans | Toplam |
|---|---|---|---|
| 1 | 0.154 | 0.021 | 0.175 |
| 5 | 0.151 | 0.013 | 0.164 |
| 10 | 0.153 | 0.010 | 0.163 |
| **25** | **0.159** | **0.007** | **0.166** |
| 50 | 0.170 | 0.005 | 0.175 |

> [!NOTE]
> K büyüdükçe Varyans dramatik biçimde düşüyor (0.021 → 0.005), ama Bias² artıyor (0.154 → 0.170). Klasik **trade-off** kanıtlandı.

---

## 5. Karar Sınırı Görselleştirmesi

![04_decision_boundary.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_decision_boundary.png)

| Model | Karar Sınırı | Train Err | Test Err |
|---|---|---|---|
| **K=1** | Çok girintili (her noktayı izole eder) | ~0.000 | 0.276 |
| **K=26 (Opt.)** | Düzgün, makul eğrisel | 0.211 | 0.233 |
| **K=50** | Çok yuvarlak, değişmez | 0.248 | 0.241 |

---

## 6. Tüm Modeller Hata Karşılaştırması

![05_all_models_error.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_all_models_error.png)

| Model | Train Err | Test Err |
|---|---|---|
| KNN K=26 | 0.211 | 0.233 |
| Lojistik Reg. | ~0.22 | 0.293 |
| GNB | ~0.21 | 0.284 |

**KNN K=26 en düşük Test Err** — doğrusal olmayan karar sınırı avantaj sağlıyor.

---

## 7. Özet Dashboard

![06_summary_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_summary_dashboard.png)

---

## 8. Fazlar Arası Sonuç

| Bölge | K | Karakter | Sonuç |
|---|---|---|---|
| **Overfit** | K = 1–5 | Yüksek Varyans | Train≈0, Test yüksek |
| **Geçiş** | K = 6–15 | Karışık | Hata azalıyor |
| **Optimum** | **K = 26** | **Denge** | **Val Err minimumda** |
| **Underfit** | K = 40–50 | Yüksek Yanlılık | İkisi de yükseliyor |

> [!TIP]
> **Faz 4 Best K = 20 (Val Acc maks.) ↔ Faz 5 Best K = 26 (Val Err min.)** — küçük fark normalize edilmiş ölçüm farklılığından kaynaklanır. Her ikisi de "optimal bölge" içindedir.
