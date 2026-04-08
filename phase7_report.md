# FAZ 7 — Başarı Ölçümü & Final Model Raporu

**Script:** `phase7_final_evaluation.py` | **Çıktı:** `phase7_outputs/`

---

## 1. ROC Eğrisi — Tüm Modeller (Aynı Grafikte)

![01_roc_curves.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_roc_curves.png)

### AUC Skorları (Test Seti)

| Model | AUC-ROC | Yorum |
|---|---|---|
| **Lojistik Reg.** | **0.8352** | ★ En iyi AUC — "İyi Eşik"in üstünde |
| KNN (K=26) | 0.8268 | ★ İyi — LR'ye çok yakın |
| Gaussian NB | 0.7786 | Kabul edilebilir |
| Rastgele sınıflandırıcı | 0.5000 | Referans (yazı-tura) |

> [!IMPORTANT]
> **AUC Yorumu:**
> - **1.0** → Mükemmel (gerçek hayatta olmaz)
> - **0.8–0.9** → İyi ✅ (bizim modellerimiz bu bölgede)
> - **0.7–0.8** → Kabul edilebilir
> - **0.5** → Yazı-tura kadar işe yaramaz (rastgele)
>
> Lojistik Regresyon AUC=0.8352 ile **her olası eşikte** KNN ve GNB'den daha iyi ayrım yapıyor.

---

## 2. Precision-Recall Eğrisi

![02_precision_recall.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_precision_recall.png)

Sınıf dengesizliği (%35 pozitif) olduğunda ROC yerine **PR eğrisi daha bilgilendirici**:

| Model | Average Precision (AP) |
|---|---|
| Lojistik Reg. | En yüksek |
| KNN (K=26) | İkinci |
| GNB | Üçüncü |

---

## 3. AUC Karşılaştırma

![03_auc_comparison.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_auc_comparison.png)

---

## 4. Sınıflandırma Final Metrikleri (Test Seti)

| Model | Accuracy | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Lojistik Reg.** | 0.7069 | **0.8352** | 0.5641 | 0.5789 | 0.5500 |
| **KNN (K=26)** | **0.7672** | 0.8268 | **0.6301** | **0.6970** | 0.5750 |
| Gaussian NB | 0.7155 | 0.7786 | 0.6207 | 0.5745 | **0.6750** |

````carousel
![06_final_model_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_final_model_dashboard.png)
<!-- slide -->
![08_summary_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/08_summary_dashboard.png)
````

---

## 5. Regresyon Metrikleri — BMI Tahmini

**Model:** Lineer Regresyon (EKK / MLE — Faz 2 sonuçları)
**Hedef:** BMI değerini diğer özelliklerden tahmin et

| Set | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Validation | 0.7707 | 0.8779 | 0.6563 | 0.3064 |
| **Test** | **0.7140** | **0.8450** | **0.6622** | **0.4314** |

> [!NOTE]
> **MSE = 0.714** — standardize edilmiş z-skor uzayında ölçülmüştür. Orijinal BMI birimlerine çevrildiğinde ~31.85 kg²/m⁴ (Faz 2 çıktısıyla tutarlı).
>
> **R² = 0.43** — Model BMI varyansının %43'ünü açıklıyor. Doğrusal model orta düzey; daha iyi R² için non-lineer modeller gerekebilir.

![04_regression_results.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_regression_results.png)

![05_residual_analysis.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_residual_analysis.png)

---

## 6. Kalibrasyon Eğrisi

![07_calibration_curve.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_calibration_curve.png)

Çapraz çizgiye (mükemmel kalibrasyon) en yakın model **Lojistik Regresyon** — olasılık tahminleri en güvenilir.

---

## 7. Final Model Seçimi

| Kriter | Seçim | Gerekçe |
|---|---|---|
| **AUC-ROC** (Genel Ayrımcılık) | **Lojistik Reg. (0.835)** | En geniş ROC eğri alanı |
| **Accuracy** (Doğruluk) | **KNN K=26 (0.767)** | Doğrusal olmayan sınırlar avantaj sağlıyor |
| **Recall** (Hasta kaçırmama) | **GNB (0.675)** | Klinik bağlamda kritik |
| **Yorumlanabilirlik** | **Lojistik Reg.** | p-değeri, z-istatistiği, log-odds hesaplanabilir |
| **Regresyon (BMI)** | **Lineer Reg.** | RMSE=0.845, R²=0.43 |

> [!TIP]
> **Klinik bir karar destek sisteminde Lojistik Regresyon önerilir:**
> - AUC en yüksek (0.835) → her eşikte en iyi ayrım
> - Katsayılar yorumlanabilir (β, p-değeri, Odds Oranı)
> - İstatistiksel güven aralıkları Bootstrap ile doğrulandı (Faz 6)
>
> **Ham doğruluk için KNN (K=26):** %76.7 ile en yüksek accuracy.

---

## 8. Tüm Fazlar Özeti

| Faz | Konu | Kilit Sonuç |
|---|---|---|
| 1 | Preprocessing | 0-değer impütasyonu, %70/%15/%15 bölme |
| 2 | Regresyon (EKK=MLE) | β fark = 3.16×10⁻¹⁶, R²=0.43 |
| 3 | Gradient Descent | α=0.01 optimal, Acc=0.776 |
| 4 | Sınıflandırma + İstat. | Glucose (p<0.001, Z=8.06) en anlamlı |
| 5 | Bias-Variance | K=26 optimal, K=1 overfit, K=50 underfit |
| 6 | CV + Bootstrap | CV Acc=0.768±0.027, 4 anlamlı katsayı |
| **7** | **Final** | **LR AUC=0.835, KNN Acc=0.767, R²=0.43** |
