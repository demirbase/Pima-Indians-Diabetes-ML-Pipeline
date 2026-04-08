# FAZ 1 — Veri Ön İşleme ve Hazırlık Raporu
**Veri Seti:** Pima Indians Diabetes Dataset  
**Toplam Örnek:** 768 satır × 9 sütun  
**Hedef Değişken:** `Outcome` (0 = Diyabet Yok, 1 = Diyabet Var)  
**Script:** `phase1_preprocessing.py`  
**Çıktı Klasörü:** `phase1_outputs/`

---

## 1. Veri Seti Genel Bakış

| Özellik | Açıklama | Tür |
|---|---|---|
| Pregnancies | Gebelik sayısı | int |
| Glucose | Plazma glikoz konsantrasyonu (mg/dL) | float |
| BloodPressure | Diyastolik kan basıncı (mm Hg) | float |
| SkinThickness | Triceps deri kalınlığı (mm) | float |
| Insulin | 2 saatlik serum insülin (mu U/ml) | float |
| BMI | Vücut kitle indeksi | float |
| DiabetesPedigreeFunction | Diyabet soy ağacı fonksiyonu | float |
| Age | Yaş (yıl) | int |
| **Outcome** | **Diyabet tanısı (0/1)** | **int** |

**Sınıf dengesi:**  
- Negatif (0): **500 örnek (%65.1%)**  
- Pozitif (1): **268 örnek (%34.9%)**  
→ Hafif dengesiz — stratified split ile korundu.

---

## 2. Adım 1 — Biyolojik 0 Değer Analizi

Bazı sütunlarda `0` değeri biyolojik olarak imkânsızdır (örneğin Glukoz = 0 veya BMI = 0 yaşayan bir kişide olamaz). Bu değerler eksik veri olarak ele alındı.

![01_zero_value_counts.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/01_zero_value_counts.png)

| Sütun | 0 Değer Sayısı | Oranı | Doldurmada Kullanılan Medyan |
|---|---|---|---|
| Glucose | 5 | %0.7 | **117.00** |
| BloodPressure | 35 | %4.6 | **72.00** |
| SkinThickness | 227 | **%29.6** | **29.00** |
| Insulin | 374 | **%48.7** | **125.00** |
| BMI | 11 | %1.4 | **32.30** |

> [!WARNING]
> **Insulin** ve **SkinThickness** sütunlarındaki 0 oranları sırasıyla **%48.7** ve **%29.6** ile oldukça yüksektir. Bu durum, bu iki özelliğin model performansında istatistiksel gürültüye neden olabileceğini gösterir. İlerideki fazlarda bu özellikler için özellik seçimi tekrarlanmalıdır.

---

## 3. Adım 2 — Medyan ile Doldurma (Önce / Sonra)

**Yöntem:** Her sütun için yalnızca `≠ 0` olan değerlerin medyanı hesaplanarak sıfır değerler yerine konuldu.  
**Neden Medyan?** Aşırı uç değerlere (outlier) karşı ortalamaya kıyasla daha dayanıklıdır.

![03_before_after_comparison.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/03_before_after_comparison.png)

Ham veri dağılımları (0 değerler dahil):

![02_raw_distributions.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/02_raw_distributions.png)

**Temiz veri istatistikleri (doldurma sonrası):**

| Özellik | Min | Q1 | Medyan | Ortalama | Q3 | Max |
|---|---|---|---|---|---|---|
| Pregnancies | 0 | 1 | 3 | 3.85 | 6 | 17 |
| Glucose | 44 | 99.75 | 117 | 121.66 | 140.25 | 199 |
| BloodPressure | 24 | 64 | 72 | 72.39 | 80 | 122 |
| SkinThickness | 7 | 25 | 29 | 29.11 | 32 | 99 |
| Insulin | 14 | 121.5 | 125 | 140.67 | 127.25 | 846 |
| BMI | 18.2 | 27.5 | 32.3 | 32.46 | 36.6 | 67.1 |
| DiabetesPedigreeFunction | 0.08 | 0.24 | 0.37 | 0.47 | 0.63 | 2.42 |
| Age | 21 | 24 | 29 | 33.24 | 41 | 81 |

---

## 4. Adım 3 — Korelasyon Analizi

![04_correlation_heatmap.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/04_correlation_heatmap.png)

**Outcome ile korelasyon (Pearson, düşükten yükseğe):**

| Özellik | Korelasyon | Yorum |
|---|---|---|
| **Glucose** | **+0.493** | En güçlü prediktör |
| **BMI** | **+0.312** | İkinci en güçlü |
| **Age** | **+0.238** | Orta düzey |
| **Pregnancies** | +0.222 | Orta düzey |
| SkinThickness | +0.215 | Zayıf–orta |
| Insulin | +0.204 | Zayıf (yüksek 0 oranı etkisi) |
| DiabetesPedigreeFunction | +0.174 | Zayıf |
| BloodPressure | +0.166 | En zayıf |

> [!TIP]
> Glukoz tek başına en ayırt edici özelliktir. KNN için özellik ağırlıklandırması yapılacaksa **Glucose → BMI → Age** önceliklendirilmelidir.

---

## 5. Adım 4 — Veriyi Bölme (%70 / %15 / %15)

**Strateji:** Önce tüm veriden `%15 Test` ayrıldı (holdout). Kalan `%85`'ten `%15 Val` ayrıldı. Tüm split'ler `stratify=y` ile yapıldı.

> [!IMPORTANT]
> Test seti en başta ayrılarak **veri sızıntısı (data leakage)** önlendi. Scaler yalnızca Train setine `fit` edildi; Val ve Test setleri yalnızca `transform` edildi.

![05_split_distribution.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/05_split_distribution.png)

| Set | Satır | Oran | Negatif (0) | Pozitif (1) | Pozitif % |
|---|---|---|---|---|---|
| **Train** | **536** | **%69.8** | 349 | 187 | **%34.9** |
| **Val** | **116** | **%15.1** | 75 | 41 | **%35.3** |
| **Test** | **116** | **%15.1** | 76 | 40 | **%34.5** |

✅ **Stratified split başarılı:** Üç sette de sınıf oranı ~%35 pozitif olarak korundu.

---

## 6. Adım 5 — StandardScaler ile Ölçeklendirme

**Scaler Train setine fit edildi → Val ve Test'e yalnızca transform uygulandı.**

| Özellik | μ (Train) | σ (Train) |
|---|---|---|
| Pregnancies | 3.8340 | 3.3824 |
| Glucose | 121.5280 | 29.4890 |
| BloodPressure | 71.8302 | 12.3317 |
| SkinThickness | 28.8153 | 9.0939 |
| Insulin | 137.9067 | 78.6393 |
| BMI | 32.2326 | 6.6789 |
| DiabetesPedigreeFunction | 0.4688 | 0.3244 |
| Age | 33.5392 | 11.8155 |

**Ölçeklendirme sonrası Train kontrolü:** μ ≈ 0.000, σ ≈ 1.001 ✅

````carousel
![Violin — Ölçeklendirme Öncesi/Sonrası](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/06_scaling_violin.png)
<!-- slide -->
![Box-Plot — Train/Val/Test Karşılaştırması](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/07_scaled_boxplots.png)
<!-- slide -->
![Pair Plot — Top-4 Özellik](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/08_pair_plot.png)
````

**Pair plot gözlemi:** Ölçeklenmiş Train setinde Glucose ve BMI, iki sınıfı nispeten iyi ayırt etmektedir. Age ile Glucose birlikte kullanıldığında sınıflar arasındaki örtüşme azalmaktadır.

---

## 7. Özet Dashboard

![09_summary_dashboard.png](/Users/erendemirbas/.gemini/antigravity/brain/64540d57-931a-4c9e-9576-bb597ce82e72/09_summary_dashboard.png)

---

## 8. Üretilen Dosyalar

| Dosya | Satır | Açıklama |
|---|---|---|
| `train_raw.csv` | 536 | Ham (doldurulmuş, ölçeklenmemiş) train seti |
| `val_raw.csv` | 116 | Ham validation seti |
| `test_raw.csv` | 116 | Ham test seti |
| `train_scaled.csv` | 536 | Ölçeklenmiş train seti |
| `val_scaled.csv` | 116 | Ölçeklenmiş validation seti |
| `test_scaled.csv` | 116 | Ölçeklenmiş test seti |

---

## 9. Özet ve Sonraki Adım İçin Öneriler

> [!NOTE]
> **Faz 1 başarıyla tamamlandı.** Veri temiz, bölünmüş ve ölçeklendirilmiş haldedir.

**Yapılanlar:**
- ✅ 652 adet biyolojik olarak imkânsız `0` değeri medyan ile dolduruldu
- ✅ Stratified 70/15/15 bölünmesi uygulandı (sınıf dengesi korundu)
- ✅ StandardScaler yalnızca Train'e fit edildi → veri sızıntısı yok
- ✅ 9 koyu tema grafik üretildi, 6 CSV dosyası kaydedildi

**Faz 2 için dikkat edilmesi gerekenler:**
- `train_scaled.csv` → **KNN** için hazır (Öklid mesafesi z-skor'larla çalışır)
- `train_raw.csv` → Karar ağacı / lojistik regresyon için kullanılabilir
- **Insulin** ve **SkinThickness** sütunlarındaki yüksek 0 oranı nedeniyle bu özelliklerin model katkısı Faz 2'de ayrıca değerlendirilmeli
- Sınıf imbalance (%35 pos) ilerleyen fazlarda `class_weight='balanced'` veya SMOTE ile ele alınabilir
