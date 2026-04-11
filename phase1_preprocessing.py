"""
=============================================================
  FAZ 1: VERİ ÖN İŞLEME ve HAZIRLIK (Preprocessing)
  Pima Indians Diabetes Dataset
=============================================================
  Adımlar:
    1. Veriyi yükle ve keşfet (EDA)
    2. Biyolojik olarak imkânsız 0 değerlerini doldur (medyan)
    3. Veriyi %70 Train / %15 Val / %15 Test olarak ayır
    4. StandardScaler ile ölçeklendir
    5. Tüm adımları koyu tema grafiklerle görselleştir
    6. Temiz setleri CSV olarak kaydet
=============================================================

Modül Amacı
-----------
  Bu modül, Pima Indians Diabetes veri seti üzerinde 7 fazlı
  makine öğrenmesi pipeline'ının ilk halkasını oluşturur.
  Sonraki tüm fazlar (Faz 2–7) bu modülün ürettiği
  ``phase1_outputs/``  dizinindeki CSV dosyalarını girdi olarak
  kullanır.

Veri Seti Hakkında
------------------
  Kaynak : Ulusal Diyabet ve Sindirim ve Böbrek Hastalıkları
           Enstitüsü (NIDDK)
  Örneklem: Yalnızca kadın, Pima Indian kökenli, ≥21 yaş
  Boyut  : 768 örnek × 9 değişken (8 özellik + 1 hedef)
  Sınıf  : %65.1 Negatif (Outcome=0) / %34.9 Pozitif (Outcome=1)
  → FINAL_REPORT.md §2 ve phase1_report.md'ye bakınız.

Girdiler
--------
  diabetes.csv : Ham veri dosyası (proje kök dizininde)

Çıktılar
--------
  phase1_outputs/train_raw.csv      – Ham (ölçeklenmemiş) train seti
  phase1_outputs/val_raw.csv        – Ham validation seti
  phase1_outputs/test_raw.csv       – Ham test seti
  phase1_outputs/train_scaled.csv   – z-skor ölçeklenmiş train seti
  phase1_outputs/val_scaled.csv     – z-skor ölçeklenmiş val seti
  phase1_outputs/test_scaled.csv    – z-skor ölçeklenmiş test seti
  phase1_outputs/*.png              – 9 adet görselleştirme grafiği
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  KOYU TEMA AYARI
#  GitHub dark-mode renk paleti kullanılmaktadır.
#  Tüm fazlarda tutarlılık için bu değerler
#  aynen korunmalıdır (FINAL_REPORT.md'de
#  referans alınan görseller bu temaya göre
#  üretilmiştir).
# ─────────────────────────────────────────────
DARK_BG    = "#0d1117"
CARD_BG    = "#161b22"
ACCENT1    = "#58a6ff"   # mavi
ACCENT2    = "#3fb950"   # yeşil
ACCENT3    = "#ff7b72"   # kırmızı
ACCENT4    = "#d2a8ff"   # mor
ACCENT5    = "#ffa657"   # turuncu
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"

PALETTE_COLS = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
                "#79c0ff", "#56d364", "#f47067", "#bc8cff", "#ffb86c"]

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

OUTPUT_DIR = "phase1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def save_fig(fname: str):
    """
    Mevcut Matplotlib figürünü ``OUTPUT_DIR`` altına kaydeder ve kapatır.

    Parametreler
    ------------
    fname : str
        Kaydedilecek dosyanın adı (örn. "01_zero_value_counts.png").
        Yol otomatik olarak ``OUTPUT_DIR`` ile birleştirilir.

    Notlar
    ------
    - ``facecolor=DARK_BG`` ile arka plan koyu temada kaydedilir;
      bu sayede açık tema ekranlarda bile görsel tutarlılığı sağlanır.
    - ``bbox_inches="tight"`` eksen etiketlerinin kırpılmasını önler.
    """
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches="tight", facecolor=DARK_BG, dpi=150)
    plt.close()
    print(f"  ✅  Grafik kaydedildi → {path}")


def section_header(title: str):
    """
    Konsol çıktısına görsel ayırıcı başlık basar.

    Parametreler
    ------------
    title : str
        Bölüm başlığı metni.

    Notlar
    ------
    ASCII sanat formatı tüm fazlarda aynı kalmalıdır;
    bu format phase1_report.md'deki bölüm yapısıyla örtüşür.
    """
    line = "═" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ═══════════════════════════════════════════════════════════
#  ADIM 0 – VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════
section_header("ADIM 0 — VERİ YÜKLEME")

# Ham CSV dosyasını belleğe al.
# Veri seti 768 satır (örnek) × 9 sütun (8 özellik + Outcome).
df = pd.read_csv("diabetes.csv")
print(f"\n  Satır: {df.shape[0]}  |  Sütun: {df.shape[1]}")
print("\n  İlk 5 satır:")
print(df.head().to_string(index=False))
print("\n  Veri tipleri:")
print(df.dtypes.to_string())
print("\n  Temel İstatistikler:")
print(df.describe().round(2).to_string())

# Bağımsız değişkenler (özellikler) ve hedef değişken tanımı.
# FINAL_REPORT.md §2.1 tablosuna bakınız.
FEATURE_COLS = ["Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age"]
TARGET_COL   = "Outcome"

# Biyolojik olarak 0 olamayacak sütunlar.
# Pregnancies ve Age için 0 olası olduğundan listeye alınmaz.
# FINAL_REPORT.md §2.1 ve phase1_report.md — Veri Kalitesi bölümüne bakınız.
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ═══════════════════════════════════════════════════════════
#  ADIM 1A – SIFIR DEĞER ANALİZİ GRAFİĞİ (Ham Veri)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 1A — SIFIR DEĞER ANALİZİ (Ham Veri)")

# Her sütundaki 0 değerlerini say ve oranını hesapla.
# Örnek: Insulin'de 374 sıfır (%48.7) — bu oran, later fazlarda
# Insulin katsayısının neden anlamsız çıktığının temel gerekçesidir.
# FINAL_REPORT.md §9.2'ye ("Anlamsız Değişkenler Meselesi") bakınız.
zero_counts = (df[ZERO_COLS] == 0).sum()
zero_pct    = (zero_counts / len(df) * 100).round(1)

print("\n  Biyolojik 0 değer sayıları:")
for col in ZERO_COLS:
    print(f"    {col:<25} {zero_counts[col]:>4} adet  ({zero_pct[col]:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Biyolojik Olarak İmkânsız '0' Değerleri — Ham Veri",
             fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)

# Bar grafik – sayı
bars = axes[0].bar(zero_counts.index, zero_counts.values,
                   color=PALETTE_COLS[:len(ZERO_COLS)],
                   edgecolor=DARK_BG, linewidth=1.2)
axes[0].set_title("Sıfır Değer Sayısı", fontsize=12, pad=10)
axes[0].set_ylabel("Adet", fontsize=10)
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(axis="y")
for bar, val in zip(bars, zero_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, str(val),
                 ha="center", va="bottom", fontsize=9,
                 color=TEXT_COLOR, fontweight="bold")

# Bar grafik – yüzde
bars2 = axes[1].bar(zero_pct.index, zero_pct.values,
                    color=PALETTE_COLS[:len(ZERO_COLS)],
                    edgecolor=DARK_BG, linewidth=1.2)
axes[1].set_title("Sıfır Değer Oranı (%)", fontsize=12, pad=10)
axes[1].set_ylabel("Yüzde (%)", fontsize=10)
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(axis="y")
for bar, val in zip(bars2, zero_pct.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"{val}%",
                 ha="center", va="bottom", fontsize=9,
                 color=TEXT_COLOR, fontweight="bold")

plt.tight_layout()
save_fig("01_zero_value_counts.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 1B – HAM VERİ DAĞILIM GRAFİĞİ
# ═══════════════════════════════════════════════════════════
section_header("ADIM 1B — HAM VERİ DAĞILIM GRAFİĞİ")

# Ham dağılımlar çizildiğinde sıfır değerleri (yapay) spike'lar olarak görünür.
# Bu grafik, medyan doldurma ihtiyacını görsel olarak kanıtlar.
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Ham Veri — Özellik Dağılımları (0 değerleri dahil)",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

for i, (col, ax) in enumerate(zip(FEATURE_COLS, axes.flatten())):
    color = PALETTE_COLS[i % len(PALETTE_COLS)]
    ax.hist(df[col], bins=30, color=color, edgecolor=DARK_BG,
            linewidth=0.8, alpha=0.85)
    # Ortalama ve medyan çizgileri: eğer dağılım simetrik değilse
    # (özellikle 0 değerleri sonrası), ortalama sola kayar; bu
    # impütasyon için medyan tercihini güçlendirir.
    ax.axvline(df[col].mean(),   color=ACCENT3, linestyle="--",
               linewidth=1.4, label="Ort.")
    ax.axvline(df[col].median(), color=ACCENT2, linestyle=":",
               linewidth=1.4, label="Med.")
    ax.set_title(col, fontsize=11, pad=6)
    ax.set_xlabel("Değer", fontsize=9)
    ax.set_ylabel("Frekans", fontsize=9)
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
save_fig("02_raw_distributions.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 2 – 0 DEĞERLERİNİ MEDYAN İLE DOLDURMA
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2 — MEDYAN İLE DOLDURMA")

# Neden Medyan?
# -------------
# Ortalama (mean) aykırı değerlere duyarlıdır. Özellikle Insulin gibi
# sağa çarpık dağılımlarda medyan, merkezi eğilimi daha iyi temsil eder.
# Ayrıca medyan, diğer 0 olmayan gözlemlerden hesaplandığı için
# 0 değerlerinin kendisi hesaba katılmaz (bias önlemi).
# FINAL_REPORT.md §2.1 Tablo'suna bakınız.
df_clean = df.copy()
median_vals = {}

for col in ZERO_COLS:
    # Yalnızca 0 OLMAYAN değerlerin medyanını al.
    # Eğer tüm değerler dahil edilseydi, sıfırlar medyanı aşağı çekerdi.
    median_val = df_clean[col][df_clean[col] != 0].median()
    median_vals[col] = median_val
    filled_count = (df_clean[col] == 0).sum()
    df_clean[col] = df_clean[col].replace(0, median_val)
    print(f"  {col:<25} → medyan = {median_val:.2f}  "
          f"({filled_count} değer dolduruldu)")

print("\n  Doldurma sonrası kalan 0 sayıları (ZERO_COLS):")
print((df_clean[ZERO_COLS] == 0).sum().to_string())

# ═══════════════════════════════════════════════════════════
#  ADIM 2B – ÖNCE / SONRA KARŞILAŞTIRMA GRAFİĞİ
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2B — ÖNCE / SONRA KARŞILAŞTIRMA GRAFİĞİ")

# Bu grafik FINAL_REPORT.md §2.1'deki uyarıyı görselleştirir:
# Insulin ve SkinThickness'ta çok yüksek oranla medyan uygulandığından
# dağılımlarda yapay bir yoğunlaşma (spike) oluşur; bu durum söz konusu
# değişkenlerin sonraki analizlerdeki güvenilirliğini azaltır.
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Medyan Doldurma Öncesi / Sonrası Karşılaştırma",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

for i, col in enumerate(ZERO_COLS):
    color = PALETTE_COLS[i % len(PALETTE_COLS)]
    # Önce
    axes[0, i].hist(df[col], bins=30, color=color,
                    edgecolor=DARK_BG, linewidth=0.8, alpha=0.8)
    axes[0, i].axvline(0, color=ACCENT3, linewidth=2, label="0 değerleri")
    axes[0, i].set_title(f"{col}\n(Ham)", fontsize=10, pad=6)
    axes[0, i].grid(axis="y")
    axes[0, i].legend(fontsize=7)

    # Sonra: medyan noktasındaki dikey çizgi, doldurma değerini gösterir.
    axes[1, i].hist(df_clean[col], bins=30, color=color,
                    edgecolor=DARK_BG, linewidth=0.8, alpha=0.8)
    axes[1, i].axvline(median_vals[col], color=ACCENT2, linewidth=2,
                        linestyle="--", label=f"Medyan={median_vals[col]:.1f}")
    axes[1, i].set_title(f"{col}\n(Temiz)", fontsize=10, pad=6)
    axes[1, i].grid(axis="y")
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
save_fig("03_before_after_comparison.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 2C – KORELASYON ISI HARİTASI (Temiz Veri)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 2C — KORELASYON ISI HARİTASI")

# Pearson korelasyon matrisi: FINAL_REPORT.md §2.4'te yorumlanan
# korelasyon değerleri buradan elde edilmiştir.
# Öne çıkan bulgu: Glucose–Outcome korelasyonu (r≈+0.493) en yüksektir
# ve bu tüm fazlarda tutarlı şekilde birinci sırada çıkar.
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(DARK_BG)

corr = df_clean.corr()
# Alt üçgen maskesi: matrisin üst kısmı kaldırılır, tekrar eden değerler gizlenir.
mask = np.triu(np.ones_like(corr, dtype=bool))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            linecolor=DARK_BG, ax=ax,
            annot_kws={"size": 9, "color": TEXT_COLOR},
            cbar_kws={"shrink": 0.8})

ax.set_title("Korelasyon Matrisi — Temiz Veri", fontsize=14,
             fontweight="bold", color=TEXT_COLOR, pad=16)
ax.tick_params(colors=TEXT_COLOR)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)

save_fig("04_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 3 – VERİYİ BÖLME  (%70 Train / %15 Val / %15 Test)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 3 — VERİYİ BÖLME (%70 / %15 / %15)")

# Neden 3'e Bölme?
# ----------------
# Sadece train-test bölmesi "tek seferlik şans" riskine maruz kalır.
# Validation seti, hiperparametre seçimi (örn. KNN'de K değeri, Faz 4-5)
# için kullanılır; test seti yalnızca final değerlendirmede açılır.
# Bu tasarım Faz 5'te bias-variance analizinde ve Faz 6'da CV
# kıyaslamasında kritik bir rol oynar. (FINAL_REPORT.md §2.2)

X = df_clean[FEATURE_COLS]
y = df_clean[TARGET_COL]

# 1. Adım: %85 Train+Val  |  %15 Test
# ``stratify=y`` sınıf oranını (%34.9 pozitif) her sette korur;
# bu sayede küçük test setinde şans eseri dengesizlik önlenir.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2. Adım: Train+Val → %70 Train  |  %15 Val
#   (0.15 / 0.85 ≈ 0.1765 → %15 orijinalin %17.65'i)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_ratio,
    random_state=42, stratify=y_trainval
)

print(f"\n  {'Set':<12} {'Satır':>6}  {'%':>5}  "
      f"{'Target=0':>9}  {'Target=1':>9}  {'Denge':>8}")
print("  " + "-" * 60)
for name, X_s, y_s in [("Train", X_train, y_train),
                         ("Val",   X_val,   y_val),
                         ("Test",  X_test,  y_test)]:
    n     = len(y_s)
    pct   = n / len(y) * 100
    n0    = (y_s == 0).sum()
    n1    = (y_s == 1).sum()
    ratio = n1 / n * 100
    print(f"  {name:<12} {n:>6}  {pct:>4.1f}%  "
          f"{n0:>9}  {n1:>9}  {ratio:>6.1f}% pos")

# ═══════════════════════════════════════════════════════════
#  ADIM 3B – SET BOYUTLARI GRAFİĞİ
# ═══════════════════════════════════════════════════════════
section_header("ADIM 3B — SET BOYUTLARI GRAFİĞİ")

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)
gs  = gridspec.GridSpec(1, 2, figure=fig)

# — Pasta grafik (boyutlar)
ax_pie = fig.add_subplot(gs[0])
ax_pie.set_facecolor(CARD_BG)
sizes  = [len(X_train), len(X_val), len(X_test)]
labels = [f"Train\n{len(X_train)} ({len(X_train)/len(X)*100:.0f}%)",
          f"Val\n{len(X_val)} ({len(X_val)/len(X)*100:.0f}%)",
          f"Test\n{len(X_test)} ({len(X_test)/len(X)*100:.0f}%)"]
wedge_props = {"linewidth": 2, "edgecolor": DARK_BG}
ax_pie.pie(sizes, labels=labels, colors=[ACCENT1, ACCENT2, ACCENT3],
           wedgeprops=wedge_props, autopct="%1.1f%%",
           pctdistance=0.7, labeldistance=1.12,
           textprops={"color": TEXT_COLOR, "fontsize": 10})
ax_pie.set_title("Veri Seti Bölünmesi", fontsize=12,
                 fontweight="bold", color=TEXT_COLOR, pad=14)

# — Gruplu bar grafik (sınıf dengesi)
# stratify=y sayesinde her sette pozitif oran ~%35'te tutulmuştur.
ax_bar = fig.add_subplot(gs[1])
ax_bar.set_facecolor(CARD_BG)
set_names  = ["Train", "Val", "Test"]
neg_counts = [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()]
pos_counts = [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()]
x = np.arange(len(set_names))
w = 0.35
b1 = ax_bar.bar(x - w/2, neg_counts, w, label="Outcome = 0 (Negatif)",
                color=ACCENT1, edgecolor=DARK_BG, linewidth=1.2)
b2 = ax_bar.bar(x + w/2, pos_counts, w, label="Outcome = 1 (Pozitif)",
                color=ACCENT3, edgecolor=DARK_BG, linewidth=1.2)
for bar in list(b1) + list(b2):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9,
                color=TEXT_COLOR, fontweight="bold")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(set_names, fontsize=11)
ax_bar.set_ylabel("Örnek Sayısı", fontsize=10)
ax_bar.set_title("Her Sette Sınıf Dağılımı", fontsize=12,
                 fontweight="bold", color=TEXT_COLOR, pad=14)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis="y")

plt.tight_layout()
save_fig("05_split_distribution.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 4 – STANDARDİZASYON (StandardScaler)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4 — STANDARDİZASYON (StandardScaler)")

# Neden Sadece Train'e Fit?
# -------------------------
# StandardScaler z-skor dönüşümü uygular: z = (x - μ_train) / σ_train
#
# Scaler'ı train+val+test'e fit etmek "veri sızıntısı" (data leakage)
# yaratır: model, test/val verilerinin istatistiklerini önceden "görmüş"
# olur. Bu, genelleme performansını gerçekçi olmayan biçimde
# şişirir. (FINAL_REPORT.md §2.3 — "Neden train'e fit?")
#
# Faz 3'teki kritik gözlem (FINAL_REPORT.md §4.2): α=0.1 gibi yüksek
# öğrenme oranı bile ıraksamadı — çünkü Faz 1'de ölçeklendirme
# yapıldı. Ölçeklenmemiş veride bu LR tipik olarak patlama yapardı.
scaler = StandardScaler()

# YALNIZCA Train üzerinde fit! → μ ve σ train'den öğrenilir.
X_train_sc = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=FEATURE_COLS, index=X_train.index
)
# Val ve Test: sadece transform (fit değil) → veri sızıntısı yok.
X_val_sc = pd.DataFrame(
    scaler.transform(X_val),
    columns=FEATURE_COLS, index=X_val.index
)
X_test_sc = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURE_COLS, index=X_test.index
)

print("\n  Scaler parametreleri (Train üzerinden öğrenildi):")
print(f"  {'Özellik':<28} {'Ortalama':>10}  {'Std Dev':>10}")
print("  " + "-" * 52)
for col, mu, sig in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    print(f"  {col:<28} {mu:>10.4f}  {sig:>10.4f}")

# Başarılı dönüşüm sonrası Train ortalaması ≈0, std ≈1 olmalıdır.
print("\n  Ölçeklenmiş Train → İstatistikler (yaklaşık μ≈0, σ≈1 beklenir):")
print(X_train_sc.describe().loc[["mean", "std"]].round(4).to_string())

# ═══════════════════════════════════════════════════════════
#  ADIM 4B – ÖLÇEKLENDİRME ÖNCESİ / SONRASI VİOLİN GRAFİĞİ
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4B — ÖLÇEKLENDİRME VİOLİN GRAFİĞİ")

# Violin grafiği, ham verideki ölçek farklılıklarını (örn. Glucose ~120,
# Pregnancies ~3) netçe gösterir. Ölçekleme sonrası tüm özellikler
# aynı z-skor ekseni üzerinde karşılaştırılabilir hale gelir.
# KNN gibi mesafe tabanlı algoritmalar için bu kritiktir:
# büyük ölçekli özellik küçük ölçeklileri ezip "baskın" hale gelir.
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("StandardScaler — Ölçeklendirme Öncesi / Sonrası (Train Seti)",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

# Önce
data_before = [X_train[c].values for c in FEATURE_COLS]
vp1 = axes[0].violinplot(data_before, showmedians=True,
                          showextrema=True)
for pc, col in zip(vp1["bodies"], PALETTE_COLS):
    pc.set_facecolor(col)
    pc.set_alpha(0.75)
vp1["cmedians"].set_color(ACCENT5)
vp1["cmedians"].set_linewidth(2)
vp1["cmins"].set_color(TEXT_COLOR)
vp1["cmaxes"].set_color(TEXT_COLOR)
vp1["cbars"].set_color(TEXT_COLOR)
axes[0].set_xticks(range(1, len(FEATURE_COLS) + 1))
axes[0].set_xticklabels(FEATURE_COLS, rotation=30, ha="right", fontsize=9)
axes[0].set_title("Ölçeklendirme Öncesi", fontsize=12, pad=10)
axes[0].set_ylabel("Değer", fontsize=10)
axes[0].grid(axis="y")

# Sonra: tüm özellikler μ≈0 çevresinde, σ≈1 genişliğinde toplanır.
data_after = [X_train_sc[c].values for c in FEATURE_COLS]
vp2 = axes[1].violinplot(data_after, showmedians=True,
                          showextrema=True)
for pc, col in zip(vp2["bodies"], PALETTE_COLS):
    pc.set_facecolor(col)
    pc.set_alpha(0.75)
vp2["cmedians"].set_color(ACCENT5)
vp2["cmedians"].set_linewidth(2)
vp2["cmins"].set_color(TEXT_COLOR)
vp2["cmaxes"].set_color(TEXT_COLOR)
vp2["cbars"].set_color(TEXT_COLOR)
axes[1].set_xticks(range(1, len(FEATURE_COLS) + 1))
axes[1].set_xticklabels(FEATURE_COLS, rotation=30, ha="right", fontsize=9)
axes[1].set_title("Ölçeklendirme Sonrası (z-score)", fontsize=12, pad=10)
axes[1].set_ylabel("Z-Skoru", fontsize=10)
# μ=0 referans çizgisi: tüm özelliklerin ortalamadan sapmasını görselleştirir.
axes[1].axhline(0, color=ACCENT3, linewidth=1, linestyle="--", alpha=0.6)
axes[1].grid(axis="y")

plt.tight_layout()
save_fig("06_scaling_violin.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 4C – BOX-PLOT: ÖLÇEKLENMİŞ TRAIN/VAL/TEST KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4C — ÖLÇEKLENMİŞ SET KARŞILAŞTIRMASI (BOX-PLOT)")

# Train/Val/Test box-plotları: üç setin dağılımının benzer olması,
# stratifiye bölmenin başarılı olduğunun kanıtıdır.
# Büyük farklılıklar veri sızıntısına veya temsiliyet sorununa işaret eder.
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Ölçeklenmiş Değerler — Train / Val / Test Karşılaştırması",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)

set_labels  = ["Train", "Val", "Test"]
set_colors  = [ACCENT1, ACCENT2, ACCENT3]
set_dfs     = [X_train_sc, X_val_sc, X_test_sc]

for i, (col, ax) in enumerate(zip(FEATURE_COLS, axes.flatten())):
    data  = [s[col].values for s in set_dfs]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": ACCENT5, "linewidth": 2},
                    whiskerprops={"color": TEXT_COLOR},
                    capprops={"color": TEXT_COLOR},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": ACCENT4,
                                "alpha": 0.6})
    for patch, color in zip(bp["boxes"], set_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(set_labels, fontsize=9)
    ax.set_title(col, fontsize=10, pad=6)
    ax.set_ylabel("Z-Skoru", fontsize=8)
    # μ=0 referans: her özelliğin train ortalamasının sıfır olduğunu doğrular.
    ax.axhline(0, color=ACCENT3, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y")

plt.tight_layout()
save_fig("07_scaled_boxplots.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 4D – PAIR PLOT (Ölçeklenmiş Train Seti)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 4D — PAIR PLOT (Ölçeklenmiş Train Seti)")

# Pair plot, Faz 4'te anlamlı bulunan değişkenlerin (Glucose, BMI,
# Age, DiabetesPedigreeFunction) ikili ilişkilerini ve sınıflar
# arası ayrışmayı görselleştirir.
# → FINAL_REPORT.md §5.1 ve §9.1'de yorumlanan katsayılar bu
#   görsel ön keşifle tutarlıdır.
train_plot = X_train_sc.copy()
train_plot["Outcome"] = y_train.values

# Yalnızca 4 en önemli özelliği seç (okunabilirlik).
# Bu seçim korelasyon analizine (FINAL_REPORT.md §2.4) dayanmaktadır:
# Glucose (r=0.493), BMI (r=0.312), Age (r=0.238), DPF (r=0.173).
TOP4 = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]

g = sns.pairplot(
    train_plot, vars=TOP4, hue="Outcome",
    palette={0: ACCENT1, 1: ACCENT3},
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 20, "edgecolor": "none"},
    diag_kws={"fill": True, "alpha": 0.5}
)

g.figure.suptitle("Pair Plot — Top-4 Özellik (Ölçeklenmiş Train Seti)",
                   fontsize=13, fontweight="bold",
                   color=TEXT_COLOR, y=1.02)
g.figure.set_facecolor(DARK_BG)
for ax in g.axes.flatten():
    ax.set_facecolor(CARD_BG)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

save_fig("08_pair_plot.png")

# ═══════════════════════════════════════════════════════════
#  ADIM 5 – TEMİZ VERİLERİ KAYDET
# ═══════════════════════════════════════════════════════════
section_header("ADIM 5 — TEMİZ VERİLERİ CSV OLARAK KAYDET")

# İki farklı format kaydedilir:
# 1. Ham (raw): Medyan doldurulmuş ama ölçeklenmemiş → Faz 2'nin girdisi
#    (Faz 2 kendi ölçeklendirmesini yapar çünkü hedef değişken BMI'dır).
# 2. Ölçeklenmiş (scaled): z-skor dönüşümlü → Faz 3, 4, 5, 6, 7'nin girdisi
#    (sınıflandırma ve gradient descent analizleri için).

# Ham (doldurulmuş ama ölçeklenmemiş) setler
train_raw = X_train.copy(); train_raw[TARGET_COL] = y_train
val_raw   = X_val.copy();   val_raw[TARGET_COL]   = y_val
test_raw  = X_test.copy();  test_raw[TARGET_COL]  = y_test

train_raw.to_csv(os.path.join(OUTPUT_DIR, "train_raw.csv"), index=False)
val_raw.to_csv(os.path.join(OUTPUT_DIR, "val_raw.csv"),     index=False)
test_raw.to_csv(os.path.join(OUTPUT_DIR, "test_raw.csv"),   index=False)

# Ölçeklenmiş setler: index=False çünkü indeks sütunu sonraki
# fazlarda FEATURE_COLS listesiyle çakışmaya neden olabilir.
train_sc = X_train_sc.copy(); train_sc[TARGET_COL] = y_train.values
val_sc   = X_val_sc.copy();   val_sc[TARGET_COL]   = y_val.values
test_sc  = X_test_sc.copy();  test_sc[TARGET_COL]  = y_test.values

train_sc.to_csv(os.path.join(OUTPUT_DIR, "train_scaled.csv"), index=False)
val_sc.to_csv(os.path.join(OUTPUT_DIR, "val_scaled.csv"),     index=False)
test_sc.to_csv(os.path.join(OUTPUT_DIR, "test_scaled.csv"),   index=False)

print(f"\n  {'Dosya':<30} {'Satır':>6}  {'Sütun':>6}")
print("  " + "-" * 46)
for fname in ["train_raw.csv", "val_raw.csv", "test_raw.csv",
              "train_scaled.csv", "val_scaled.csv", "test_scaled.csv"]:
    tmp = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname:<30} {tmp.shape[0]:>6}  {tmp.shape[1]:>6}")

# ═══════════════════════════════════════════════════════════
#  ADIM 6 – ÖZET GRAFİĞİ (Dashboard)
# ═══════════════════════════════════════════════════════════
section_header("ADIM 6 — ÖZET DASHBOARD")

# Bu dashboard tek bir görselde Faz 1'in tüm bulgularını özetler:
# 0 değerleri, sınıf dengesi, set boyutları, medyanlar ve
# ölçeklenmiş özellik dağılımları (Outcome'a göre ayrılmış).
fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("FAZ 1 — VERİ ÖN İŞLEME ÖZET DASHBOARD",
             fontsize=17, fontweight="bold", color=TEXT_COLOR, y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)

# — Panel 1: 0 sayıları
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD_BG)
bars = ax1.barh(zero_counts.index, zero_counts.values,
               color=PALETTE_COLS[:len(ZERO_COLS)],
               edgecolor=DARK_BG, linewidth=1)
for bar, val in zip(bars, zero_counts.values):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", fontsize=8, color=TEXT_COLOR)
ax1.set_title("Doldurulan\n0 Değerleri", fontsize=10, fontweight="bold")
ax1.grid(axis="x")

# — Panel 2: Sınıf dengesi (tam veri)
# %34.9 pozitif oran — hafif dengesizlik, sonraki fazlarda
# Recall ve F1 üzerinde baskı oluşturur. (FINAL_REPORT.md §9.4)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_BG)
cls_counts = df_clean[TARGET_COL].value_counts().sort_index()
ax2.bar(["Negatif (0)", "Pozitif (1)"], cls_counts.values,
        color=[ACCENT1, ACCENT3], edgecolor=DARK_BG, linewidth=1.2)
for i, v in enumerate(cls_counts.values):
    ax2.text(i, v + 2, str(v), ha="center", fontsize=9,
             fontweight="bold", color=TEXT_COLOR)
ax2.set_title("Sınıf Dağılımı\n(Tüm Veri)", fontsize=10, fontweight="bold")
ax2.grid(axis="y")

# — Panel 3: Split boyutları
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(CARD_BG)
split_sizes = [len(X_train), len(X_val), len(X_test)]
labels_s    = [f"Train\n{len(X_train)}", f"Val\n{len(X_val)}", f"Test\n{len(X_test)}"]
ax3.bar(labels_s, split_sizes, color=[ACCENT1, ACCENT2, ACCENT3],
        edgecolor=DARK_BG, linewidth=1.2)
ax3.set_title("Set Boyutları\n(70/15/15)", fontsize=10, fontweight="bold")
ax3.grid(axis="y")

# — Panel 4: Medyan değerleri
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_facecolor(CARD_BG)
ax4.barh(list(median_vals.keys()), list(median_vals.values()),
         color=PALETTE_COLS[:len(ZERO_COLS)],
         edgecolor=DARK_BG, linewidth=1)
ax4.set_title("Doldurmada\nKullanılan Medyanlar", fontsize=10, fontweight="bold")
ax4.grid(axis="x")

# — Panel 5-8: Ölçeklenmiş özellik dağılımları (Outcome bazlı)
# Glucose ve BMI'nın iki sınıf arasındaki ayrışması en belirgindir;
# bu, FINAL_REPORT.md §5.1'deki katsayı analizinin görsel karşılığıdır.
for i, col in enumerate(["Glucose", "BMI", "Age", "Insulin"]):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor(CARD_BG)
    for outcome, color, label in [(0, ACCENT1, "Negatif"), (1, ACCENT3, "Pozitif")]:
        subset = X_train_sc[y_train == outcome][col]
        ax.hist(subset, bins=20, color=color, alpha=0.65,
                edgecolor=DARK_BG, linewidth=0.5, label=label)
    ax.set_title(f"{col}\n(Ölçeklenmiş Train)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y")
    ax.set_xlabel("Z-Skoru", fontsize=8)

save_fig("09_summary_dashboard.png")

# ═══════════════════════════════════════════════════════════
#  SONUÇ
# ═══════════════════════════════════════════════════════════
section_header("FAZ 1 — TAMAMLANDI ✅")

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │                    FAZ 1 ÖZET                           │
  ├──────────────────────────────────────────────────────────┤
  │  Toplam Örnek         : {len(df):>5}                          │
  │  Doldurulan Sütunlar  : {", ".join(ZERO_COLS)}  │
  │  Doldurmada Yöntem    : Medyan (0≠ değerlerin)           │
  ├──────────────────────────────────────────────────────────┤
  │  Train Seti           : {len(X_train):>5} örnek  (%70)            │
  │  Validation Seti      : {len(X_val):>5} örnek  (%15)            │
  │  Test Seti            : {len(X_test):>5} örnek  (%15)            │
  ├──────────────────────────────────────────────────────────┤
  │  Ölçeklendirme        : StandardScaler (μ=0, σ=1)        │
  │  Scaler fit yeri      : Yalnızca Train seti              │
  │  Veri Sızıntısı       : YOK ✅                           │
  ├──────────────────────────────────────────────────────────┤
  │  Kaydedilen Dosyalar  : {OUTPUT_DIR}/                    │
  │    • train_raw.csv    • val_raw.csv   • test_raw.csv     │
  │    • train_scaled.csv • val_scaled.csv• test_scaled.csv  │
  │  Grafikler (9 adet)   : {OUTPUT_DIR}/*.png               │
  └──────────────────────────────────────────────────────────┘
""")
