[Anasayfa](../bootcamp-main.md)

> “Geometri, sonsuzca var olanın bilgisidir.”
> — Pisagor

# Ders 5: Keşifsel Veri Analizi (EDA) - Tanımlayıcı İstatistikler ve Görselleştirme

Keşifsel Veri Analizi (EDA), veri bilimi sürecinde verinin temel özelliklerinin özetlendiği, genellikle görsel yöntemlerin kullanıldığı kritik bir adımdır. EDA, veriyi anlamaya, anomalileri tespit etmeye, desenleri bulmaya ve hipotezler oluşturmaya yardımcı olur. Bu bölümde, tanımlayıcı istatistikler ve temel görselleştirmeler yardımıyla bir veri kümesini keşfetmeye odaklanacağız.

- **Tanımlayıcı istatistikler:** ortalama, medyan, mod, varyans, standart sapma
- **Veri dağılımı:** histogramlar, kutu grafikleri
- **Değişkenler arasındaki ilişkiler:** saçılım grafikleri, korelasyon matrisleri

## Tanımlayıcı İstatistikler: Ortalama, Medyan, Mod, Varyans, Standart Sapma

**Tanımlayıcı istatistikler**, bir veri kümesinin temel özelliklerini özetler ve açıklar. Veri kümesinin dağılımı, merkezi eğilimi ve değişkenliği hakkında bilgi sağlar.

### Ortalama

**Ortalama** (aritmetik ortalama), veri kümesindeki tüm değerlerin toplamının, veri sayısına bölünmesiyle bulunur. Verinin merkezi eğilimini temsil eder.

```math
\text{Ortalama} = \frac{\sum_{i=1}^{n} x_i}{n}
```

Burada $`x_i`$, değerleri, $`n`$ ise gözlemlerin sayısını temsil eder.

```python
import numpy as np

veri = [10, 20, 30, 40, 50]
ortalama = np.mean(veri)
print(ortalama)  # Çıktı: 30.0
```

### Medyan

**Medyan**, veri sıralandığında ortada kalan değerdir. Verinin çarpık olduğu veya aykırı değerler içerdiği durumlarda, medyan merkezi eğilimi ölçmek için ortalamadan daha iyi bir göstergedir.

```python
medyan = np.median(veri)
print(medyan)  # Çıktı: 30.0
```

### Mod

**Mod**, veri kümesindeki en sık görülen değerdir. Kategorik verilerde, mod en yaygın kategoriyi temsil eder.

```python
from scipy import stats

mod = stats.mode(veri)
print(mod.mode[0])  # Çıktı: 10 (eğer 10 en sık geçiyorsa)
```

### Varyans

**Varyans**, veri kümesindeki değerlerin ortalamadan ne kadar uzaklaştığını ölçer. Ortalama ile olan farkların karesinin ortalamasıdır.

```math
\text{Varyans} = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}
```

Burada $`\mu`$, veri kümesinin ortalamasıdır.

```python
varyans = np.var(veri)
print(varyans)  # Çıktı: Verinin varyansı
```

### Standart Sapma

**Standart sapma**, varyansın kareköküdür. Verinin yayılımı hakkında bilgi verir ve verinin birimleriyle aynı birimde ifade edilir.

```math
\text{Standart Sapma} = \sqrt{\text{Varyans}}
```

```python
std_sapma = np.std(veri)
print(std_sapma)  # Çıktı: Verinin standart sapması
```

## Veri Dağılımı

### Histogramlar

**Histogram**, sayısal verilerin dağılımını gösteren bir grafiksel temsildir. Verilerin belirli aralıklara (binler) göre sıklığını gösterir ve dağılımın şeklini görselleştirmeye yardımcı olur (örneğin, çarpıklık, çok modluluk).

Python'da **Matplotlib** kullanarak histogram çizimine örnek:

```python
import matplotlib.pyplot as plt

# Örnek veri oluştur
veri = np.random.normal(0, 1, 1000)

# Histogramı çiz
plt.hist(veri, bins=30, edgecolor='black')
plt.title('Veri Histogramı')
plt.xlabel('Değer')
plt.ylabel('Sıklık')
plt.show()
```

![histogram](../bootcamp-subpages/plots/lesson-05-histogram.png)

**Şekil 1.** Normal dağılıma sahip veri değerlerinin histogramı.

### Kutu Grafikler

Bir **kutu grafik** (ya da **kutu ve bıyık grafiği**), verinin dağılımını medyan, çeyrekler ve aykırı değerleri göstererek görselleştirir. Çarpıklığı, yayılımı ve olası aykırı değerleri belirlemeye yardımcı olur.

Ana bileşenler:

- **Medyan**: Kutunun içindeki çizgi.
- **Çeyrekler arası aralık (IQR)**: 1. çeyrek (Q1) ile 3. çeyrek (Q3) arasındaki aralık.
- **Bıyıklar**: Aykırı olarak kabul edilmeyen veri aralığını temsil eder.
- **Aykırı değerler**: Bıyıkların dışında kalan veri noktaları.

Python'da **Matplotlib** kullanarak kutu grafik örneği:

```python
import matplotlib.pyplot as plt
import numpy as np

# Örnek veri oluştur
veri = np.random.normal(0, 1, 1000)

plt.boxplot(veri)
plt.title("Veri Kutu Grafiği")
plt.ylabel("Değer")
plt.show()
```

![boxplot](../bootcamp-subpages/plots/lesson-05-boxplot.png)

**Şekil 2.** Normal dağılıma sahip veri değerlerinin kutu grafiği.

## Değişkenler Arasındaki İlişkiler

### Saçılım Grafikleri

Bir **saçılım grafiği**, iki sayısal değişken arasındaki ilişkiyi 2D bir düzlemde veri noktalarını kullanarak görselleştirir. Her nokta bir gözlemi temsil eder; bir değişken x ekseninde, diğeri y ekseninde gösterilir.

Saçılım grafikleri şunları tespit etmek için faydalıdır:

- **Doğrusal ilişkiler** (pozitif veya negatif korelasyon),
- **Veri noktalarının kümeleri**,
- **Aykırı değerler**.

Python'da saçılım grafiği örneği:

```python
# Örnek veri oluştur
x = np.random.rand(100)
y = 2 * x + np.random.randn(100) * 0.1

# Saçılım grafiğini çiz
plt.scatter(x, y)
plt.title('X ve Y Saçılım Grafiği')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Korelasyon Matrisleri

Bir **korelasyon matrisi**, birden fazla değişken arasındaki ikili korelasyon katsayılarını gösterir. Korelasyon katsayısı -1 ile 1 arasında değişir:

- **+1** mükemmel pozitif korelasyonu gösterir,
- **-1** mükemmel negatif korelasyonu gösterir,
- **0** korelasyon olmadığını gösterir.

İki değişken arasındaki doğrusal ilişkiyi ölçmek için yaygın olarak **Pearson korelasyon katsayısı** kullanılır.

Python'da **Pandas** ve **Seaborn** kullanarak korelasyon matrisi hesaplama ve görselleştirme örneği:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Örnek bir DataFrame oluştur
veri = {"X1": np.random.rand(100), "X2": np.random.rand(100), "X3": np.random.rand(100)}
df = pd.DataFrame(veri)

# Korelasyon matrisini hesapla
corr_matrix = df.corr()

# Korelasyon matrisini ısı haritası olarak görselleştir
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()
```

![correlation-matrices](../bootcamp-subpages/plots/lesson-05-correlation-matrices.png)

**Şekil 3.** X1, X2 ve X3 değişkenlerinin korelasyon matrisi.

## Problem

**Problem Tanımı:** Size, çeşitli arabalar hakkında marka, model, yıl, motor hacmi, beygir gücü ve yakıt tüketimi bilgilerini içeren bir veri kümesi verilmiştir.

**Görevleriniz şunlardır:**

1. Sayısal sütunlar için tanımlayıcı istatistikleri hesaplayın.
2. Motor hacmi ve beygir gücü dağılımını histogramlarla görselleştirin.
3. Yakıt tüket

imi dağılımını gösteren bir kutu grafik oluşturun. 4. Motor hacmi ile beygir gücü arasındaki ilişkiyi incelemek için bir saçılım grafiği oluşturun. 5. Sayısal sütunlar için korelasyon matrisini hesaplayın ve görselleştirin.

**Veri Kümesi:**

```csv
Marka, Model, Yıl, MotorHacmi, BeygirGücü, YakıtTüketimi
Toyota, Corolla, 2010, 1.8, 132, 30
Honda, Civic, 2012, 2.0, 158, 32
Ford, Focus, 2015, 2.0, 160, 28
Chevrolet, Malibu, 2018, 1.5, 160, 29
Nissan, Sentra, 2013, 1.8, 130, 31
```

## Açıklama

1. **DataFrame Oluşturun:** Verilen veri kümesinden bir DataFrame oluşturuyoruz.
2. **Tanımlayıcı İstatistikler:** Sayısal sütunlar için `describe()` yöntemi ile tanımlayıcı istatistiklerin özetini alıyoruz.
3. **Histogramlar:** 'MotorHacmi' ve 'BeygirGücü' sütunları için histogramlar oluşturuyoruz.
4. **Kutu Grafik:** 'YakıtTüketimi' sütunu için bir kutu grafik oluşturup, dağılımı ve olası aykırı değerleri gösteriyoruz.
5. **Saçılım Grafiği:** 'MotorHacmi' ile 'BeygirGücü' arasındaki ilişkiyi incelemek için bir saçılım grafiği oluşturuyoruz.
6. **Korelasyon Matrisi:** Sayısal sütunlar için korelasyon matrisini hesaplayıp, bir ısı haritası kullanarak görselleştiriyoruz.

## Çözüm

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adım 1: Veri kümesinden bir DataFrame oluştur
veri = {
    "Marka": ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"],
    "Model": ["Corolla", "Civic", "Focus", "Malibu", "Sentra"],
    "Yıl": [2010, 2012, 2015, 2018, 2013],
    "MotorHacmi": [1.8, 2.0, 2.0, 1.5, 1.8],
    "BeygirGücü": [132, 158, 160, 160, 130],
    "YakıtTüketimi": [30, 32, 28, 29, 31]
}
df = pd.DataFrame(veri)

# Adım 2: Sayısal sütunlar için tanımlayıcı istatistikleri hesapla
tanımlayıcı_istatistikler = df.describe()

# Adım 3: Motor hacmi ve beygir gücü dağılımını histogramlarla görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["MotorHacmi"], bins=5, kde=True)
plt.title("Motor Hacmi Dağılımı")

plt.subplot(1, 2, 2)
sns.histplot(df["BeygirGücü"], bins=5, kde=True)
plt.title("Beygir Gücü Dağılımı")

plt.tight_layout()
plt.show()

# Adım 4: Yakıt tüketimi dağılımını gösteren bir kutu grafik oluştur
plt.figure(figsize=(6, 5))
sns.boxplot(y=df["YakıtTüketimi"])
plt.title("Yakıt Tüketimi Dağılımı")
plt.show()

# Adım 5: Motor hacmi ile beygir gücü arasındaki ilişkiyi incelemek için saçılım grafiği oluştur
plt.figure(figsize=(6, 5))
sns.scatterplot(x="MotorHacmi", y="BeygirGücü", data=df)
plt.title("Motor Hacmi ve Beygir Gücü İlişkisi")
plt.show()

# Adım 6: Sayısal sütunlar için korelasyon matrisini hesapla ve görselleştir
korelasyon_matrisi = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(korelasyon_matrisi, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()

# Sonuçlar
print("Tanımlayıcı İstatistikler:")
print(tanımlayıcı_istatistikler)
```

---

[Veri Temizleme ve Ön İşleme ←](../bootcamp-subpages/lesson-04.md) | [Anasayfa](../bootcamp-main.md) | [→ Numpy ve Pandas](../bootcamp-subpages/lesson-06.md)

---
