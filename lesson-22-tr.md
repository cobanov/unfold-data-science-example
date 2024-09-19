[Ana Sayfa](../bootcamp-main.md)

> "Her zorluğu, çözmek için gerekli olduğu kadar küçük parçalara böl."  
> — René Descartes

# Gün 22: Kümeleme Algoritmaları - K-Means, Hiyerarşik Kümeleme

Kümeleme, benzer veri noktalarını kümeler halinde gruplamak için kullanılan gözetimsiz bir öğrenme tekniğidir. Veri keşfi, müşteri segmentasyonu, görüntü analizi ve daha birçok alanda yaygın olarak kullanılmaktadır. Bugün, iki popüler kümeleme algoritmasını keşfedeceğiz: K-Means ve Hiyerarşik Kümeleme. Bu algoritmaların kavramlarını, Python'un scikit-learn kütüphanesi kullanılarak nasıl uygulanacağını ve kümeleme sonuçlarının nasıl değerlendirileceğini öğreneceğiz.

- Kümelemeye giriş
- K-Means kümeleme: teori, uygulama ve değerlendirme
- Hiyerarşik kümeleme: teori, uygulama ve değerlendirme
- Kümeleme sonuçlarının görselleştirilmesi

## Kümeleme

Kümeleme, gözetimsiz öğrenmenin temel bir görevidir. Burada amaç, bir veri setini öyle gruplara (_kümeler_) ayırmaktır ki aynı küme içindeki noktalar birbirine, diğer kümelerdeki noktalardan daha benzer olsun. Bu teknik müşteri segmentasyonu, anomali tespiti, görüntü işleme gibi çeşitli alanlarda kullanılır. Gözetimli öğrenmenin aksine, kümeleme algoritmaları etiketli verilere dayanmaz. Bunun yerine, verinin kendisindeki desenleri ve ilişkileri keşfederek çalışırlar.

Resmi olarak, kümeleme $` X = \{x_1, x_2, \dots, x_n\} `$ veri setinin $` K `$ kümesine $` C = \{C_1, C_2, \dots, C_K\} `$ bölünmesi olarak tanımlanır. Aynı küme içindeki noktalar, belirli bir benzerlik veya uzaklık ölçütüne (genellikle Öklid mesafesi) göre benzer olmalıdır. Farklı kümeleme algoritmaları "benzerliği" farklı şekillerde tanımlar ve algoritma seçimi verinin yapısına ve dağılımına bağlıdır.

## K-Means Kümeleme

**K-Means Kümeleme**, basitliği ve verimliliği nedeniyle en yaygın kullanılan kümeleme algoritmalarından biridir. K-Means'in temel fikri, veriyi $` K `$ kümeye ayırmak ve her veri noktasını en yakın ortalamaya sahip kümeye atamaktır. Bu, bir merkezi tabanlı kümeleme yöntemidir ve süreç, kümelerin merkez noktalarını (merkezleri) yinelemeli olarak iyileştirerek küme içi varyansı minimize etmeyi amaçlar.

Verilen $` X \in \mathbb{R}^n `$ veri seti için algoritma $` K `$ başlangıç merkezi (rastgele veya K-Means++ gibi stratejilerle) seçerek başlar. Algoritma, iki ana adım arasında dönüşümlü olarak çalışır:

1. **Atama Adımı**: Her veri noktası en yakın merkeze atanır. Her bir $` x_i `$ noktası için en yakın merkez $` \mu_j `$ Öklid mesafesi kullanılarak bulunur:

```math
\text{arg min}\_j \| x_i - \mu_j \|^2
```

Bu adım, veriyi merkezlerine en yakın olan kümelere böler.

2. **Güncelleme Adımı**: Tüm noktalar atandıktan sonra, her kümenin merkezi, o kümedeki noktaların ortalaması alınarak güncellenir:

```math
\mu*j = \frac{1}{|C_j|} \sum*{x_i \in C_j} x_i
```

Algoritma, merkezler değişmeyene (veya değişiklik belirli bir eşik değerinin altına düşene) kadar veya maksimum yineleme sayısına ulaşana kadar bu iki adımı tekrarlar. K-Means algoritmasının amacı, toplam küme içi varyansı minimize etmektir:

```math
J = \sum*{j=1}^{K} \sum*{x_i \in C_j} \| x_i - \mu_j \|^2
```

K-Means algoritması yerel bir minimuma ulaşmayı garantiler, ancak performansı başlangıçta seçilen merkezlere bağlıdır. Bu nedenle, farklı başlatmalarla birden fazla çalıştırma veya K-Means++ başlatma stratejisi önerilir.

### K-Means Uygulaması

Python'da, K-Means `scikit-learn` kütüphanesindeki `KMeans` sınıfı ile uygulanır. Basit bir uygulama şu şekilde görünür:

```python
from sklearn.cluster import KMeans

# Veri seti X olarak varsayalım
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
etiketler = kmeans.labels_  # Her nokta için küme etiketleri
merkezler = kmeans.cluster_centers_  # Merkezlerin konumları
```

### K-Means Değerlendirme

K-Means kümeleme kalitesini değerlendirmenin birkaç yolu vardır:

1. **Küme İçi Kareler Toplamı (WCSS)**: Bu, yukarıda bahsedilen maliyet fonksiyonu $` J `$'dir. Daha düşük WCSS değerleri daha iyi kümelemeyi gösterir.
2. **Silhouette Skoru**: Bu, noktaların kendi kümelerine ne kadar benzer olduğunu diğer kümelere kıyasla ölçer. Skor -1 ile 1 arasında değişir ve 1'e yakın değerler daha iyi tanımlanmış kümeler anlamına gelir.

```math
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
```

Bu formülde $` a(i) `$ aynı kümede bulunan diğer noktalara olan ortalama mesafeyi, $` b(i) `$ ise farklı kümelerdeki en yakın noktalara olan ortalama mesafeyi ifade eder.

3. **Dirsek Yöntemi**: Bu yöntem, optimal küme sayısını $` K `$ belirlemede kullanılır. WCSS'yi farklı $` K `$ değerlerine karşı grafiğe döker ve "dirsek noktası" olarak bilinen yerde durulur. Bu noktadan sonra kümeleri artırmak, WCSS'yi önemli ölçüde azaltmaz ve bu, en uygun küme sayısını belirler.

## Hiyerarşik Kümeleme

**Hiyerarşik Kümeleme**, bir diğer popüler kümeleme yöntemidir. K-Means'in aksine, önceden bir küme sayısı belirtmeyi gerektirmez. Bunun yerine, kümeler arasında bir hiyerarşi oluşturur ve bu hiyerarşi bir **dendrogram** kullanılarak görselleştirilebilir. Hiyerarşik kümeleme iki ana formda gelir: **aşağıdan yukarıya (agglomeratif)** ve **yukarıdan aşağıya (divisive)**.

### Agglomeratif Hiyerarşik Kümeleme

Agglomeratif kümeleme, her veri noktasını kendi başına bir küme olarak başlatır ve ardından en yakın kümeleri birleştirerek tek bir küme kalana kadar devam eder. Kümeler arasındaki mesafe çeşitli şekillerde tanımlanabilir:

1. **Tek Bağlantı (Single Linkage)**: İki küme arasındaki mesafe, kümelerdeki herhangi iki nokta arasındaki en kısa mesafe ile tanımlanır:

```math
D(C*i, C_j) = \min*{x \in C_i, y \in C_j} \|x - y\|
```

2. **Tam Bağlantı (Complete Linkage)**: İki küme arasındaki mesafe, kümelerdeki herhangi iki nokta arasındaki en uzun mesafe ile tanımlanır:

```math
D(C*i, C_j) = \max*{x \in C_i, y \in C_j} \|x - y\|
```

3. **Ortalama Bağlantı (Average Linkage)**: İki küme arasındaki mesafe, kümelerdeki tüm noktalar arasındaki ortalama mesafe ile tanımlanır:

```math
D(C*i, C_j) = \frac{1}{|C_i| |C_j|} \sum*{x \in C*i} \sum*{y \in C_j} \|x - y\|
```

4. **Ward Yöntemi**: Bu yöntem, kümeler arası varyansı en aza indirmeyi amaçlar. Varyansı en az artıracak kümeler birleştirilir.

Agglomeratif kümeleme sonucunda bir ağaç yapısı (dendrogram) elde edilir. Her birleştirme, yatay bir çizgi ile temsil edilir. Çizginin yüksekliği, birleştirilen kümeler arasındaki mesafeyi gösterir. Dendrogram'ı belirli bir seviyeden keserek, istenilen küme sayısına ulaşılabilir.

### Hiyerarşik Kümeleme Uygulaması

Python'da hiyerarşik kümeleme, `scikit-learn` kütüphanesindeki `AgglomerativeClustering` sınıfı ve `scipy` kütüphanesindeki `dendrogram` fonksiyonu kullanılarak uygulanabilir:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Veri seti X olarak varsayalım
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
etiketler = hierarchical.fit_predict(X)

# Dendrogram oluşturmak için
Z = linkage(X, method='ward')
dendrogram(Z)
plt.show()
```

### Hiyerarşik Kümeleme Değerlendirmesi

Hiyerarşik kümelemenin değerlendirilmesi, dendrogram'ın veri noktaları arasındaki mesafeleri ne kadar doğru temsil ettiğini ölçen **kofenetik korelasyon katsayısı** ile yapılabilir. Silhouette skorları ve dendrogram analizi de kümelerin kalitesini değerlendirmek için kullanılabilir.

Dendrogram, kümeleme yapısını görselleştirerek içgörüler sunabilir. Ağaç yapısını farklı seviyelerde keserek, küme sayısı kolayca belirlenebilir. Bu nedenle, hiyerarşik kümeleme, küme sayısının önceden bilinmediği durumlarda oldukça esnek bir yöntemdir.

K-Means ve Hiyerarşik Kümeleme, güçlü kümeleme yöntemleri olup, her ikisinin de kendine özgü avantajları vardır. K-Means, büyük veri setleriyle çalışırken hesaplama açısından verimlidir ve küme sayısı bilindiğinde iyi çalışır. Hiyerarşik Kümeleme ise daha esnektir ve verinin yapısına dair daha derin içgörüler sunabilir, ancak hesaplama maliyeti daha yüksektir. Hangi yöntemin kullanılacağı problem yapısına bağlıdır.

## Problem

**Problem Tanımı:** Müşterilerin yıllık gelirleri ve harcama puanlarını içeren bir veri setiniz var. Görevleriniz şunlardır:

1. Müşteri segmentasyonu için K-Means kümeleme algoritmasını uygulayın.
2. Müşteri segmentasyonu için Hiyerarşik kümeleme algoritmasını uygulayın.
3. Kümeleme sonuçlarını görselleştirin.

**Veri Seti:**

```csv
CustomerID, Annual Income (k$), Spending Score (1-100)
1, 15, 39
2, 16, 81
3, 17, 6
4, 18, 77
5, 19, 40
6, 20, 76
7, 21, 6
8, 22, 94
9, 23, 3
10, 24, 72
```

## Açıklama

1. **Veri Çerçevesi Oluşturma:** Müşteri bilgilerini içeren veri setinden bir DataFrame oluşturuyoruz.
2. **Özellik Seçimi:** Kümeleme için özellikleri (Yıllık Gelir ve Harcama Puanı) seçiyoruz.
3. **K-Means Kümeleme Uygulaması:** K-Means algoritmasını `k=3` ile uygulayıp müşterileri kümeliyoruz. Küme etiketlerini DataFrame'e ekleyip sonuçları bir dağılım grafiği ile görselleştiriyoruz.
4. **Hiyerarşik Kümeleme Uygulaması:** Hiyerarşik Kümeleme algoritmasını `n_clusters=3` ile uygulayıp müşterileri kümeliyoruz. Küme etiketlerini DataFrame'e ekleyip sonuçları bir dağılım grafiği ile görselleştiriyoruz.
5. **Dendrogram Oluşturma:** Kümeleme sürecini görselleştirmek ve küme birleşimlerini anlamak için bir dendrogram oluşturuyoruz.

## Çözüm

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

# Adım 1: Veri setinden bir DataFrame oluşturun
data = pd.read_csv("./dataset/customer_dataset.csv")
df = pd.DataFrame(data)

# Kümeleme için özellikleri seçin
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Adım 2: K-Means kümeleme uygulaması
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

# K-Means kümeleme sonuçlarını görselleştirin
plt.figure(figsize=(10, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
)
plt.title("K-Means Kümeleme")
plt.xlabel("Yıllık Gelir (k$)")
plt.ylabel("Harcama Puanı (1-100)")
plt.colorbar(label="Küme")
plt.show()

# Adım 3: Hiyerarşik kümeleme uygulaması
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X)

# Hiyerarşik kümeleme sonuçlarını görselleştirin
plt.figure(figsize=(10, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Hierarchical_Cluster"],
    cmap="viridis",
)
plt.title("Hiyerarşik Kümeleme")
plt.xlabel("Yıllık Gelir (k$)")
plt.ylabel("Harcama Puanı (1-100)")
plt.colorbar(label="Küme")
plt.show()

# Adım 4: Hiyerarşik kümeleme için dendrogram oluşturun
Z = linkage(X, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Müşteri ID")
plt.ylabel("Mesafe")
plt.show()
```
