


# **Association Rule Learning (ARL)**

Pattern, ilişki, yapı gibi veri içerisindeki örüntüleri bulmak için kullanılan kural tabanlı bir makine öğrenmesi tekniğidir.
X filmi beğenildiğinde Y filminin beğenilme olasılığı gibi.
Sepetlere odaklanarak birlikte geçme frekansları üzerinden tavsiye sistemi geliştirmesinde kullanılır.

# **ARMUT - Association Rule Based Recommender System**

Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association
Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

***VERİ SETİ HİKAYESİ***

Veri seti müşterilerin aldıkları servislerden ve bu servislerin Kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat
bilgisini içermektedir.


*  **UserId:** Müşteri numarası
*   **ServiceId:** Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi).
    Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
(Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)

*   **CategoryId:** Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)

*   **CreateDate:** Hizmetin satın alındığı tarih

# Görev 1: Veriyi Hazırlama

**Adım 1:** armut_data.csv dosyasını okutunuz
"""

import pandas as pd
pd.set_option('display.max_columns', None)

# !pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.head()

df.dtypes

"""Veri setinde tarih alanı kategorik değişken formunda, eğer tarih alanı ile ilgili bir işlem gerçekleştirilecekse öncelikle datetime formuna çevirilmeli.

**Adım 2:** ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
temsil edecek yeni bir değişken oluşturunuz. Elde edilmesi gereken çıktı:


df.dtypes

"""Bu 2 alan numeric değişken olduğu için öncelikle string'e çevirerek birleştirme işlemi uygulayabiliriz:"""

df.iloc[:, 1:3]

# apply dataframe ve serilere uygulanabilir, ancak dataframe'in satır ve sütunlarını adreslediğimiz için hangi kolon olduğunu ifade etmeliyiz:

df["New_Hizmet"] = df.iloc[:, 1:3].apply(lambda x: str(x[0]) + "_" + str(x[1]), axis=1)
df.head()

df.head().values

# Alternatif Yöntem: tüm satırlarda gezerek 1. ve 2.indexi birleştirmek

# df["New_Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

"""**Adım 3:** Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı
hizmetlerdir.

Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7
hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren
yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
Elde edilmesi gereken çıktı:

***Uygulama:*** ARL uygulaması için öncelikle invoice (sepet Id) X product matris yapısı kurulmalı:


df.head()

# Yıl ay bilgisini elde edebilmek için öncelikle tarih bilgisini datetime formuna çevirelim:
df["CreateDate"] = pd.to_datetime(df["CreateDate"])

# Yıl-Ay bilgisinin oluşturulması:

df["New_CreateDate_YM"] = df['CreateDate'].dt.year.astype(str) + "_" + df['CreateDate'].dt.month.astype(str)

# Alternatif Yöntemler:
# df["CreateDate"].dt.strftime("%Y-%m")
# df["New_CreateDate_YM"] = df['CreateDate'].dt.to_period('M')

df.head()

# Şimdi SepetId alanını oluşturalım:

df["New_SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

"""# GÖREV 2: Birliktelik Kuralları Üretiniz

Adımlar:
  - Invoice X product matrisinin oluşturulması ( Sepet Id vs Hizmet+Servis)
  - Apriori algoritması ile ürün birlikteliklerinin olasılıklarının hesaplanması
  - Ürünlerin her olası birlikteliklerinin support/lift değerlerine göre sıralama yapılması

**Adım 1:** Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.


# Yukarıdaki çıktıya göre yeni türettiğimiz "New_Hizmet" alanı kolonlarda "SepetId" alanı indexlerde yer almalı
# pivot table ile ya da groupby ile bu matris oluşturulabilir

df.head()

# kesişimde hangi feature üzerinden saydırma işlemi yapılacağı belirtilmediği durumda tüm kolonlar bir seviye de kolon olarak türetilir
# df.head(2).pivot_table(columns=["New_Hizmet"], index=["New_SepetID"], aggfunc="count")

df.pivot_table(columns=["New_Hizmet"], index=["New_SepetID"], values=["ServiceId"], aggfunc="count").head(3)

"""Kesişimde yer alan NaN Ddeğerleri 0 ile >0 olan değerleri 1 ile  dönüştürerek 1-0 lardan oluşan bir matris dizayn etmeliyiz:"""

invoice_product_df = df.pivot_table(columns=["New_Hizmet"], 
                                    index=["New_SepetID"], 
                                    values=["ServiceId"], 
                                    aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# Alternatif Yöntem:
# invoice_product_df = df.groupby(['New_SepetID', 'New_Hizmet'])['New_Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df.head()

# ServiceId başlığından kurtulalım (Multiindex silmek için droplevel() metodu kullanılabilir):
invoice_product_df.columns

invoice_product_df.columns = invoice_product_df.columns.droplevel(0)

invoice_product_df.head()

"""**Adım 2:** Birliktelik kurallarını oluşturunuz."""

# Tüm olası ürün birlikteliklerinin olasılıklarına bakalım, bizim belirlediğimiz threshold değeri 0.01 (%1 olsun)
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
frequent_itemsets.shape

# frequent_itemsets = apriori(invoice_product_df, min_support=0.5, use_colnames=True)
# frequent_itemsets

# frequent_itemsets = apriori(invoice_product_df, min_support=0.1, use_colnames=True)
# frequent_itemsets.shape

# min threshold -> min confidence değerini yani X ürünü alındığında Y nin alınma olasılığını göstermektedir:
# Şimdi min confidence'a göre rule'ları çıkaralım:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

"""**Adım3:** "arl_recommender" fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

**Adımlar:**

- İş kuralı ile lift ya da support değerine göre hangisi ile ilerlenecekse büyükten küçüğe sıralanması.
- İlgili hizmetin "antecendent" kolonunda aranması
- Antecendent'ta tespit edilen hizmetin index bilgisinin bulunması
- consequents değişkeninde ilgili index'e gidilerek ürün önerisi yapılması.
"""

#Yukarıda oluşturduğumux kuralları  sıralayım (birlikte alınma olasılığı lift'e göre daha yüksek olanlardan ilerleyelim)
sorted_rules = rules.sort_values("lift", ascending=False)

sorted_rules.head()

"""(15_1)	ürününü ele alalım:"""

# antecendent alanında ürünü ararken yani elemanına erişirken aynı zamanda index bilgisini bulacağız; enumarate metodunu kullanalım:

for idx, product in enumerate(sorted_rules["antecedents"].head()):
  print(idx, "_", product)
  print(list(product))

"""Örneğin 3.index'te tespit ettik:"""

sorted_rules.iloc[3]

sorted_rules.iloc[3]["consequents"]

# Bir listede değerleri tutacağımız için liste formuna çevirerek append yapabiliriz:
list(sorted_rules.iloc[3]["consequents"])

product_id = "15_1"
recommendation_list = []

for idx, product in enumerate(sorted_rules["antecedents"]):
    # antecendent tuple olduğu için listeye çevirelim ve liste içinde arayalım:
    for j in list(product):
        if j == product_id:
            # bu yakaladığımız değerin indexi ne ise (idx) consequentte onu arayacağız,
            # bulduğumuz satırlar için ilk ürünü [0]  önerelim
            recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])

print(recommendation_list)

# ['33_4', '2_0', '38_4']

product_id = "2_0"
recommendation_list = []

for idx, product in enumerate(sorted_rules["antecedents"]):
    # antecendent tuple olduğu için listeye çevirelim ve liste içinde arayalım:
    for j in list(product):
        if j == product_id:
            # bu yakaladığımız değerin indexi ne ise (idx) consequentte onu arayacağız,
            # bulduğumuz satırlar için ilk ürünü [0]  önerelim
            recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"]))

print(recommendation_list[0:3])
print(recommendation_list[0:4])

# Fonksiyonlaştıralım:

def arl_recommender(rules_df, product_id, rec_count=1):
    recommendation_list = []
    sorted_rules = rules.sort_values("lift", ascending=False)
    for idx, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
              recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 5)

# ['22_0', '25_0', '15_1', '13_11', '38_4']

arl_recommender(rules, "2_0", 3)

# ['22_0', '25_0', '15_1']

arl_recommender(rules, "38_4", 3)

# ['15_1', '2_0']

arl_recommender(rules, "38_4", 2)

#['15_1', '2_0']
