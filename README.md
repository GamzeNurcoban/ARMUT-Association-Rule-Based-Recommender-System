# ARMUT Association Rule Based Recommender System
<img width="505" alt="image" src="https://user-images.githubusercontent.com/101832704/175570700-7739931d-4b7c-407a-be7f-48e1a72920c0.png">
![pexels-karolina-grabowska-5717993](https://user-images.githubusercontent.com/101832704/175570961-39f2a6ca-d23d-476b-a0f5-397b1b167f00.jpg)

# İŞ PROBLEMİ 
Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association
Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


# VERİ SETİ HİKAYESİ

Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat
bilgisini içermektedir.

# DEĞİŞKENLER

UserId = Müşteri numarası

ServiceId = Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi) 
Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder. 
(Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)

CategoryId = Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)

CreateDate = Hizmetin satın alındığı tarih
