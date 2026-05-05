# Contextual Embedding ve BERT Analizi
### BM5204 Doğal Dil İşleme — Hafta 9 Ödevi
**Ahmetcan PEKTAŞ** | Bandırma Onyedi Eylül Üniversitesi

---

## Hakkında

Bu çalışmada, Türkçe çok anlamlı kelimeler üzerinde bağlamsal gömme (contextual embedding) davranışı `dbmdz/bert-base-turkish-cased` modeli ile analiz edilmiştir. Dört farklı görev üzerinden BERT mimarisinin statik modellere (Word2Vec, FastText) kıyasla sağladığı anlam ayrımı avantajı gözlemlenmiş; fine-tuning gerekliliği deneysel olarak ortaya konmuştur.

> **Önceki haftayla bağlantı:** Hafta 6'da aynı türdeki çok anlamlı kelimeler üzerinde Word2Vec ve FastText karşılaştırması yapılmıştı. Her iki modelin de polisemi problemini çözemediği gözlemlenmişti. Bu hafta BERT'in bu soruna getirdiği çözüm incelenmektedir.

---

## Temel Bulgular

- **Bağlamsal ayrışma:** Aynı kelimenin (banka, kol, yüz, dil) farklı anlamlardaki CLS embedding'leri arasındaki cosine similarity **0.8179–0.9497** aralığında ölçülmüştür. Statik modellerde bu değer zorunlu olarak **1.0** olurdu.
- **MLM bağlam ayrımı:** `banka` kelimesi finans bağlamında `parayı` (%23.1), nehir bağlamında `gidip` (%27.2) tahminlerine yol açmıştır — aynı kelime için tam anlamıyla iki farklı temsil.
- **Fine-tuning gerekliliği:** Ham BERT her iki duygu cümlesine `LABEL_0` (~0.66 skor) verirken, fine-tuned model aynı cümleleri **%98+ güven** skoru ile doğru sınıflandırmıştır.

---

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `hafta9_nlp_odevi_final.ipynb` | Model analizleri ve tüm deney kodu |
| `Bert_Rapor.pdf` | Araştırma raporu (2 sayfa, akademik format) |

---

## Kullanılan Teknolojiler

- Python 3.10
- `transformers` 5.7.0 — BERT modeli ve pipeline'lar
- `torch` 2.11.0 — tensor işlemleri
- `scikit-learn` — cosine similarity hesabı
- HuggingFace Hub — model indirme

---

## Modeller

| Model | Görev | Kaynak |
|-------|-------|--------|
| `dbmdz/bert-base-turkish-cased` | Embedding analizi, MLM, ham sınıflandırma | HuggingFace Hub |
| `savasy/bert-base-turkish-sentiment-cased` | Fine-tuned duygu analizi (bonus) | HuggingFace Hub |

---

## Deney Özeti

### Part 1 — Contextual Embedding Analizi
`get_embedding()` fonksiyonu ile her cümledeki CLS token vektörü çıkarılmış, aynı kelimenin iki farklı anlamdaki cümlesi karşılaştırılmıştır.

| Kelime | Cosine Similarity |
|--------|------------------|
| banka  | 0.9497 |
| kol    | **0.8179** ← en belirgin ayrışma |
| yüz    | 0.8900 |
| dil    | 0.9488 |

### Part 2 — MLM Maskeli Tahmin Deneyi
`fill-mask` pipeline ile üç cümle test edilmiştir. `banka` kelimesi bağlama göre finans ve fiziksel mekân olmak üzere iki farklı tahmin kümesi üretmiştir.

### Part 3 — Limitasyon Analizi
Ham BERT'in `classifier.weight | MISSING` uyarısıyla rastgele başlatılmış ağırlıklarla çalıştığı ve anlamlı duygu ayrımı yapamadığı gözlemlenmiştir.

### Bonus — Fine-tuned Karşılaştırma
| Cümle | Ham BERT | Fine-tuned |
|-------|----------|------------|
| Bu film harikaydı | LABEL_0 — 0.6624 | positive — **0.9838** |
| Bu film çok kötüydü | LABEL_0 — 0.6720 | negative — **0.9934** |

---

## Kurulum ve Çalıştırma

```bash
pip install transformers torch scikit-learn
```

Notebook'u açıp hücreleri sırayla çalıştırın. Model ilk çalıştırmada HuggingFace Hub'dan otomatik indirilir (~500 MB).

> **Not:** HuggingFace token tanımlanmamışsa yükleme yavaş olabilir. `huggingface-cli login` ile token eklenebilir.

---

## Kurs Bilgisi

Bu çalışma **BM5204 Doğal Dil İşleme** dersi kapsamında hazırlanmıştır.

Hafta 6 ödevi ile bağlantılıdır → [`NLP_Word2Vec_FastText`](https://github.com/PEKTASCH/NLP_Word2Vec_FastText)
: Aynı çok anlamlı kelimeler üzerinde statik embedding karşılaştırması yapılmış, polisemi problemi gözlemlenmiştir.
