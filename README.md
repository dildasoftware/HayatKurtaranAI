# 🚑 HayatKurtaran AI — İlk Yardım ve Sağlık Chatbotu

> RAG (Retrieval-Augmented Generation) mimarisi ile güçlendirilmiş,  
> Türkçe doğal dili anlayan, halüsinasyon yapmayan ilk yardım asistanı.

---

## 🚀 Hızlı Başlangıç (2 Dakikada Çalıştır)

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Gemini API Anahtarını Ekle
`.env` dosyasını açın ve kendi anahtarınızı yazın:
```
GEMINI_API_KEY=BURAYA_API_ANAHTARINIZI_YAZIN
```
API anahtarı almak için: https://aistudio.google.com/app/apikey

### 3. Uygulamayı Başlat
```bash
streamlit run app.py
```

---

## 🧠 Mimari

```
Kullanıcı Sorusu
     │
     ▼
[Acil Kelime Tespiti] ─── Kritikse ──► 🚨 112 Uyarısı
     │
     ▼
[FAISS Vektör Arama]  ─── Bulunamazsa ──► Güvenli Yanıt
     │ (Top-K=3 chunk, %35+ benzerlik)
     ▼
[Gemini Flash LLM]    ─── Katı System Prompt (Halüsinasyon Koruması)
     │
     ▼
Madde İmli, Kısa, Net Türkçe Yanıt + Kaynak Etiketi
```

## 📁 Dosya Yapısı

```
life_saver_ai/
├── app.py                          # Streamlit arayüzü
├── requirements.txt
├── .env                            # API Key (Git'e gönderme!)
├── data/
│   ├── ilk_yardim_bilgileri.txt   # Temel ilk yardım verileri
│   ├── acil_durumlar.txt          # Acil durum yönergeleri
│   └── saglik_onerileri.txt       # Günlük sağlık önerileri
└── backend/
    ├── __init__.py
    ├── vector_db.py               # FAISS + Sentence-Transformers
    └── rag_engine.py              # Gemini LLM + RAG Pipeline
```

## ✨ Özellikler

| Özellik | Detay |
|---|---|
| 🔍 **Semantic Search** | Türkçe destekli multilingual embedding |
| 🛡️ **Halüsinasyon Koruması** | Bağlam dışı sorularda güvenli yanıt |
| 🚨 **Acil Tespit** | 14+ kritik kelime, anında 112 yönlendirmesi |
| ⚡ **Hızlı Butonlar** | 6 kritik senaryo tek tıkla erişimi |
| 📄 **Kaynak Şeffaflığı** | Her yanıtta kaynak dosya gösterimi |
| 🌙 **Koyu Tema** | Stres anında göz yormayan arayüz |

## ⚠️ Önemli Uyarı

Bu uygulama bir **ilk yardım bilgi asistanı**dır.  
Tıbbi tanı veya tedavi yerine **geçmez**.  
Acil durumlarda **derhal 112'yi arayın**.

---

## 📊 Teknik Stack

- **Frontend**: Streamlit
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vektör DB**: FAISS (CPU)
- **LLM**: Google Gemini 1.5 Flash
- **Chunking**: Semantic (paragraf bazlı)
