# 🎓 Okullar İçin Yüz Tanıma ve Duygu Analizi Sistemi (Masaüstü Uygulaması)

Modern yapay zeka teknolojileri kullanarak öğrenci yüz tanıma ve duygu analizi yapan masaüstü uygulaması olarak tasarlanip geliştirilmiştir.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![InsightFace](https://img.shields.io/badge/AI-InsightFace-orange.svg)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-red.svg)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Teknolojiler](#teknolojiler)
- [Dosya Yapısı](#dosya-yapısı)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Konfigürasyon](#konfigürasyon)
- [Dokümantasyon](#dokümantasyon)
- [Lisans](#lisans)

## ✨ Özellikler

### 🤖 Yapay Zeka Yetenekleri
- **Yüz Tespiti**: RetinaFace algoritması ile yüksek doğruluklu yüz tespiti
- **Yüz Tanıma**: InsightFace Buffalo_l modeli ile 512 boyutlu embedding çıkarımı
- **Duygu Analizi**: DeepFace ile 7 temel duygu tespiti (Mutlu, Üzgün, Kızgın, Şaşkın, Korkmuş, Tiksinmiş, Nötr)
- **Kalite Kontrol**: 5 farklı kriterde otomatik fotoğraf kalite analizi
- Tanınamayan yüzler için kalite metrik ve verileri veritabanında saklanmaktadır ve bu sayede de sisteme kaydedilemeyen kullanıcılar için geliştirici tarafından manuel destek sunulmaktadır.

### 📸 Fotoğraf İşleme
- Tekli veya grup fotoğrafı desteği
- Çoklu yüz tespiti ve tanıma
- Otomatik fotoğraf kalite kontrolü
- Manuel kayıt sistemi (tanınamayan yüzler için)

### 💾 Veritabanı
- SQLite ve MSSQL desteği
- Otomatik yedekleme sistemi
- Veritabanı migration araçları

### 🎨 Kullanıcı Arayüzü
- Modern Tkinter tabanlı GUI
- Gerçek zamanlı log gösterimi
- Duygu analizi görselleştirmesi
- Sezgisel menü yapısı

## 🛠 Teknolojiler

### AI/ML
- **InsightFace** (0.7.3) - Yüz tanıma ve embedding
- **DeepFace** (≥0.0.79) - Duygu analizi
- **TensorFlow** (≥2.12.0) - Deep learning framework
- **OpenCV** (4.8.1.78) - Görüntü işleme

### Backend
- **Python** (3.9+)
- **SQLAlchemy** (≥2.0.0) - ORM
- **PyODBC** (≥4.0.39) - MSSQL bağlantısı

### GUI
- **Tkinter** - Python built-in GUI
- **Pillow** (10.0.1) - Görüntü manipülasyonu

### Konfigürasyon
- **PyYAML** (≥6.0) - YAML parser
- **python-dotenv** (≥1.0.0) - Environment variables

## 📁 Dosya Yapısı

```
okuldan-face-recognition/
│
├── 📄 main.py                      # Ana uygulama giriş noktası
├── 📄 config.py                    # Konfigürasyon yönetimi
├── 📄 database.py                  # Veritabanı işlemleri (SQLite/MSSQL)
├── 📄 face_processor.py            # Yüz işleme ve AI modelleri
├── 📄 gui.py                       # Tkinter GUI arayüzü
├── 📄 optimize_thresholds.py       # Eşik değer optimizasyon aracı
│
├── 📋 requirements.txt             # Python bağımlılıkları

├── 📄 README_GITHUB.md             # Bu dosya
│
├── 📁 test_py/                     # Test dosyaları
│   ├── test_system_accuracy.py    # Sistem doğruluk testleri
│   ├── test_recognition_performance.py  # Performans testleri
│   ├── test_emotion_analysis.py   # Duygu analizi testleri
│   └── test_models.py              # Model yükleme testleri
│
├── 📁 models/                      # AI modelleri (otomatik indirilir)
│   ├── buffalo_l/                  # InsightFace yüz tanıma modeli
│   │   ├── det_10g.onnx           # Yüz tespit modeli
│   │   ├── w600k_r50.onnx         # Tanıma modeli
│   │   └── ...
│   └── deepface_models/            # DeepFace duygu analizi modelleri
│       └── emotion-ferplus-8.onnx
│
│
├── 📁 logs/                        # Log dosyaları
│   └── app.log                    # Uygulama logları
```

### Ana Dosyalar

| Dosya | Açıklama | Satır Sayısı |
|-------|----------|--------------|
| `main.py` | Uygulamanın başlangıç noktası, GUI'yi başlatır | ~150 |
| `config.py` | YAML ve .env dosyalarından konfigürasyon yükleme | ~600 |
| `database.py` | SQLAlchemy ORM modelleri ve veritabanı işlemleri | ~1000 |
| `face_processor.py` | InsightFace/DeepFace entegrasyonu, yüz işleme | ~1300 |
| `gui.py` | Tkinter tabanlı kullanıcı arayüzü | ~4800 |
| `optimize_thresholds.py` | Tanıma eşik değerlerini optimize etme aracı | ~160 |

### Konfigürasyon Dosyaları

| Dosya | Format | Açıklama |
|-------|--------|----------|
| `config.yaml` | YAML | Ana uygulama ayarları, AI parametreleri |
| `.env` | ENV | Hassas bilgiler (DB şifreleri, API anahtarları) |
| `requirements.txt` | TXT | Python paket bağımlılıkları |

### Veritabanı Yapısı

Sistem şu tabloları kullanır:

- `students` - Öğrenci bilgileri (ad, ID, kayıt tarihi)
- `student_photos` - Fotoğraf verileri ve embeddings
- `photo_quality_scores` - Fotoğraf kalite skorları
- `face_recognitions` - Tanıma işlem kayıtları
- `emotion_analyses` - Duygu analizi sonuçları

### Model Dosyaları

**InsightFace Buffalo_l** (~600 MB):
- `det_10g.onnx` - RetinaFace yüz tespit modeli
- `w600k_r50.onnx` - ArcFace tanıma modeli (512D embeddings)
- `genderage.onnx` - Yaş/cinsiyet tahmini (opsiyonel)

**DeepFace Emotion** (~30 MB):
- `emotion-ferplus-8.onnx` - FER+ duygu analizi modeli

## 📦 Gereksinimler

### Sistem Gereksinimleri
- **İşletim Sistemi**: Windows 10/11, Linux, macOS
- **Python**: 3.9 veya üzeri
- **RAM**: Minimum 4 GB (önerilen 8 GB)
- **Disk**: 2 GB boş alan (modeller için)
- **İnternet**: İlk kurulum için gerekli (model indirme)

### Yazılım Gereksinimleri
- Python 3.9+
- pip (Python paket yöneticisi)
- Git (opsiyonel)

### MSSQL Kullanımı için (Opsiyonel)
- Microsoft SQL Server (Express veya üzeri)
- ODBC Driver 17 for SQL Server

## 📖 Kullanım

### Öğrenci Kayıt

1. Ana menüden "👨‍🎓 ÖĞRENCİ KAYIT" seçeneğini tıklayın
2. Öğrenci adı ve ID'sini girin
3. "📸 TEK FOTOĞRAF EKLE" ile sırayla fotoğraf ekleyin
4. Sistem otomatik kalite analizi yapar
5. 2+ kaliteli fotoğraf sonrası otomatik kayıt gerçekleşir

### Yüz Tanıma

1. Ana menüden "🔍 YÜZ TANIMA" seçeneğini tıklayın
2. "📷 Fotoğraf Yükle" butonuna tıklayın
3. Tanınacak fotoğrafı seçin
4. Sistem otomatik tanıma yapar ve sonuçları gösterir
5. Tanınamayan yüzler için "👤 Manuel Kayıt" butonu görünür

### Manuel Kayıt

Tanınamayan yüzleri manuel olarak kaydetmek için:
- Tanıma sonrası görünen "Manuel Kayıt" butonuna tıklayın
- Öğrenci bilgilerini girin
- Kaydet

### Fotoğraf Kalite Kriterleri

Sistem 5 kriter ile fotoğraf kalitesini değerlendirir:

1. **Netlik (Kritik)**: Laplacian variance > 100
2. **Gözler (Destek)**: Göz açıklığı > 0.5
3. **Açı (Kritik)**: Yaw ≤30°, Pitch ≤30°, Roll ≤15°
4. **Bütünlük (Kritik)**: Yüz tamamlılığı ≥ 85%
5. **Işık (Destek)**: Parlaklık 50-200 arası

**Kabul Kriteri**: 3/3 kritik + 1/2 destek kriter geçmeli

## 🐛 Bilinen Sorunlar

- İlk model indirmesi internet kesilirse tekrar başlatma gerekir
- CUDA desteği için ek konfigürasyon gerekebilir
- Tkinter bazı Linux dağıtımlarında ek paket gerektirebilir

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden faydalanmıştır:

- [InsightFace](https://github.com/deepinsight/insightface) - Yüz tanıma
- [DeepFace](https://github.com/serengil/deepface) - Duygu analizi
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [OpenCV](https://opencv.org/) - Görüntü işleme

### Entegrasyon
- Projenin ana mantığı (yüz tanıma, duygu analizi), aynı zamanda bir API olarak tasarlanmıştır.
- Geliştirilen bu API, bir mobil uygulamaya (iOS/Android) başarıyla entegre edilerek, sistemin mobil platformlarda da kullanılması sağlanmıştır.

## 📧 İletişim

Sorularınız ve önerileriniz için:
- GitHub Issues: [Yeni Issue Aç](https://github.com/KULLANICI_ADI/okuldan-face-recognition/issues)
- Email: atakliim20@gmail.com

---

## 📄 Lisans ve Telif Hakkı

Copyright (c) [2025] [Mustafa Ataklı]

**Tüm Hakları Saklıdır (All Rights Reserved).**

Bu proje, portfolyo amacıyla sergilenmektedir. Bu yazılımın ve kaynak kodlarının hiçbir bölümü, yazılı ve açık izin alınmaksızın kopyalanamaz, çoğaltılamaz, dağıtılamaz veya ticari amaçla kullanılamaz.

---

## 👨‍💻 Geliştirici

**Mustafa Ataklı**

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
