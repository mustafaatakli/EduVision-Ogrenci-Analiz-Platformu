[🇹🇷 Türkçe](#-okullar-için-yüz-tanıma-ve-duygu-analizi-sistemi-masaüstü-uygulaması) | [🇬🇧 English](#-face-recognition-and-emotion-analysis-system-for-schools-desktop-application)

---

# 🎓 Okullar İçin Yüz Tanıma ve Duygu Analizi Sistemi (Masaüstü Uygulaması)

Bu proje, okullara yönelik tasarlanmış modern bir Yüz Tanıma ve Duygu Analizi Masaüstü Uygulamasıdır. Python tabanlı bu sistem, gelişmiş yapay zeka modellerini kullanarak öğrenci kayıt, tanıma ve duygu analizi süreçlerini otomatikleştirmeyi amaçlamaktadır.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![InsightFace](https://img.shields.io/badge/AI-InsightFace-orange.svg)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-red.svg)

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Teknolojiler](#-teknolojiler)
- [Dosya Yapısı](#-dosya-yapısı)
- [Gereksinimler](#-gereksinimler)
- [Kullanım](#-kullanım)
- [Bilinen Sorunlar](#-bilinen-sorunlar)
- [Teşekkürler](#-teşekkürler)
- [İletişim](#-i̇letişim)
- [Lisans ve Telif Hakkı](#-lisans-ve-telif-hakkı)
- [Örnek Çalışma Görüntüleri](#-örnek-çalışma-görüntüleri)
- [Geliştirici](#-geliştirici)

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

### Entegrasyon
- Projenin ana mantığı (yüz tanıma, duygu analizi), aynı zamanda bir API olarak tasarlanmıştır.
- Geliştirilen bu API, bir mobil uygulamaya (iOS/Android) başarıyla entegre edilerek, sistemin mobil platformlarda da kullanılması sağlanmıştır.

### Kullanım Senaryoları

1. **Yoklama Takibi:** Öğretmen, derste bir sınıf fotoğrafı çektiğinde, uygulama bu fotoğraftaki öğrencilerin yüzlerini tanır ve anında kimlerin derste, kimlerin derste olmadığını belirler. Bu, manuel yoklama işlemini ortadan kaldırır.

2. **Öğrenci Durum Takibi:** Sistem, yüzleri tanırken aynı zamanda öğrencilerin duygularını da (mutlu, üzgün, şaşkın, ilgisiz vb.) analiz eder. Bu sayede öğretmen, dersin genel atmosferini ve öğrencilerin konuya olan ilgisini anlayarak dersin gidişatına müdahale edebilir.

---

## 🐛 Bilinen Sorunlar

- İlk model indirmesi internet kesilirse tekrar başlatma gerekir
- CUDA desteği için ek konfigürasyon gerekebilir
- Tkinter bazı Linux dağıtımlarında ek paket gerektirebilir

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden faydalanmıştır:

- [InsightFace](https://github.com/deepinsight/insightface) - Yüz tanıma
- [DeepFace](https://github.com/serengil/deepface) - Duygu analizi
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [OpenCV](https://opencv.org/) - Görüntü işleme

---

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

## 🖼️ Örnek Çalışma Görüntüleri

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184159.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184224.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184252.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184349.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184443.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184520.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184542.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184956.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20185051.png" width="auto">

---

## 👨‍💻 Geliştirici

**Mustafa Ataklı**

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

---
---
---
# ENG
# 🎓 Face Recognition and Emotion Analysis System for Schools (Desktop Application)

This project is a modern Face Recognition and Emotion Analysis Desktop Application designed for schools. This Python-based system aims to automate student registration, recognition, and emotion analysis processes using advanced artificial intelligence models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![InsightFace](https://img.shields.io/badge/AI-InsightFace-orange.svg)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-red.svg)

## 📋 Contents

- [Features](#-features)
- [Technologies](#-technologies)
- [File Structure](#-file-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Known Issues](#-known-issues)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)
- [License and Copyright](#-license-and-copyright)
- [Sample Work Images](#-sample-work-images)
- [Developer](#-developer)

## ✨ Features

### 🤖 Artificial Intelligence Capabilities
- **Face Detection**: High-accuracy face detection with RetinaFace algorithm
- **Face Recognition**: 512-dimensional embedding extraction with InsightFace Buffalo_l model
- **Emotion Analysis**: Detection of 7 basic emotions (Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral) with DeepFace
- **Quality Control**: Automatic photo quality analysis with 5 different criteria
- Quality metrics and data for unrecognized faces are stored in the database, providing manual support by the developer for users who cannot be registered in the system.

### 📸 Photo Processing
- Single or group photo support
- Multiple face detection and recognition
- Automatic photo quality control
- Manual registration system (for unrecognized faces)

### 💾 Database
- SQLite and MSSQL support
- Automatic backup system
- Database migration tools

### 🎨 User Interface
- Modern Tkinter-based GUI
- Real-time log display
- Emotion analysis visualization
- Intuitive menu structure

## 🛠 Technologies

### AI/ML
- **InsightFace** (0.7.3) - Face recognition and embedding
- **DeepFace** (≥0.0.79) - Emotion analysis
- **TensorFlow** (≥2.12.0) - Deep learning framework
- **OpenCV** (4.8.1.78) - Image processing

### Backend
- **Python** (3.9+)
- **SQLAlchemy** (≥2.0.0) - ORM
- **PyODBC** (≥4.0.39) - MSSQL connection

### GUI
- **Tkinter** - Python built-in GUI
- **Pillow** (10.0.1) - Image manipulation

### Configuration
- **PyYAML** (≥6.0) - YAML parser
- **python-dotenv** (≥1.0.0) - Environment variables

## 📁 File Structure

```
okuldan-face-recognition/
│
├── 📄 main.py                      # Main application entry point
├── 📄 config.py                    # Configuration management
├── 📄 database.py                  # Database operations (SQLite/MSSQL)
├── 📄 face_processor.py            # Face processing and AI models
├── 📄 gui.py                       # Tkinter GUI interface
├── 📄 optimize_thresholds.py       # Threshold optimization tool
│
├── 📋 requirements.txt             # Python dependencies
│
├── 📄 README_GITHUB.md             # This file
│
├── 📁 test_py/                     # Test files
│   ├── test_system_accuracy.py    # System accuracy tests
│   ├── test_recognition_performance.py  # Performance tests
│   ├── test_emotion_analysis.py   # Emotion analysis tests
│   └── test_models.py              # Model loading tests
│
├── 📁 models/                      # AI models (downloaded automatically)
│   ├── buffalo_l/                  # InsightFace face recognition model
│   │   ├── det_10g.onnx           # Face detection model
│   │   ├── w600k_r50.onnx         # Recognition model
│   │   └── ...
│   └── deepface_models/            # DeepFace emotion analysis models
│       └── emotion-ferplus-8.onnx
│
│
├── 📁 logs/                        # Log files
│   └── app.log                    # Application logs
```

### Main Files

| File | Description | Line Count |
|------|-------------|------------|
| `main.py` | Application entry point, launches GUI | ~150 |
| `config.py` | Configuration loading from YAML and .env files | ~600 |
| `database.py` | SQLAlchemy ORM models and database operations | ~1000 |
| `face_processor.py` | InsightFace/DeepFace integration, face processing | ~1300 |
| `gui.py` | Tkinter-based user interface | ~4800 |
| `optimize_thresholds.py` | Recognition threshold optimization tool | ~160 |

### Configuration Files

| File | Format | Description |
|------|--------|-------------|
| `config.yaml` | YAML | Main application settings, AI parameters |
| `.env` | ENV | Sensitive information (DB passwords, API keys) |
| `requirements.txt` | TXT | Python package dependencies |

### Database Structure

The system uses the following tables:

- `students` - Student information (name, ID, registration date)
- `student_photos` - Photo data and embeddings
- `photo_quality_scores` - Photo quality scores
- `face_recognitions` - Recognition transaction records
- `emotion_analyses` - Emotion analysis results

### Model Files

**InsightFace Buffalo_l** (~600 MB):
- `det_10g.onnx` - RetinaFace face detection model
- `w600k_r50.onnx` - ArcFace recognition model (512D embeddings)
- `genderage.onnx` - Age/gender prediction (optional)

**DeepFace Emotion** (~30 MB):
- `emotion-ferplus-8.onnx` - FER+ emotion analysis model

## 📦 Requirements

### System Requirements
- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.9 or higher
- **RAM**: Minimum 4 GB (recommended 8 GB)
- **Disk**: 2 GB free space (for models)
- **Internet**: Required for initial setup (model download)

### Software Requirements
- Python 3.9+
- pip (Python package manager)
- Git (optional)

### For MSSQL Usage (Optional)
- Microsoft SQL Server (Express or higher)
- ODBC Driver 17 for SQL Server

## 📖 Usage

### Student Registration

1. Click "👨‍🎓 STUDENT REGISTRATION" option from the main menu
2. Enter student name and ID
3. Add photos sequentially with "📸 ADD SINGLE PHOTO"
4. System performs automatic quality analysis
5. Automatic registration occurs after 2+ quality photos

### Face Recognition

1. Click "🔍 FACE RECOGNITION" option from the main menu
2. Click "📷 Load Photo" button
3. Select the photo to be recognized
4. System performs automatic recognition and displays results
5. "👤 Manual Registration" button appears for unrecognized faces

### Manual Registration

To manually register unrecognized faces:
- Click the "Manual Registration" button that appears after recognition
- Enter student information
- Save

### Photo Quality Criteria

The system evaluates photo quality with 5 criteria:

1. **Sharpness (Critical)**: Laplacian variance > 100
2. **Eyes (Support)**: Eye openness > 0.5
3. **Angle (Critical)**: Yaw ≤30°, Pitch ≤30°, Roll ≤15°
4. **Completeness (Critical)**: Face completeness ≥ 85%
5. **Lighting (Support)**: Brightness between 50-200

**Acceptance Criteria**: Must pass 3/3 critical + 1/2 support criteria

### Integration
- The main logic of the project (face recognition, emotion analysis) is also designed as an API.
- This developed API has been successfully integrated into a mobile application (iOS/Android), enabling the system to be used on mobile platforms as well.

### Use Case Scenarios

1. **Attendance Tracking:** When a teacher takes a class photo during the lesson, the application recognizes the faces of students in the photo and instantly determines who is present and who is absent. This eliminates the need for manual attendance taking.

2. **Student Status Monitoring:** While recognizing faces, the system also analyzes students' emotions (happy, sad, surprised, disengaged, etc.). This allows the teacher to understand the general atmosphere of the class and students' interest in the subject, enabling intervention in the course of the lesson.

---

## 🐛 Known Issues

- Model download requires restart if internet is interrupted
- Additional configuration may be required for CUDA support
- Tkinter may require additional packages on some Linux distributions

---

## 🙏 Acknowledgments

This project has benefited from the following open-source projects:

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [DeepFace](https://github.com/serengil/deepface) - Emotion analysis
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [OpenCV](https://opencv.org/) - Image processing

---

## 📧 Contact

For questions and suggestions:
- GitHub Issues: [Open New Issue](https://github.com/KULLANICI_ADI/okuldan-face-recognition/issues)
- Email: atakliim20@gmail.com

---

## 📄 License and Copyright

Copyright (c) [2025] [Mustafa Ataklı]

**All Rights Reserved.**

This project is exhibited for portfolio purposes. No part of this software and source code may be copied, reproduced, distributed, or used for commercial purposes without written and explicit permission.

---

## 🖼️ Sample Work Images

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184159.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184224.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184252.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184349.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184443.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184520.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184542.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20184956.png" width="auto">

---

<img src="https://github.com/mustafaatakli/Okullar-Icin-Yuz-Tanima-ve-Duygu-Analizi-Sistemi/blob/main/images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202025-10-28%20185051.png" width="auto">

---

## 👨‍💻 Developer

**Mustafa Ataklı**

⭐ If you liked this project, don't forget to give it a star!
