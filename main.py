#!/usr/bin/env python3

import sys
import os
import traceback
import time
from gui import FaceRecognitionGUI

def check_dependencies():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder"""
    required_packages = [
        'cv2',
        'numpy',
        'insightface',
        'PIL',
        'tkinter'
    ]
    
    missing_packages = []
    
    print(" Kütüphane kontrolü yapılıyor...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"   Yüklendi {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   Yükleme başarısız {package}")
    
    if missing_packages:
        print("\n Eksik kütüphaneler:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n Kütüphaneleri yüklemek için:")
        print("   pip install -r requirements.txt")
        return False
    
    print(" Tüm kütüphaneler mevcut!\n")
    return True

def create_directories():
    """Gerekli dizinleri oluşturur"""
    directories = [
        'photos',
        'models',
        'logs'
    ]
    
    print(" Dizin kontrolü yapılıyor...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"    {directory} dizini oluşturuldu")
        else:
            print(f"    {directory} dizini mevcut")

def print_system_info():
    """Sistem bilgilerini yazdırır"""
    print("💻 Sistem Bilgileri:")
    print(f"   Python Sürümü: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")
    print(f"   Çalışma Dizini: {os.getcwd()}")
    print()

def print_startup_info():
    """Başlangıç bilgilerini yazdırır"""
    print("  ÖNEMLİ BİLGİLER:")
    print("   • İlk çalıştırmada modeller internet üzerinden indirilir")
    print("   • Model indirme süreci 2-5 dakika sürebilir")
    print("   • İnternet bağlantınızın aktif olduğundan emin olun")
    print("   • Firewall/antivirus ayarlarını kontrol edin")
    print("   • GUI açıldıktan sonra sağ tarafta detaylı logları görebilirsiniz")
    print()
    print(" GUI açılacak ve model yükleme işlemi başlayacak...")
    print("   Log alanından ilerlemeyi takip edebilirsiniz.")
    print()

def main():
    """Ana uygulama fonksiyonu"""
    start_time = time.time()
    
    print("🎓 Yüz Tanıma Sistemi başlatılıyor...")
    print("=" * 60)
    
    # Sistem bilgilerini göster
    print_system_info()
    
    # Kütüphane kontrolü
    if not check_dependencies():
        print("\n⚠️  Lütfen eksik kütüphaneleri yükleyip tekrar deneyin.")
        input("Devam etmek için Enter'a basın...")
        sys.exit(1)
    
    # Dizinleri oluştur
    create_directories()
    print()
    
    # Başlangıç bilgileri
    print_startup_info()
    
    try:
        # GUI'yi başlat
        print(" GUI arayüzü açılıyor...")
        print("   ⏳ Modeller yüklenene kadar lütfen bekleyin...")
        print()
        
        app = FaceRecognitionGUI()
        
        startup_time = time.time() - start_time
        print(f"Uygulama {startup_time:.1f} saniyede başlatıldı")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n Uygulama kullanıcı tarafından kapatılıyor...")
        
    except Exception as e:
        print(f"\n  Beklenmeyen hata: {e}")
        print(f" Hata türü: {type(e).__name__}")
        print("\n Detaylı hata bilgisi:")
        traceback.print_exc()
        
        # Hata logunu dosyaya yaz
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/error.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Hata Zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Hata: {e}\n")
                f.write(f"Hata Türü: {type(e).__name__}\n")
                f.write(traceback.format_exc())
                f.write(f"{'='*60}\n")
            print(" Hata detayları 'logs/error.log' dosyasına kaydedildi.")
        except:
            print("  Hata logu kaydedilemedi.")
        
        # Yaygın hataların çözümleri
        if "insightface" in str(e).lower():
            print("\n InsightFace ile ilgili problem:")
            print("   - pip install insightface komutu ile yeniden yükleyin")
            print("   - Python 3.8+ kullandığınızdan emin olun")
        
        if "cv2" in str(e).lower():
            print("\n OpenCV ile ilgili problem:")
            print("   - pip install opencv-python komutu ile yeniden yükleyin")
        
        print(f"\n Hata oluşma zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        input("Devam etmek için Enter'a basın...")

if __name__ == "__main__":
    main()
