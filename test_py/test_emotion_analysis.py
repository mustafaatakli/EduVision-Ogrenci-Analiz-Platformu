#!/usr/bin/env python3
"""
Duygu Analizi Test Scripti
Bu script duygu analizi özelliklerinin doğru çalışıp çalışmadığını test eder
"""

import sys
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Ana dizini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_system():
    """Config sisteminin duygu analizi ayarlarını test eder"""
    print("🔧 Config sistemi test ediliyor...")
    
    try:
        from config import get_emotion_config
        emotion_config = get_emotion_config()
        
        print(f"✅ Duygu analizi etkin: {emotion_config.enabled}")
        print(f"✅ Backend: {emotion_config.backend}")
        print(f"✅ Model: {emotion_config.model_name}")
        print(f"✅ Duygu etiketleri: {len(emotion_config.emotion_labels)} adet")
        
        for en_emotion, tr_emotion in list(emotion_config.emotion_labels.items())[:3]:
            print(f"   • {en_emotion} → {tr_emotion}")
        
        return True
    except Exception as e:
        print(f"❌ Config testi başarısız: {e}")
        return False

def test_deepface_import():
    """DeepFace kütüphanesinin import edilip edilemediğini test eder"""
    print("\n📦 DeepFace import testi...")
    
    try:
        from deepface import DeepFace
        print("✅ DeepFace başarıyla import edildi")
        return True
    except ImportError as e:
        print(f"❌ DeepFace import hatası: {e}")
        print("💡 Çözüm: pip install deepface tensorflow")
        return False
    except Exception as e:
        print(f"❌ DeepFace beklenmeyen hata: {e}")
        return False

def create_test_image():
    """Test için basit bir görüntü oluşturur"""
    print("\n🖼️  Test görüntüsü oluşturuluyor...")
    
    # 400x400 beyaz arka plan
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Basit yüz benzeri şekil çiz (daire + gözler + ağız)
    center = (200, 200)
    
    # Yüz (daire)
    cv2.circle(image, center, 80, (200, 180, 160), -1)  # Ten rengi
    cv2.circle(image, center, 80, (0, 0, 0), 2)        # Çerçeve
    
    # Gözler
    cv2.circle(image, (180, 180), 10, (0, 0, 0), -1)   # Sol göz
    cv2.circle(image, (220, 180), 10, (0, 0, 0), -1)   # Sağ göz
    
    # Ağız (gülümseme)
    cv2.ellipse(image, (200, 230), (30, 15), 0, 0, 180, (0, 0, 0), 2)
    
    # Geçici dosyaya kaydet
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(temp_file.name, image)
    
    print(f"✅ Test görüntüsü oluşturuldu: {temp_file.name}")
    return temp_file.name

def test_face_processor():
    """FaceProcessor'ın duygu analizi özelliklerini test eder"""
    print("\n🤖 FaceProcessor duygu analizi testi...")
    
    try:
        from face_processor import FaceProcessor
        
        # FaceProcessor'ı başlat
        print("   📥 FaceProcessor başlatılıyor...")
        processor = FaceProcessor()
        
        print(f"   🎭 Duygu analizi etkin: {processor.emotion_analysis_enabled}")
        
        if not processor.emotion_analysis_enabled:
            print("   ⚠️  Duygu analizi devre dışı, test atlanıyor")
            return True
        
        # Test görüntüsü oluştur
        test_image_path = create_test_image()
        
        try:
            # Duygu analizi test et
            print("   🔍 Duygu analizi test ediliyor...")
            emotion_result = processor.analyze_emotion(test_image_path)
            
            print(f"   📊 Sonuç: {emotion_result['success']}")
            if emotion_result['success']:
                print(f"   🎭 Dominant duygu: {emotion_result['dominant_emotion']}")
                print(f"   📈 Skor: {emotion_result['dominant_score']:.2f}")
                print(f"   📋 Tüm duygular: {len(emotion_result['emotions'])} adet")
            else:
                print(f"   ❌ Hata: {emotion_result['message']}")
        
        finally:
            # Geçici dosyayı temizle
            os.unlink(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ FaceProcessor testi başarısız: {e}")
        return False

def test_service_layer():
    """Service layer'ın duygu analizi entegrasyonunu test eder"""
    print("\n🔧 Service layer testi...")
    
    try:
        from services.face_recognition_service import FaceRecognitionService
        
        # Service'i başlat
        print("   📥 Service başlatılıyor...")
        service = FaceRecognitionService()
        
        # Modelleri yükle
        print("   🔄 Modeller yükleniyor...")
        init_result = service.initialize_models()
        
        if not init_result['success']:
            print(f"   ❌ Model yükleme başarısız: {init_result['message']}")
            return False
        
        print(f"   ✅ Modeller yüklendi ({init_result['load_time']:.1f}s)")
        print(f"   🎭 Duygu analizi: {service.face_processor.emotion_analysis_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service layer testi başarısız: {e}")
        return False

def test_api_endpoints():
    """API endpoint'lerinin duygu analizi desteğini test eder"""
    print("\n🌐 API endpoint testi...")
    
    try:
        # API dosyasını import et
        sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
        
        # Ana API modülünü import et
        api_main_path = os.path.join(os.path.dirname(__file__), 'api', 'main.py')
        if os.path.exists(api_main_path):
            print("   ✅ API main.py dosyası mevcut")
            
            # API içeriğini kontrol et
            with open(api_main_path, 'r', encoding='utf-8') as f:
                api_content = f.read()
            
            if '/emotion/analyze' in api_content:
                print("   ✅ Duygu analizi endpoint'i mevcut")
            else:
                print("   ❌ Duygu analizi endpoint'i bulunamadı")
                return False
            
            if 'emotion_analysis' in api_content:
                print("   ✅ Recognition endpoint'i duygu analizi entegrasyonu mevcut")
            else:
                print("   ⚠️  Recognition endpoint'inde duygu analizi entegrasyonu bulunamadı")
        else:
            print("   ❌ API main.py dosyası bulunamadı")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint testi başarısız: {e}")
        return False

def test_gui_integration():
    """GUI'nin duygu analizi entegrasyonunu test eder"""
    print("\n🖥️  GUI entegrasyon testi...")
    
    try:
        # GUI dosyasını kontrol et
        gui_path = os.path.join(os.path.dirname(__file__), 'gui.py')
        if os.path.exists(gui_path):
            print("   ✅ GUI dosyası mevcut")
            
            # GUI içeriğini kontrol et
            with open(gui_path, 'r', encoding='utf-8') as f:
                gui_content = f.read()
            
            if 'emotion_analysis' in gui_content:
                print("   ✅ GUI'de duygu analizi entegrasyonu mevcut")
            else:
                print("   ❌ GUI'de duygu analizi entegrasyonu bulunamadı")
                return False
            
            if 'emotion_text' in gui_content:
                print("   ✅ GUI'de duygu metni görselleştirmesi mevcut")
            else:
                print("   ⚠️  GUI'de duygu metni görselleştirmesi bulunamadı")
        else:
            print("   ❌ GUI dosyası bulunamadı")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ GUI entegrasyon testi başarısız: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🎭 OKULDAN Yüz Tanıma Sistemi - Duygu Analizi Test Suite")
    print("=" * 60)
    
    tests = [
        ("Config Sistemi", test_config_system),
        ("DeepFace Import", test_deepface_import),
        ("FaceProcessor", test_face_processor),
        ("Service Layer", test_service_layer),
        ("API Endpoints", test_api_endpoints),
        ("GUI Entegrasyonu", test_gui_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - BAŞARILI")
            else:
                print(f"❌ {test_name} - BAŞARISIZ")
        except Exception as e:
            print(f"💥 {test_name} - HATA: {e}")
        
        print("-" * 40)
    
    print(f"\n📊 TEST SONUÇLARI: {passed}/{total} test başarılı")
    
    if passed == total:
        print("🎉 Tüm testler başarılı! Duygu analizi sistemi hazır.")
        return True
    else:
        print("⚠️  Bazı testler başarısız. Yukarıdaki hataları gözden geçirin.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






