#!/usr/bin/env python3
"""
Model Test Dosyası
Bu dosya ile modellerin doğru yüklenip yüklenmediğini test edebilirsiniz.
"""

import time
import sys
import traceback

def test_model_loading():
    """Model yükleme sürecini test eder"""
    print("🧪 Model yükleme testi başlatılıyor...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        print("1️⃣ InsightFace import edilmeye çalışılıyor...")
        import insightface
        print("   ✅ InsightFace başarıyla import edildi")
        
        print("\n2️⃣ FaceAnalysis modeli yükleniyor...")
        print("   ⏳ Bu işlem 2-5 dakika sürebilir (ilk kez çalıştırılıyorsa)")
        
        face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        print("   ✅ FaceAnalysis modeli oluşturuldu")
        
        print("\n3️⃣ Model hazırlanıyor...")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        elapsed = time.time() - start_time
        print(f"\n✅ Tüm modeller başarıyla yüklendi!")
        print(f"⏱️  Toplam süre: {elapsed:.1f} saniye")
        
        # Basit test
        print("\n4️⃣ Basit işlevsellik testi...")
        # Model bilgilerini yazdır
        print(f"   Model sayısı: {len(face_app.models)}")
        for name, model in face_app.models.items():
            print(f"   - {name}: {type(model).__name__}")
        
        print("\n🎉 Test başarıyla tamamlandı!")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Test başarısız!")
        print(f"⏱️  Hata oluşma zamanı: {elapsed:.1f} saniye")
        print(f"🔍 Hata: {e}")
        print(f"📋 Hata türü: {type(e).__name__}")
        
        # Yaygın hataların nedenleri
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            print("\n💡 İnternet bağlantı problemi:")
            print("   - İnternet bağlantınızı kontrol edin")
            print("   - VPN kapalı olduğundan emin olun")
            print("   - Firewall ayarlarını kontrol edin")
            
        elif "permission" in str(e).lower():
            print("\n💡 İzin problemi:")
            print("   - Uygulamayı yönetici olarak çalıştırmayı deneyin")
            print("   - Klasör yazma izinlerini kontrol edin")
            
        elif "memory" in str(e).lower() or "allocation" in str(e).lower():
            print("\n💡 Bellek problemi:")
            print("   - Diğer uygulamaları kapatın")
            print("   - En az 4GB RAM'in olduğundan emin olun")
        
        print(f"\n📋 Detaylı hata:")
        traceback.print_exc()
        
        return False

def main():
    """Ana test fonksiyonu"""
    print("🎓 Yüz Tanıma Sistemi - Model Test")
    print("Bu test, modellerin doğru yüklenip yüklenmediğini kontrol eder.\n")
    
    # Test başlat
    success = test_model_loading()
    
    if success:
        print("\n✅ Test başarılı! Ana uygulamayı çalıştırabilirsiniz:")
        print("   python main.py")
    else:
        print("\n❌ Test başarısız! Lütfen hataları düzeltin:")
        print("   1. İnternet bağlantınızı kontrol edin")
        print("   2. Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt")
        print("   3. Python 3.8+ kullandığınızdan emin olun")
    
    print(f"\n⏰ Test tamamlanma zamanı: {time.strftime('%H:%M:%S')}")
    input("\nÇıkmak için Enter'a basın...")

if __name__ == "__main__":
    main() 