#!/usr/bin/env python3
"""
GERÇEK YÜZZ TANIMA PERFORMANS TESTİ
InsightFace Buffalo_l modelinin farklı fotoğraflarda aynı kişiyi tanıma kabiliyetini test eder
"""

import os
import cv2
import numpy as np
from face_processor import FaceProcessor
from database import DatabaseManager
import glob

class RecognitionPerformanceTester:
    def __init__(self):
        """Test sistemini başlat"""
        self.face_processor = FaceProcessor()
        self.db_manager = DatabaseManager()
        
    def test_same_person_different_photos(self, test_photos_dir):
        """
        Aynı kişinin farklı fotoğraflarını test eder
        
        Klasör yapısı:
        test_photos/
        ├── person1/
        │   ├── register.jpg  (kayıt için)
        │   ├── test1.jpg     (test için)
        │   ├── test2.jpg     (test için)
        └── person2/
            ├── register.jpg
            ├── test1.jpg
        """
        print("🧪 GERÇEK YÜZZ TANIMA PERFORMANS TESTİ")
        print("=" * 50)
        
        if not os.path.exists(test_photos_dir):
            print(f"❌ Test klasörü bulunamadı: {test_photos_dir}")
            return
        
        # Her kişi için test
        person_dirs = [d for d in os.listdir(test_photos_dir) 
                      if os.path.isdir(os.path.join(test_photos_dir, d))]
        
        if not person_dirs:
            print("❌ Test klasöründe kişi klasörleri bulunamadı")
            return
        
        total_tests = 0
        successful_recognitions = 0
        similarity_scores = []
        
        for person_name in person_dirs:
            person_dir = os.path.join(test_photos_dir, person_name)
            print(f"\n👤 Test: {person_name}")
            print("-" * 30)
            
            # Kayıt fotoğrafını bul
            register_photo = os.path.join(person_dir, "register.jpg")
            if not os.path.exists(register_photo):
                register_photos = glob.glob(os.path.join(person_dir, "*register*"))
                if register_photos:
                    register_photo = register_photos[0]
                else:
                    print(f"⚠️  Kayıt fotoğrafı bulunamadı: {person_name}")
                    continue
            
            # Kayıt fotoğrafından embedding çıkar
            try:
                register_faces = self.face_processor.detect_faces(register_photo)
                if not register_faces:
                    print(f"⚠️  Kayıt fotoğrafında yüz bulunamadı: {register_photo}")
                    continue
                
                register_embedding = register_faces[0]['embedding']
                print(f"✅ Kayıt embedding'i hazır: {register_embedding.shape}")
                
            except Exception as e:
                print(f"❌ Kayıt embedding hatası: {e}")
                continue
            
            # Test fotoğraflarını bul
            test_photos = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_photos.extend(glob.glob(os.path.join(person_dir, ext)))
            
            # register.jpg'yi çıkar
            test_photos = [p for p in test_photos if 'register' not in os.path.basename(p).lower()]
            
            if not test_photos:
                print(f"⚠️  Test fotoğrafı bulunamadı: {person_name}")
                continue
            
            # Her test fotoğrafı için
            for test_photo in test_photos:
                total_tests += 1
                print(f"\n🔍 Test fotoğrafı: {os.path.basename(test_photo)}")
                
                try:
                    # Test fotoğrafından embedding çıkar
                    test_faces = self.face_processor.detect_faces(test_photo)
                    if not test_faces:
                        print("❌ Yüz bulunamadı")
                        continue
                    
                    test_embedding = test_faces[0]['embedding']
                    
                    # Benzerlik hesapla
                    similarity = self.face_processor.compare_embeddings(
                        register_embedding, test_embedding
                    )
                    
                    similarity_scores.append(similarity)
                    
                    # Farklı threshold'larda test et
                    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    print(f"📊 Benzerlik skoru: {similarity:.3f}")
                    
                    for threshold in thresholds:
                        status = "✅ TANINDI" if similarity >= threshold else "❌ TANıNMADI"
                        print(f"   Threshold {threshold:.1f}: {status}")
                    
                    # Mevcut sistem threshold'u (0.7)
                    if similarity >= 0.7:
                        successful_recognitions += 1
                        print(f"🎯 Mevcut sistem (≥0.7): ✅ BAŞARILI")
                    else:
                        print(f"🎯 Mevcut sistem (≥0.7): ❌ BAŞARISIZ")
                    
                except Exception as e:
                    print(f"❌ Test hatası: {e}")
        
        # SONUÇLAR
        print("\n" + "=" * 50)
        print("📊 TEST SONUÇLARI")
        print("=" * 50)
        
        if total_tests > 0:
            success_rate = (successful_recognitions / total_tests) * 100
            print(f"🎯 Toplam test: {total_tests}")
            print(f"✅ Başarılı tanıma: {successful_recognitions}")
            print(f"📈 Başarı oranı: {success_rate:.1f}%")
            
            if similarity_scores:
                avg_similarity = np.mean(similarity_scores)
                min_similarity = np.min(similarity_scores)
                max_similarity = np.max(similarity_scores)
                
                print(f"\n📊 Benzerlik İstatistikleri:")
                print(f"   Ortalama: {avg_similarity:.3f}")
                print(f"   Minimum: {min_similarity:.3f}")
                print(f"   Maksimum: {max_similarity:.3f}")
                
                # Threshold önerileri
                print(f"\n💡 Threshold Önerileri:")
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    success_count = sum(1 for s in similarity_scores if s >= threshold)
                    success_percent = (success_count / len(similarity_scores)) * 100
                    print(f"   {threshold:.1f} threshold: {success_percent:.1f}% başarı")
                
                # En iyi threshold önerisi
                best_threshold = 0.5  # Default
                for threshold in [0.4, 0.5, 0.6]:
                    success_count = sum(1 for s in similarity_scores if s >= threshold)
                    if success_count / len(similarity_scores) >= 0.9:  # %90+ başarı
                        best_threshold = threshold
                        break
                
                print(f"\n🎯 ÖNERİLEN THRESHOLD: {best_threshold:.1f}")
                print(f"   Bu threshold ile %90+ tanıma başarısı beklenir")
        else:
            print("❌ Hiç test yapılamadı")
    
    def create_sample_test_structure(self, base_dir="test_photos"):
        """Test için örnek klasör yapısı oluşturur"""
        print(f"📁 Örnek test yapısı oluşturuluyor: {base_dir}")
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Örnek kişiler için klasörler
        sample_persons = ["ahmet", "ayse", "mehmet"]
        
        for person in sample_persons:
            person_dir = os.path.join(base_dir, person)
            os.makedirs(person_dir, exist_ok=True)
            
            # README dosyası
            readme_path = os.path.join(person_dir, "README.txt")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"""Bu klasör {person} için test fotoğraflarını içermelidir:

1. register.jpg - Kayıt için kullanılacak fotoğraf
2. test1.jpg - Test fotoğrafı 1 (farklı açı/aydınlatma)
3. test2.jpg - Test fotoğrafı 2 (farklı açı/aydınlatma)
4. test3.jpg - Test fotoğrafı 3 (opsiyonel)

ÖNEMLI: Tüm fotoğraflar aynı kişiye ait olmalı!
""")
        
        print(f"✅ Test yapısı hazır: {base_dir}")
        print(f"💡 Her kişi klasörüne o kişinin fotoğraflarını ekleyin")
        print(f"📖 Detaylar için README.txt dosyalarını okuyun")

if __name__ == "__main__":
    tester = RecognitionPerformanceTester()
    
    # Örnek test yapısı oluştur
    tester.create_sample_test_structure()
    
    # Test çalıştır (fotoğraflar varsa)
    print("\n" + "="*50)
    print("🚀 Test fotoğrafları hazırsa test başlatılacak...")
    tester.test_same_person_different_photos("test_photos")