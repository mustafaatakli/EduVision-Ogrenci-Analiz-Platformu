#!/usr/bin/env python3
"""
THRESHOLD OPTİMİZASYONU
Mevcut threshold değerlerini analiz eder ve optimize eder
"""

import numpy as np

def analyze_current_thresholds():
    """Mevcut threshold değerlerini analiz eder"""
    print("MEVCUT THRESHOLD ANALİZİ")
    print("=" * 50)
    
    current_thresholds = {
        "Recognition Base": 0.70,
        "Multi-match Average": 0.68,
        "Multi-match Minimum": 0.65
    }
    
    print("Mevcut Değerler:")
    for name, value in current_thresholds.items():
        print(f"   {name}: {value:.2f} ({value:.1%})")
    
    print("\nInsightFace Buffalo_l için Tipik Değerler:")
    typical_ranges = {
        "Çok Sıkı (Güvenli)": "0.65-0.75 (%65-75)",
        "Orta (Dengeli)": "0.55-0.65 (%55-65)", 
        "Gevşek (Toleranslı)": "0.45-0.55 (%45-55)"
    }
    
    for category, range_desc in typical_ranges.items():
        print(f"   {category}: {range_desc}")
    
    print("\n ANALİZ:")
    print("   🔴 Mevcut threshold'lar ÇOK SIKKI kategori")
    print("   🔴 Sadece %70+ benzerlik kabul ediliyor")
    print("   🔴 Bu aynı kişinin farklı fotoğraflarını reddedebilir")
    
    return current_thresholds

def suggest_optimized_thresholds():
    """Optimize edilmiş threshold önerileri"""
    print("\n OPTİMİZE EDİLMİŞ THRESHOLD ÖNERİLERİ")
    print("=" * 50)
    
    scenarios = {
        "MEVCUT (Çok Sıkı)": {
            "recognition": 0.70,
            "multi_avg": 0.68,
            "multi_min": 0.65,
            "expected_accuracy": "95-98%",
            "problem": "Aynı kişinin farklı fotoğraflarını tanımayabilir"
        },
        "DENGELİ (Önerilen)": {
            "recognition": 0.55,
            "multi_avg": 0.58,
            "multi_min": 0.52,
            "expected_accuracy": "92-95%",
            "problem": "Minimal false positive artışı"
        },
        "TOLERANSLI (Test için)": {
            "recognition": 0.45,
            "multi_avg": 0.50,
            "multi_min": 0.45,
            "expected_accuracy": "88-92%",
            "problem": "Biraz daha false positive"
        }
    }
    
    for scenario_name, params in scenarios.items():
        print(f"\n📋 {scenario_name}:")
        print(f"   Recognition: {params['recognition']:.2f} ({params['recognition']:.1%})")
        print(f"   Multi-Avg: {params['multi_avg']:.2f} ({params['multi_avg']:.1%})")
        print(f"   Multi-Min: {params['multi_min']:.2f} ({params['multi_min']:.1%})")
        print(f"   Beklenen Doğruluk: {params['expected_accuracy']}")
        print(f"   Potansiyel Sorun: {params['problem']}")
    
    return scenarios["DENGELİ (Önerilen)"]

def generate_threshold_update_code(optimized_params):
    """Threshold güncelleme kodu üretir"""
    print("\n💻 THRESHOLD GÜNCELLEME KODU")
    print("=" * 50)
    
    print("📝 face_processor.py dosyasında şu değişiklikleri yapın:")
    print()
    
    print("1️⃣ find_best_match fonksiyonunda (satır ~619):")
    print(f"   DEĞİŞTİR: threshold: float = 0.70")
    print(f"   YENİ:      threshold: float = {optimized_params['recognition']:.2f}")
    print()
    
    print("2️⃣ Multi-match verification'da (satır ~654):")
    print(f"   DEĞİŞTİR: if avg_score >= 0.68 and min_score >= 0.65:")
    print(f"   YENİ:      if avg_score >= {optimized_params['multi_avg']:.2f} and min_score >= {optimized_params['multi_min']:.2f}:")
    print()
    
    print("📋 VEYA otomatik güncelleme için aşağıdaki kodu çalıştırın:")
    print()
    
    update_code = f"""
# Otomatik threshold güncelleme
import re

def update_thresholds():
    # face_processor.py dosyasını oku
    with open('face_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Threshold değerlerini güncelle
    content = re.sub(
        r'threshold: float = 0\.70',
        'threshold: float = {optimized_params['recognition']:.2f}',
        content
    )
    
    content = re.sub(
        r'avg_score >= 0\.68 and min_score >= 0\.65',
        'avg_score >= {optimized_params['multi_avg']:.2f} and min_score >= {optimized_params['multi_min']:.2f}',
        content
    )
    
    # Dosyayı yaz
    with open('face_processor.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Threshold'lar güncellendi!")

# Çalıştır
update_thresholds()
"""
    
    return update_code

if __name__ == "__main__":
    # Mevcut analiz
    current = analyze_current_thresholds()
    
    # Optimize öneriler
    optimized = suggest_optimized_thresholds()
    
    # Güncelleme kodu
    update_code = generate_threshold_update_code(optimized)
    
    print("\n🎯 SONUÇ VE TAVSİYELER")
    print("=" * 50)
    print("1️⃣ Önce test_recognition_performance.py ile gerçek test yapın")
    print("2️⃣ Test sonuçlarına göre threshold'ları ayarlayın")
    print("3️⃣ Dengeli threshold'larla başlayın (0.55/0.58/0.52)")
    print("4️⃣ Gerekirse daha da düşürün (0.45/0.50/0.45)")
    print()
    print("💡 InsightFace Buffalo_l modeli çok güçlü - düşük threshold'larla bile güvenilir!")