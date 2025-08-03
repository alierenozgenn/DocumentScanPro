"""
Ana çalıştırma dosyası
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# src klasörünü Python path'ine ekle
sys.path.append(str(Path(__file__).parent / "src"))

# İşlem modüllerini içe aktar
from src.step1_masking import test_step1
from src.step2_contrast import process_step2
from src.step3_perspective import process_step3

def create_directories():
    """Gerekli klasörleri oluştur"""
    directories = ["input", "output"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Klasörler oluşturuldu")

def process_all_steps():
    """Tüm adımları sırayla çalıştır - HER ADIM ORİJİNAL INPUT KULLANIR"""
    print("🚀 EVRAK TARAMA İŞLEMİ BAŞLIYOR")
    print("=" * 60)
    print("ℹ️  Her adım orijinal input görüntüsünü kullanır")
    print("=" * 60)
    
    # Klasörleri oluştur
    create_directories()
    
    # Input klasöründeki görüntüleri bul
    input_dir = Path("input")
    test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not test_images:
        print("❌ Input klasöründe görüntü bulunamadı!")
        print("📁 Lütfen input/ klasörüne .jpg veya .png formatında evrak fotoğrafları ekleyin.")
        return
    
    print(f"📸 {len(test_images)} adet görüntü bulundu")
    
    # Her görüntü için işlem yap
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🔄 İşleniyor ({i}/{len(test_images)}): {image_path.name}")
        print("-" * 40)
        
        try:
            base_name = image_path.stem
            extension = image_path.suffix
            
            # ADIM 1: Maskeleme (ORİJİNAL INPUT KULLAN)
            print("\n1️⃣ ADIM 1: MASKELEME")
            step1_output = Path("output") / f"step1_masked_{base_name}{extension}"
            # test_step1 fonksiyonu doğrudan çağrılmak yerine, process_step1 benzeri bir fonksiyon kullanılmalı
            # Ancak mevcut kodda test_step1 fonksiyonu kullanılıyor
            test_step1()
            
            # ADIM 2: Kontrast İyileştirme (ORİJİNAL INPUT KULLAN)
            print("\n2️⃣ ADIM 2: KONTRAST İYİLEŞTİRME")
            step2_output = Path("output") / f"step2_enhanced_{base_name}{extension}"
            process_step2(str(image_path), str(step2_output))
            
            # ADIM 3: Perspektif Düzeltme (ORİJİNAL INPUT KULLAN)
            print("\n3️⃣ ADIM 3: PERSPEKTİF DÜZELTİMİ")
            step3_output = Path("output") / f"step3_final_{base_name}{extension}"
            process_step3(str(image_path), str(step3_output))
            
            print(f"\n✅ {image_path.name} başarıyla işlendi!")
            print(f"📁 Çıktılar:")
            print(f"   • {step1_output}")
            print(f"   • {step2_output}")
            print(f"   • {step3_output}")
            
            # Sonuçları karşılaştır
            compare_results(str(image_path), str(step1_output), str(step2_output), str(step3_output))
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")
            continue
    
    print("\n🎉 TÜM İŞLEMLER TAMAMLANDI!")
    print("📂 Sonuçları output/ klasöründe bulabilirsiniz:")
    print("   • step1_masked_*.jpg - Maskelenmiş evraklar")
    print("   • step2_enhanced_*.jpg - Kontrast iyileştirilmiş")
    print("   • step3_final_*.jpg - Perspektif düzeltilmiş")

def compare_results(original_path, step1_path, step2_path, step3_path):
    """Tüm adımların sonuçlarını karşılaştır"""
    # Görüntüleri yükle
    original = cv2.imread(original_path)
    step1 = cv2.imread(step1_path)
    step2 = cv2.imread(step2_path)
    step3 = cv2.imread(step3_path)
    
    # Boyutları eşitle
    height = 300
    width = int(height * original.shape[1] / original.shape[0])
    
    original_resized = cv2.resize(original, (width, height))
    step1_resized = cv2.resize(step1, (width, height))
    step2_resized = cv2.resize(step2, (width, height))
    step3_resized = cv2.resize(step3, (width, height))
    
    # Karşılaştırma görüntüsü oluştur
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(step1_resized, cv2.COLOR_BGR2RGB))
    plt.title('Adım 1: Maskeleme')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(step2_resized, cv2.COLOR_BGR2RGB))
    plt.title('Adım 2: Kontrast İyileştirme')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(step3_resized, cv2.COLOR_BGR2RGB))
    plt.title('Adım 3: Perspektif Düzeltme')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Ana menü"""
    print("📄 EVRAK TARAMA SİSTEMİ")
    print("=" * 30)
    print("ℹ️  Her adım orijinal input'u kullanır")
    print("=" * 30)
    print("1. Tüm adımları çalıştır")
    print("2. Sadece Adım 1 (Maskeleme)")
    print("3. Sadece Adım 2 (Kontrast)")
    print("4. Sadece Adım 3 (Perspektif)")
    print("5. Çıkış")
    
    while True:
        try:
            choice = input("\nSeçiminizi yapın (1-5): ").strip()
            
            if choice == "1":
                process_all_steps()
                break
            elif choice == "2":
                test_step1()
                break
            elif choice == "3":
                # Tüm input görüntüleri için Adım 2'yi çalıştır
                input_dir = Path("input")
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
                for image_path in test_images:
                    output_path = output_dir / f"step2_enhanced_{image_path.name}"
                    process_step2(str(image_path), str(output_path))
                break
            elif choice == "4":
                # Tüm input görüntüleri için Adım 3'ü çalıştır
                input_dir = Path("input")
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
                for image_path in test_images:
                    output_path = output_dir / f"step3_final_{image_path.name}"
                    process_step3(str(image_path), str(output_path))
                break
            elif choice == "5":
                print("👋 Görüşürüz!")
                break
            else:
                print("❌ Geçersiz seçim! Lütfen 1-5 arası bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n👋 İşlem iptal edildi!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    main()
