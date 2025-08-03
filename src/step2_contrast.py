"""
İŞLEM 2: Kontrast İyileştirme
Evrakın kontrastını iyileştirilecek. Gerekirse zemin rengi tam beyaz yapılacak.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def enhance_contrast(image):
    """
    Evrakın kontrastını iyileştirme ve zemin rengini beyazlatma
    
    Args:
        image: Orijinal görüntü (BGR formatında)
        
    Returns:
        enhanced_image: Kontrast iyileştirilmiş görüntü
        steps: Adım adım sonuçlar
    """
    print("🌟 Kontrast iyileştirme işlemi başlıyor...")
    
    # Adım adım sonuçları sakla
    steps = {}
    
    # Orijinal görüntü
    steps['1. Orijinal'] = image.copy()
    
    # 1. CLAHE ile kontrast iyileştirme
    print("📈 CLAHE kontrast iyileştirme uygulanıyor...")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge((l, a, b))
    clahe_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    steps['2. CLAHE Kontrast'] = clahe_result
    
    # 2. Parlaklık ve kontrast ayarlama
    print("💡 Parlaklık ve kontrast ayarlanıyor...")
    alpha = 1.2  # Kontrast faktörü
    beta = 10    # Parlaklık değeri
    
    brightness_result = cv2.convertScaleAbs(clahe_result, alpha=alpha, beta=beta)
    steps['3. Parlaklık Ayarı'] = brightness_result
    
    # 3. Gölge giderme
    print("🌫️ Gölge giderme uygulanıyor...")
    rgb = cv2.cvtColor(brightness_result, cv2.COLOR_BGR2RGB)
    
    dilated_img = cv2.dilate(rgb, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    
    diff_img = 255 - cv2.absdiff(rgb, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    shadow_result = cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR)
    steps['4. Gölge Giderme'] = shadow_result
    
    # 4. Arka planı beyazlatma
    print("⚪ Arka plan beyazlatılıyor...")
    gray = cv2.cvtColor(shadow_result, cv2.COLOR_BGR2GRAY)
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    final_result = shadow_result.copy()
    final_result[binary == 255] = [255, 255, 255]
    steps['5. Beyaz Arka Plan'] = final_result
    
    print("✓ Kontrast iyileştirme tamamlandı")
    
    return final_result, steps

def visualize_step2(steps):
    """Adım 2 sonuçlarını görselleştir"""
    plt.figure(figsize=(15, 10))
    
    for i, (title, img) in enumerate(steps.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_step2(input_path, output_path):
    """Adım 2'yi gerçekleştir"""
    print("=" * 50)
    print("ADIM 2: KONTRAST İYİLEŞTİRME")
    print("=" * 50)
    
    # Görüntüyü yükle
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Görüntü yüklenemedi: {input_path}")
    
    print(f"✓ Görüntü yüklendi: {input_path}")
    
    # Kontrast iyileştirme
    enhanced, steps = enhance_contrast(image)
    
    # Sonucu kaydet
    cv2.imwrite(output_path, enhanced)
    print(f"✓ Sonuç kaydedildi: {output_path}")
    
    # Görselleştir
    visualize_step2(steps)
    
    return enhanced

# Test fonksiyonu
def test_step2():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Test görüntülerini bul
    test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not test_images:
        print("❌ Input klasöründe görüntü bulunamadı!")
        return
    
    for image_path in test_images:
        output_path = output_dir / f"step2_enhanced_{image_path.name}"
        try:
            process_step2(str(image_path), str(output_path))
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    test_step2()
