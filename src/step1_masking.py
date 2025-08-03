import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DocumentMasking:
    """
    ADIM 1: Evrakın dışında kalan bölgeyi maskeleme ile ayırma
    Sadece evraka ait görüntü bölgesini elde etme
    """
    
    def __init__(self):
        self.original_image = None
        self.mask = None
        self.masked_result = None
    
    def load_image(self, image_path):
        """Görüntüyü yükle"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        print(f"✓ Görüntü yüklendi: {image_path}")
        return self.original_image
    
    def create_document_mask(self, image):
        """Evrak için maske oluştur"""
        print("🔍 Evrak maskeleme işlemi başlıyor...")
        
        # 1. Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur (gürültü azaltma)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Kenar tespiti
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 4. Morfolojik işlemler (kenarları güçlendir)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 5. Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. En büyük konturu bul (evrak olması muhtemel)
        if not contours:
            raise ValueError("Hiç kontur bulunamadı!")
        
        # Konturları alan büyüklüğüne göre sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # En büyük konturu seç
        document_contour = contours[0]
        
        # Kontur alanı çok küçükse hata ver
        image_area = image.shape[0] * image.shape[1]
        contour_area = cv2.contourArea(document_contour)
        
        if contour_area < image_area * 0.1:  # Görüntünün %10'undan küçükse
            raise ValueError("Evrak konturu çok küçük görünüyor!")
        
        print(f"✓ Evrak konturu bulundu (Alan: {contour_area:.0f} piksel)")
        
        # 7. Maske oluştur
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [document_contour], 255)
        
        self.mask = mask
        return mask, document_contour
    
    def apply_mask(self, image, mask):
        """Maskeyi uygula - sadece evrak bölgesini koru"""
        print("🎭 Maske uygulanıyor...")
        
        # Maskeyi 3 kanala genişlet (BGR için)
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Maskeyi normalize et (0-1 arası)
        mask_normalized = mask_3channel.astype(float) / 255
        
        # Maskeyi uygula
        masked_image = image.astype(float) * mask_normalized
        
        # Arka planı beyaz yap
        background = np.ones_like(image, dtype=float) * 255
        inverse_mask = 1 - mask_normalized
        
        result = masked_image + (background * inverse_mask)
        result = result.astype(np.uint8)
        
        self.masked_result = result
        print("✓ Maskeleme tamamlandı")
        return result
    
    def process_step1(self, input_path, output_path):
        """Adım 1'i tamamen gerçekleştir - SADECE ORİJİNAL INPUT KULLAN"""
        print("=" * 50)
        print("ADIM 1: EVRAK MASKELEME")
        print("=" * 50)
        
        # ORİJİNAL görüntüyü yükle
        image = self.load_image(input_path)
        
        # Maske oluştur
        mask, contour = self.create_document_mask(image)
        
        # Maskeyi uygula
        result = self.apply_mask(image, mask)
        
        # Sonucu kaydet
        cv2.imwrite(output_path, result)
        print(f"✓ Sonuç kaydedildi: {output_path}")
        
        # Görselleştir
        self.visualize_step1(image, mask, contour, result)
        
        return result
    
    def visualize_step1(self, original, mask, contour, result):
        """Adım 1 sonuçlarını görselleştir"""
        plt.figure(figsize=(15, 5))
        
        # Orijinal görüntü
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('1. Orijinal Görüntü')
        plt.axis('off')
        
        # Kontur tespiti
        plt.subplot(1, 4, 2)
        contour_img = original.copy()
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title('2. Tespit Edilen Kontur')
        plt.axis('off')
        
        # Maske
        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('3. Oluşturulan Maske')
        plt.axis('off')
        
        # Maskelenmiş sonuç
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('4. Maskelenmiş Evrak')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Test fonksiyonu
def test_step1():
    masker = DocumentMasking()
    
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Test görüntülerini bul
    test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not test_images:
        print("❌ Input klasöründe görüntü bulunamadı!")
        return
    
    for image_path in test_images:
        output_path = output_dir / f"step1_masked_{image_path.name}"
        try:
            masker.process_step1(str(image_path), str(output_path))
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    test_step1()
