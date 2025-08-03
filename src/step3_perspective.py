"""
İŞLEM 3: Perspektif Düzeltme
Evrakta oluşan sağ-sol ve alt-üst köşe noktaları (ya da sınır çizgileri) belirlenecek.
Herhangi bir trapezoid (yamuk) şekil varsa düzeltilecek.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_document_contour(image):
    """Belge konturunu daha hassas şekilde bul"""
    print("🔍 Belge konturu aranıyor...")
    
    # 1. Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Adaptive threshold (daha iyi kenar tespiti için)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 4. Morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. Kenar tespiti
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # 6. Kenarları güçlendir
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 7. Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️ Hiç kontur bulunamadı!")
        return None, edges
    
    # 8. Konturları filtrele (alan ve şekil kriterlerine göre)
    image_area = image.shape[0] * image.shape[1]
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Kontur alanı görüntünün %10'undan büyük ve %90'ından küçük olmalı
        if image_area * 0.1 < area < image_area * 0.9:
            # Konturun dikdörtgen benzeri olup olmadığını kontrol et
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 4-8 köşeli şekilleri kabul et
            if 4 <= len(approx) <= 8:
                valid_contours.append((contour, area))
    
    if not valid_contours:
        print("⚠️ Geçerli belge konturu bulunamadı!")
        # En büyük konturu al
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour, edges
    
    # En büyük geçerli konturu seç
    document_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    print(f"✓ Belge konturu bulundu (Alan: {cv2.contourArea(document_contour):.0f} piksel)")
    
    return document_contour, edges

def refine_corner_points(image, corners):
    """Köşe noktalarını hassas şekilde iyileştir"""
    print("🔍 Köşe noktaları hassas şekilde iyileştiriliyor...")
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Köşe noktalarını iyileştir
    refined_corners = []
    
    for corner in corners:
        # Köşe etrafında küçük bir bölge al
        x, y = corner.astype(int)
        window_size = 11  # Pencere boyutu
        half_size = window_size // 2
        
        # Görüntü sınırlarını kontrol et
        x_min = max(0, x - half_size)
        y_min = max(0, y - half_size)
        x_max = min(gray.shape[1], x + half_size + 1)
        y_max = min(gray.shape[0], y + half_size + 1)
        
        # Bölgeyi al
        window = gray[y_min:y_max, x_min:x_max]
        
        if window.size == 0:
            # Pencere boşsa, orijinal köşeyi kullan
            refined_corners.append(corner)
            continue
        
        # Harris köşe tespiti
        window_float = np.float32(window)
        dst = cv2.cornerHarris(window_float, 2, 3, 0.04)
        
        # Eşik değeri
        threshold = 0.01 * dst.max()
        
        # En güçlü köşeyi bul
        if np.any(dst > threshold):
            y_local, x_local = np.unravel_index(dst.argmax(), dst.shape)
            refined_x = x_min + x_local
            refined_y = y_min + y_local
            refined_corners.append(np.array([refined_x, refined_y], dtype=np.float32))
        else:
            # Köşe bulunamazsa, orijinal köşeyi kullan
            refined_corners.append(corner)
    
    refined_corners = np.array(refined_corners, dtype=np.float32)
    print("✓ Köşe noktaları iyileştirildi")
    
    return refined_corners

def get_corner_points(contour):
    """Konturdan 4 köşe noktasını çıkar"""
    print("📐 Köşe noktaları hesaplanıyor...")
    
    # Kontur yaklaşımı ile 4 köşe bulmaya çalış
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Farklı epsilon değerleri dene
    for eps_factor in [0.01, 0.015, 0.025, 0.03, 0.04, 0.05]:
        epsilon = eps_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            print(f"✓ 4 köşe bulundu (epsilon: {eps_factor})")
            return approx.reshape(4, 2).astype(np.float32)
    
    print(f"⚠️ 4 köşe bulunamadı ({len(approx)} köşe bulundu), en dış köşeler seçiliyor...")
    
    # En dış 4 köşeyi bul
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def correct_perspective(image):
    """
    Evrakın perspektifini düzeltme - İYİLEŞTİRİLMİŞ VERSİYON
    
    Args:
        image: Orijinal görüntü (BGR formatında)
        
    Returns:
        corrected_image: Perspektifi düzeltilmiş görüntü
        corners: Tespit edilen köşe noktaları
    """
    print("🔄 Perspektif düzeltme işlemi başlıyor...")
    
    # 1. Belge konturunu bul
    document_contour, edges = find_document_contour(image)
    
    if document_contour is None:
        print("❌ Belge konturu bulunamadı! Orijinal görüntü döndürülüyor.")
        return image, np.array([[0, 0], [image.shape[1], 0], 
                               [image.shape[1], image.shape[0]], [0, image.shape[0]]])
    
    # 2. Köşe noktalarını al
    corners = get_corner_points(document_contour)
    
    # 3. Köşe noktalarını hassas şekilde iyileştir
    refined_corners = refine_corner_points(image, corners)
    
    # 4. Köşeleri sırala
    ordered_corners = order_points(refined_corners)
    
    # 5. Hedef boyutları hesapla
    width, height = calculate_dimensions(ordered_corners)
    
    # Minimum boyut kontrolü
    if width < 100 or height < 100:
        print("⚠️ Hesaplanan boyutlar çok küçük, orijinal görüntü döndürülüyor...")
        return image, corners
    
    # 6. Perspektif dönüşümü uygula
    dst_points = np.array([
        [0, 0],                    # sol-üst
        [width - 1, 0],            # sağ-üst
        [width - 1, height - 1],   # sağ-alt
        [0, height - 1]            # sol-alt
    ], dtype="float32")
    
    # Perspektif dönüşüm matrisi
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    
    # Dönüşümü uygula
    warped = cv2.warpPerspective(image, transform_matrix, (width, height))
    
    # 7. Son düzeltme - hafif döndürme düzeltmesi
    # Görüntüyü analiz et ve gerekirse hafif döndür
    warped = fine_tune_rotation(warped)
    
    print("✓ Perspektif düzeltme tamamlandı")
    
    return warped, ordered_corners

def fine_tune_rotation(image):
    """Görüntüyü analiz ederek hafif döndürme düzeltmesi yap"""
    print("🔄 İnce döndürme ayarı yapılıyor...")
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kenar tespiti
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform ile çizgileri bul
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        print("⚠️ Düzeltme için çizgi bulunamadı, orijinal görüntü döndürülüyor...")
        return image
    
    # Yatay çizgilerin açılarını hesapla
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Yatay çizgiler için (y değerleri yakın)
        if abs(y2 - y1) < 20:  # Yatay kabul edilecek eşik
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
    
    # Açı yoksa, orijinal görüntüyü döndür
    if not angles:
        print("⚠️ Yatay çizgi bulunamadı, orijinal görüntü döndürülüyor...")
        return image
    
    # Medyan açıyı hesapla (aykırı değerlerden etkilenmemek için)
    median_angle = np.median(angles)
    
    # Çok küçük açıları düzeltme (1 dereceden küçük)
    if abs(median_angle) < 1.0:
        print(f"✓ Açı çok küçük ({median_angle:.2f}°), düzeltme gerekmiyor")
        return image
    
    # Görüntüyü döndür
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    print(f"✓ Görüntü {median_angle:.2f}° döndürüldü")
    
    return rotated

def order_points(pts):
    """Köşe noktalarını sırala: sol-üst, sağ-üst, sağ-alt, sol-alt"""
    print("📐 Köşe noktaları sıralanıyor...")
    
    # Numpy dizisine dönüştür
    pts = np.array(pts, dtype="float32")
    
    # Sıralama için boş dizi
    rect = np.zeros((4, 2), dtype="float32")
    
    # Toplam koordinatları hesapla
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # sol-üst (en küçük toplam)
    rect[2] = pts[np.argmax(s)]  # sağ-alt (en büyük toplam)
    
    # Koordinat farklarını hesapla
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sağ-üst (en küçük fark)
    rect[3] = pts[np.argmax(diff)]  # sol-alt (en büyük fark)
    
    print("✓ Köşe noktaları sıralandı")
    return rect

def calculate_dimensions(ordered_points):
    """Düzeltilmiş görüntünün boyutlarını hesapla"""
    (tl, tr, br, bl) = ordered_points
    
    # Genişlik hesapla
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Yükseklik hesapla
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    print(f"✓ Hedef boyutlar: {maxWidth}x{maxHeight}")
    return maxWidth, maxHeight

def visualize_perspective(original, corners, warped):
    """Perspektif düzeltme işlemini görselleştir"""
    plt.figure(figsize=(15, 5))
    
    # 1. Tespit Edilen Köşeler
    plt.subplot(1, 3, 1)
    img_with_corners = original.copy()
    
    # Köşe noktalarını çiz
    for i, corner in enumerate(corners):
        cv2.circle(img_with_corners, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(img_with_corners, str(i+1), tuple(corner.astype(int) + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Köşeleri birleştir
    cv2.polylines(img_with_corners, [corners.astype(int)], True, (255, 0, 0), 3)
    
    plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    plt.title('1. Tespit Edilen Köşeler')
    plt.axis('off')
    
    # 2. Önce → Sonra
    plt.subplot(1, 3, 2)
    
    # Sabit boyutlara yeniden boyutlandır
    target_height = 300
    
    # Orijinal görüntüyü yeniden boyutlandır
    orig_aspect = original.shape[1] / original.shape[0]
    orig_width = int(target_height * orig_aspect)
    resized_original = cv2.resize(original, (orig_width, target_height))
    
    # Düzeltilmiş görüntüyü yeniden boyutlandır
    warped_aspect = warped.shape[1] / warped.shape[0]
    warped_width = int(target_height * warped_aspect)
    resized_warped = cv2.resize(warped, (warped_width, target_height))
    
    # Aynı yükseklikte olduklarından emin ol
    if resized_original.shape[0] != resized_warped.shape[0]:
        min_height = min(resized_original.shape[0], resized_warped.shape[0])
        resized_original = cv2.resize(resized_original, (int(orig_width * min_height / resized_original.shape[0]), min_height))
        resized_warped = cv2.resize(resized_warped, (int(warped_width * min_height / resized_warped.shape[0]), min_height))
    
    # Ayırıcı oluştur
    separator = np.ones((resized_original.shape[0], 20, 3), dtype=np.uint8) * 255
    
    # Görüntüleri birleştir
    comparison = np.hstack([resized_original, separator, resized_warped])
    
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('2. Önce → Sonra')
    plt.axis('off')
    
    # 3. Düzeltilmiş Evrak
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('3. Düzeltilmiş Evrak')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_step3(input_path, output_path):
    """Adım 3'ü gerçekleştir"""
    print("=" * 50)
    print("ADIM 3: PERSPEKTİF DÜZELTME")
    print("=" * 50)
    
    # Görüntüyü yükle
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Görüntü yüklenemedi: {input_path}")
    
    print(f"✓ Görüntü yüklendi: {input_path}")
    
    # Perspektif düzeltme
    corrected, corners = correct_perspective(image)
    
    # Sonucu kaydet
    cv2.imwrite(output_path, corrected)
    print(f"✓ Sonuç kaydedildi: {output_path}")
    
    # Görselleştir
    visualize_perspective(image, corners, corrected)
    
    return corrected

# Test fonksiyonu
def test_step3():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Test görüntülerini bul
    test_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not test_images:
        print("❌ Input klasöründe görüntü bulunamadı!")
        return
    
    for image_path in test_images:
        output_path = output_dir / f"step3_final_{image_path.name}"
        try:
            process_step3(str(image_path), str(output_path))
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    test_step3()
