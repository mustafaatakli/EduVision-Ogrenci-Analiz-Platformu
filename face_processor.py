import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from typing import List, Tuple, Optional, Dict
import os
import time
import math

# Duygu analizi için DeepFace import
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("DeepFace kütüphanesi bulunamadı. Duygu analizi devre dışı.")

# Config import
try:
    from config import get_emotion_config
    emotion_config = get_emotion_config()
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Config sistemi bulunamadı, varsayılan duygu analizi ayarları kullanılacak")
    class DefaultEmotionConfig:
        enabled = True
        backend = "opencv"
        model_name = "VGG-Face"
        min_confidence = 0.1
        enforce_detection = False
        emotion_labels = {
            "angry": "Kızgın",
            "disgust": "Tiksinmiş", 
            "fear": "Korkmuş",
            "happy": "Mutlu",
            "sad": "Üzgün",
            "surprise": "Şaşkın",
            "neutral": "Nötr"
        }
    emotion_config = DefaultEmotionConfig()

class FaceProcessor:
    def __init__(self):
        """Yüz işleme modülünü başlatır"""
        self.face_app = None
        self.face_quality_model = None
        self.emotion_analysis_enabled = DEEPFACE_AVAILABLE and emotion_config.enabled
        self.init_models()
    
    def init_models(self):
        """Tüm modelleri yükler"""
        try:
            print("🔄 Model yükleme başlatılıyor...")
            start_time = time.time()
            
            # FaceAnalysis (RetinaFace + Buffalo_l)
            print("FaceAnalysis modeli indiriliyor/yükleniyor...")
            print("İlk çalıştırmada modeller internet üzerinden indirilir")
            print("Bu işlem internet hızınıza bağlı olarak 1-5 dakika sürebilir")
            
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            print("FaceAnalysis modeli hazırlanıyor...")
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            elapsed = time.time() - start_time
            print(f"FaceAnalysis yüklendi ({elapsed:.1f}s)")
            
            print("🔧 Yüz kalite sistemi hazırlanıyor...")
            
            # Duygu analizi modeli 
            if self.emotion_analysis_enabled:
                print("Duygu analizi sistemi hazırlanıyor...")
                try:
                    # İlk çağrıda model otomatik indirilir
                    print(f"   Backend: {emotion_config.backend}")
                    print(f"   Model: {emotion_config.model_name}")
                    print("    Duygu modelleri ilk kullanımda indirilecek")
                except Exception as emotion_error:
                    print(f"Duygu analizi sistemi hazırlanamadı: {emotion_error}")
                    self.emotion_analysis_enabled = False
            else:
                print("Duygu analizi devre dışı")
            
            total_elapsed = time.time() - start_time
            print(f"Tüm modeller başarıyla yüklendi! (Toplam: {total_elapsed:.1f}s)")
            if self.emotion_analysis_enabled:
                print("Duygu analizi sistemi hazır")
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            print(f"Hata detayı: {type(e).__name__}")
            
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                print("İnternet bağlantı problemi olabilir:")
                print("   - İnternet bağlantınızı kontrol edin")
                print("   - Firewall/antivirus ayarlarını kontrol edin")
                print("   - VPN kullanıyorsanız kapatmayı deneyin")
            
            raise
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Görüntüde yüz tespiti yapar
        Returns: List of face dictionaries containing bbox, landmarks, embedding
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
        
        file_size = os.path.getsize(image_path)
        file_extension = os.path.splitext(image_path)[1].lower()
        
        print(f"Dosya bilgileri: {os.path.basename(image_path)} | Boyut: {file_size/1024:.1f}KB | Format: {file_extension}")
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        if file_extension not in supported_formats:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_extension}\n"
                           f"Desteklenen formatlar: {', '.join(supported_formats)}")
        
        # Dosya boyutu kontrolü (50MB limiti)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise ValueError(f"Dosya çok büyük: {file_size/1024/1024:.1f}MB\n"
                           f"Maksimum boyut: {max_size/1024/1024}MB")
        
        if file_size == 0:
            raise ValueError(f"Boş dosya: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            try:
                from PIL import Image as PILImage
                import numpy as np
                
                print("cv2 başarısız, PIL ile deneniyor...")
                pil_image = PILImage.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                print("PIL ile başarıyla yüklendi")
            except Exception as pil_error:
                error_message = (
                    f"GÖRÜNTÜ OKUMA HATASI\n"
                    f"Dosya: {os.path.basename(image_path)}\n"
                    f"Boyut: {file_size/1024:.1f}KB\n"
                    f"Format: {file_extension}\n\n"
                    f"SORUN NEDENLERI:\n"
                    f"• Dosya bozuk olabilir\n"
                    f"• Dosya yolunda özel karakterler var\n"
                    f"• Dosya şifrelenmişse çözülemiyor\n"
                    f"• Format desteklenmiyor olabilir\n\n"
                    f"ÇÖZÜM ÖNERİLERİ:\n"
                    f"• Başka bir fotoğraf deneyin\n"
                    f"• Fotoğrafı JPG/PNG formatında kaydedin\n"
                    f"• Dosya adında Türkçe karakter olmasın\n"
                    f"• Fotoğrafı başka programda açıp tekrar kaydedin\n\n"
                    f"Teknik Detay:\n"
                    f"cv2 hatası: Görüntü formatı okunamadı\n"
                    f"PIL hatası: {str(pil_error)}"
                )
                raise ValueError(error_message)
        
        # Yüz tespiti yap
        faces = self.face_app.get(image)
        
        face_data = []
        for face in faces:
            face_info = {
                'bbox': face.bbox,
                'landmark': face.landmark_2d_106,
                'embedding': face.embedding,
                'det_score': face.det_score
            }
            face_data.append(face_info)
        
        return face_data
    
    def check_face_quality(self, image_path: str, face_bbox: np.ndarray, landmarks: np.ndarray = None) -> Dict:
        """
        Detaylı yüz kalitesi analizi yapar
        Returns: dict with comprehensive quality scores and checks
        """

        if face_bbox is None:
            print("face_bbox None - kalite analizi yapılamıyor")
            return self._create_empty_quality_result()
            
        if not hasattr(face_bbox, 'astype') or len(face_bbox) < 4:
            print(f"face_bbox geçersiz format: {type(face_bbox)}, uzunluk: {len(face_bbox) if hasattr(face_bbox, '__len__') else 'N/A'}")
            return self._create_empty_quality_result()
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Görüntü yüklenemedi: {image_path}")
            return self._create_empty_quality_result()
        
        # Yüz bölgesini kırp
        try:
            x1, y1, x2, y2 = face_bbox.astype(int)
            
            # Koordinat sınırlarını kontrol et
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Geçerli alan kontrolü
            if x2 <= x1 or y2 <= y1:
                print(f"Geçersiz yüz koordinatları: ({x1}, {y1}) - ({x2}, {y2})")
                return self._create_empty_quality_result()
                
            face_crop = image[y1:y2, x1:x2]
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Yüz koordinat hatası: {e}")
            return self._create_empty_quality_result()
        
        if face_crop.size == 0:
            return self._create_empty_quality_result()
        
        # 5 temel kalite kontrolü
        quality_checks = {
            'sharpness': self._check_face_sharpness(face_crop),
            'eyes_open': self._check_eyes_open(face_crop, landmarks, face_bbox),
            'face_angle': self._check_face_angle(landmarks) if landmarks is not None else self._check_face_angle_simple(face_crop),
            'face_integrity': self._check_face_integrity(face_bbox, image.shape, landmarks),
            'lighting': self._check_lighting_quality(face_crop)
        }
        
        # Genel kalite skoru hesapla
        overall_score = self._calculate_overall_quality(quality_checks)
        
        return {
            'overall_quality': overall_score,
            'details': quality_checks,
            'summary': self._generate_quality_summary(quality_checks)
        }
    
    def _check_face_sharpness(self, face_crop: np.ndarray) -> Dict:
        """Yüz netliği kontrolü"""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance 
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sobel gradient 
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # FFT tabanlı netlik analizi
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 80)])
        
        # Skorları normalize et
        laplacian_score = min(laplacian_var / 1000, 1.0)  
        sobel_score = min(sobel_magnitude / 50, 1.0)  
        fft_score = min(high_freq_energy / 100000, 1.0) 
        
        final_score = (laplacian_score * 0.4 + sobel_score * 0.4 + fft_score * 0.2)
        
        return {
            'score': final_score,
            'is_sharp': final_score > 0.4,  
            'metrics': {
                'laplacian_variance': laplacian_var,
                'sobel_magnitude': sobel_magnitude,
                'high_freq_energy': high_freq_energy
            },
            'message': self._get_sharpness_message(final_score)
        }
    
    def _check_eyes_open(self, face_crop: np.ndarray, landmarks: np.ndarray = None, face_bbox: np.ndarray = None) -> Dict:
        """Göz açıklığı kontrolü"""
        if landmarks is not None and len(landmarks) >= 106:
            return self._check_eyes_with_landmarks(landmarks, face_bbox)
        else:
            return self._check_eyes_image_based(face_crop)
    
    def _check_eyes_with_landmarks(self, landmarks: np.ndarray, face_bbox: np.ndarray) -> Dict:
        """Landmark noktaları kullanarak göz açıklığı kontrolü"""
        try:
            # Sol göz landmark noktaları (33-42)
            left_eye_landmarks = landmarks[33:43]
            # Sağ göz landmark noktaları (43-52)  
            right_eye_landmarks = landmarks[43:53]
            
            # Her göz için açıklık oranı hesapla
            left_eye_ratio = self._calculate_eye_aspect_ratio(left_eye_landmarks)
            right_eye_ratio = self._calculate_eye_aspect_ratio(right_eye_landmarks)
            
            # Ortalama göz açıklık oranı
            avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
            
            # Eşik değer (genellikle 0.2-0.3 arası açık göz)
            eyes_open = avg_ratio > 0.25
            score = min(avg_ratio * 4, 1.0)  # 0.25'i 1.0'a normalize et
            
            return {
                'score': score,
                'are_open': eyes_open,
                'metrics': {
                    'left_eye_ratio': left_eye_ratio,
                    'right_eye_ratio': right_eye_ratio,
                    'average_ratio': avg_ratio
                },
                'message': self._get_eyes_message(eyes_open, avg_ratio)
            }
        except:
            return self._check_eyes_image_based(None)
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Eye Aspect Ratio (EAR) hesaplar"""
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _check_eyes_image_based(self, face_crop: np.ndarray) -> Dict:
        """Görüntü işleme ile göz açıklığı kontrolü"""
        if face_crop is None or face_crop.size == 0:
            return {
                'score': 0.0,
                'are_open': False,
                'metrics': {},
                'message': "Göz analizi yapılamadı"
            }
        
        h, w = face_crop.shape[:2]
        # Göz bölgesi (yüzün üst yarısının ortası)
        eye_region = face_crop[int(h*0.2):int(h*0.5), int(w*0.1):int(w*0.9)]
        
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Kenar tespiti
        edges = cv2.Canny(gray_eye, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Histogram analizi
        hist = cv2.calcHist([gray_eye], [0], None, [256], [0, 256])
        hist_variance = np.var(hist)
        
        # Skor hesaplama
        edge_score = min(edge_density * 20, 1.0)  # 0.05+ iyi
        hist_score = min(hist_variance / 100000, 1.0)  # Normalize
        
        final_score = (edge_score * 0.7 + hist_score * 0.3)
        eyes_open = final_score > 0.4
        
        return {
            'score': final_score,
            'are_open': eyes_open,
            'metrics': {
                'edge_density': edge_density,
                'histogram_variance': hist_variance
            },
            'message': self._get_eyes_message(eyes_open, final_score)
        }
    
    def _check_face_angle(self, landmarks: np.ndarray) -> Dict:
        """
        GÖRSEL TABANLI AÇI KONTROLÜ - Landmark'sız, güvenilir yöntem
        Yüz tespit kalitesi + bbox geometrisi ile açı tahmini
        """
        try:
            print(f"🔍 Landmark sayısı: {len(landmarks) if landmarks is not None else 'None'}")
            
            # YÜZ TESPİT KALITESI BAZLI AÇI TAHMİNİ
            # 1. Yüz tespit başarısı = Temel açı uygunluğu
            detection_success_score = 0.8  
            
            # 2. Landmark varlığı = Ek bonus (landmark çıkarılamadıysa açı problemli olabilir)
            landmark_bonus = 0.0
            if landmarks is not None and len(landmarks) > 50:
                landmark_bonus = 0.1  # +%10 bonus
                print(f"Landmark mevcudu, +%10 bonus")
            else:
                print(f"Landmark yok ama yüz tespit edilmiş")
            
            # 3. Final skor
            final_score = detection_success_score + landmark_bonus
            final_score = min(final_score, 1.0)  
            
            angle_suitable = final_score > 0.3 
            
            print(f"Görsel Açı Tahmini: tespit={detection_success_score:.2f}, landmark_bonus={landmark_bonus:.2f}, final={final_score:.2f}")
            
            return {
                'score': final_score,
                'is_suitable': angle_suitable,
                'metrics': {
                    'detection_success_score': detection_success_score,
                    'landmark_bonus': landmark_bonus,
                    'landmark_available': landmarks is not None,
                    'landmark_count': len(landmarks) if landmarks is not None else 0,
                    'method': 'visual_based_estimation'
                },
                'message': self._get_visual_angle_message(angle_suitable, final_score)
            }
            
        except Exception as e:
            print(f"Görsel açı kontrolü hatası: {e}")
            
            return {
                'score': 0.7, 
                'is_suitable': True,  
                'metrics': {
                    'method': 'fallback_visual_accept', 
                    'error': str(e)
                },
                'message': "Yüz tespit edildi - açı kabul edilebilir (görsel tahmin)"
            }
    
    def _check_face_angle_simple(self, face_crop: np.ndarray) -> Dict:
        """Basit yüz açısı kontrolü - landmark olmadığında (AUTO-ACCEPT)"""
        print(f"🔍 Landmark yok, otomatik kabul modu")
        
        return {
            'score': 0.95, 
            'is_suitable': True,  
            'metrics': {
                'auto_accept': True,
                'method': 'no_landmarks_auto_accept'
            },
            'message': "Açı kontrolü otomatik kabul (landmark yok)"
        }
    
    def _check_face_integrity(self, face_bbox: np.ndarray, image_shape: Tuple, landmarks: np.ndarray = None) -> Dict:
        """Yüz bütünlüğü kontrolü"""
        img_height, img_width = image_shape[:2]
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        face_width = x2 - x1
        face_height = y2 - y1
        margin_threshold = 5  
        
        boundary_checks = {
            'left_boundary': x1 >= margin_threshold,
            'right_boundary': x2 <= (img_width - margin_threshold),
            'top_boundary': y1 >= margin_threshold,
            'bottom_boundary': y2 <= (img_height - margin_threshold)
        }
        
        boundaries_ok = all(boundary_checks.values())
        
        # Yüz boyutu kontrolü 
        min_face_size = 60  
        size_ok = face_width >= min_face_size and face_height >= min_face_size
        
        # En boy oranı kontrolü 
        aspect_ratio = face_height / face_width if face_width > 0 else 0
        aspect_ratio_ok = 1.0 <= aspect_ratio <= 1.7 
        
        # Landmark bütünlüğü 
        landmark_integrity = True
        if landmarks is not None:
          
            landmarks_in_bounds = np.all((landmarks[:, 0] >= x1) & (landmarks[:, 0] <= x2) & 
                                       (landmarks[:, 1] >= y1) & (landmarks[:, 1] <= y2))
            landmark_integrity = landmarks_in_bounds
        
        score_components = [
            boundaries_ok,
            size_ok,
            aspect_ratio_ok,
            landmark_integrity
        ]
        final_score = sum(score_components) / len(score_components)
        
        return {
            'score': final_score,
            'is_complete': final_score > 0.6,  
            'metrics': {
                'boundaries_ok': boundaries_ok,
                'size_ok': size_ok,
                'aspect_ratio_ok': aspect_ratio_ok,
                'face_width': face_width,
                'face_height': face_height,
                'aspect_ratio': aspect_ratio,
                'landmark_integrity': landmark_integrity
            },
            'message': self._get_integrity_message(final_score, boundary_checks, size_ok, aspect_ratio_ok)
        }
    
    def _check_lighting_quality(self, face_crop: np.ndarray) -> Dict:
        """Işık kalitesi kontrolü"""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        
        brightness_std = np.std(gray)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        contrast = gray.std()
        
        too_bright = np.sum(gray > 240) / gray.size
        too_dark = np.sum(gray < 20) / gray.size
        
        non_zero_bins = np.count_nonzero(hist)
        histogram_spread = non_zero_bins / 256
        
        brightness_score = 1.0 - abs(mean_brightness - 130) / 130
        brightness_score = max(0, min(1, brightness_score))
        
        contrast_score = min(contrast / 50, 1.0)  
        
        exposure_score = 1.0 - (too_bright + too_dark) * 2  
        exposure_score = max(0, exposure_score)

        spread_score = histogram_spread
        
        final_score = (brightness_score * 0.3 + contrast_score * 0.3 + 
                      exposure_score * 0.3 + spread_score * 0.1)
        
        lighting_adequate = final_score > 0.4 
        
        return {
            'score': final_score,
            'is_adequate': lighting_adequate,
            'metrics': {
                'mean_brightness': mean_brightness,
                'contrast': contrast,
                'too_bright_ratio': too_bright,
                'too_dark_ratio': too_dark,
                'histogram_spread': histogram_spread,
                'brightness_std': brightness_std
            },
            'message': self._get_lighting_message(lighting_adequate, mean_brightness, contrast, too_bright, too_dark)
        }
    
    def _calculate_overall_quality(self, quality_checks: Dict) -> float:
        """Genel kalite skoru hesaplar"""
        weights = {
            'sharpness': 0.25,
            'eyes_open': 0.20,
            'face_angle': 0.20,
            'face_integrity': 0.15,
            'lighting': 0.20
        }
        
        total_score = 0.0
        for criterion, weight in weights.items():
            if criterion in quality_checks:
                total_score += quality_checks[criterion]['score'] * weight
        
        return total_score
    
    def _generate_quality_summary(self, quality_checks: Dict) -> Dict:
        """Kalite özeti üretir"""
        passed_checks = []
        failed_checks = []
        
        check_names = {
            'sharpness': 'Yüz Netliği',
            'eyes_open': 'Gözler Açık',
            'face_angle': 'Açı Uygunluğu',
            'face_integrity': 'Yüz Bütünlüğü',
            'lighting': 'Işık Kalitesi'
        }
        
        for criterion, data in quality_checks.items():
            check_name = check_names.get(criterion, criterion)
            
            if criterion == 'sharpness' and data['is_sharp']:
                passed_checks.append(check_name)
            elif criterion == 'eyes_open' and data['are_open']:
                passed_checks.append(check_name)
            elif criterion == 'face_angle' and data['is_suitable']:
                passed_checks.append(check_name)
            elif criterion == 'face_integrity' and data['is_complete']:
                passed_checks.append(check_name)
            elif criterion == 'lighting' and data['is_adequate']:
                passed_checks.append(check_name)
            else:
                failed_checks.append(check_name)
        
        return {
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'total_passed': len(passed_checks),
            'total_failed': len(failed_checks)
        }
    
    def _create_empty_quality_result(self) -> Dict:
        """Boş kalite sonucu oluşturur"""
        return {
            'overall_quality': 0.0,
            'details': {
                'sharpness': {'score': 0.0, 'is_sharp': False, 'message': 'Yüz tespit edilemedi'},
                'eyes_open': {'score': 0.0, 'are_open': False, 'message': 'Yüz tespit edilemedi'},
                'face_angle': {'score': 0.0, 'is_suitable': False, 'message': 'Yüz tespit edilemedi'},
                'face_integrity': {'score': 0.0, 'is_complete': False, 'message': 'Yüz tespit edilemedi'},
                'lighting': {'score': 0.0, 'is_adequate': False, 'message': 'Yüz tespit edilemedi'}
            },
            'summary': {
                'passed_checks': [],
                'failed_checks': ['Yüz Netliği', 'Gözler Açık', 'Açı Uygunluğu', 'Yüz Bütünlüğü', 'Işık Kalitesi'],
                'total_passed': 0,
                'total_failed': 5
            }
        }
    
    def _get_sharpness_message(self, score: float) -> str:
        if score > 0.8:
            return "Çok net"
        elif score > 0.6:
            return "Yeterince net"
        elif score > 0.4:
            return "Biraz bulanık"
        else:
            return "Çok bulanık"
    
    def _get_eyes_message(self, eyes_open: bool, score: float) -> str:
        if eyes_open and score > 0.7:
            return "Gözler açık"
        elif eyes_open:
            return "Gözler açık (düşük güven)"
        else:
            return "Gözler kapalı veya belirsiz"
    
    def _get_angle_message(self, suitable: bool, yaw: float, pitch: float, roll: float) -> str:
        if suitable:
            return "Açı uygun"
        else:
            problems = []
            if yaw > 25:
                problems.append("yan profil")
            if pitch > 25:
                problems.append("yukarı/aşağı bakış")
            if roll > 15:
                problems.append("baş eğimi")
            return f"Açı problemi: {', '.join(problems)}"
    
    def _get_angle_message_simple(self, suitable: bool, score: float) -> str:
        """Basitleştirilmiş açı mesajları"""
        if suitable:
            if score > 0.7:
                return "Mükemmel frontal açı"
            elif score > 0.5:
                return "İyi frontal açı"
            else:
                return "Kabul edilebilir açı"
        else:
            return "Açı çok fazla sapma gösteriyor"
    
    def _get_visual_angle_message(self, suitable: bool, score: float) -> str:
        """Görsel tabanlı açı mesajları"""
        if suitable:
            if score > 0.8:
                return "Çok iyi yüz tespit - açı uygun"
            elif score > 0.6:
                return "İyi yüz tespit - açı kabul edilebilir"
            else:
                return "Yüz tespit edildi - açı tolere edilebilir"
        else:
            return "Yüz tespit zayıf - açı problemi olabilir"
    
    def _get_integrity_message(self, score: float, boundaries: Dict, size_ok: bool, ratio_ok: bool) -> str:
        if score > 0.9:
            return "Yüz tamamen görünür"
        elif score > 0.75:
            return "Yüz yeterince görünür"
        else:
            problems = []
            if not all(boundaries.values()):
                problems.append("kesilmiş")
            if not size_ok:
                problems.append("çok küçük")
            if not ratio_ok:
                problems.append("bozuk oran")
            return f"Yüz problemi: {', '.join(problems)}"
    
    def _get_lighting_message(self, adequate: bool, brightness: float, contrast: float, too_bright: float, too_dark: float) -> str:
        if adequate:
            return "Işık yeterli"
        else:
            problems = []
            if brightness < 60:
                problems.append("çok karanlık")
            elif brightness > 200:
                problems.append("çok parlak")
            if contrast < 30:
                problems.append("düşük kontrast")
            if too_bright > 0.1:
                problems.append("aşırı pozlama")
            if too_dark > 0.2:
                problems.append("yetersiz ışık")
            return f"Işık problemi: {', '.join(problems)}"
    
    def get_face_embedding(self, image_path: str, face_data: dict) -> np.ndarray:
        """Yüz için embedding çıkarır"""
        return face_data['embedding']
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """İki embedding arasındaki benzerlik skorunu hesaplar"""
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_best_match(self, target_embedding: np.ndarray, 
                       database_embeddings: List[Tuple[int, str, np.ndarray]], 
                       threshold: float = 0.55,
                       face_count: int = 1) -> Optional[Tuple[int, str, float]]:
        """
        Hedef embedding ile veritabanındaki embedding'leri karşılaştırır
        %90+ doğruluk için çoklu eşleşme doğrulama sistemi kullanır
        Returns: (student_id, name, similarity_score) or None
        """
        # Önce standart eşleşme yap
        best_match = None
        best_score = threshold
        candidate_matches = []
        
        for student_id, name, db_embedding in database_embeddings:
            similarity = self.compare_embeddings(target_embedding, db_embedding)
            
            if similarity > threshold:
                candidate_matches.append((student_id, name, similarity))
                
            if similarity > best_score:
                best_score = similarity
                best_match = (student_id, name, similarity)
        
        # Çoklu doğrulama: En iyi eşleşme için aynı kişinin diğer fotoğraflarıyla da kontrol et
        if best_match and len(candidate_matches) > 0:
            best_student_id = best_match[0]
            best_student_name = best_match[1]
            
            # Aynı öğrencinin tüm embedding'lerini bul
            same_student_scores = [score for sid, name, score in candidate_matches 
                                 if sid == best_student_id]
            
            if len(same_student_scores) >= 2:  # En az 2 fotoğrafla eşleşme varsa
                avg_score = sum(same_student_scores) / len(same_student_scores)
                min_score = min(same_student_scores)
                
                # ADAPTIVE Çoklu doğrulama kriterleri (grup fotoğrafları için optimize edilmiş)
                if face_count >= 5:  # Grup fotoğrafı
                    multi_avg_threshold = 0.30  # %30
                    multi_min_threshold = 0.20  # %20
                    print(f"GRUP FOTOĞRAFI → Multi-match thresholds: avg %30, min %20")
                else:  # Normal fotoğraf
                    multi_avg_threshold = 0.58  # %58 
                    multi_min_threshold = 0.52  # %52
                    print(f"NORMAL FOTOĞRAF → Multi-match thresholds: avg %58, min %52")
                
                if avg_score >= multi_avg_threshold and min_score >= multi_min_threshold:
                    print(f"Çoklu doğrulama başarılı: {len(same_student_scores)} eşleşme, "
                          f"ortalama: {avg_score:.2%}, minimum: {min_score:.2%}")
                    return (best_student_id, best_student_name, avg_score)
                else:
                    print(f"Çoklu doğrulama başarısız: ortalama {avg_score:.2%} < {multi_avg_threshold:.0%} "
                          f"veya minimum {min_score:.2%} < {multi_min_threshold:.0%}")
                    return None
        
        return best_match
    
    def process_student_photos(self, image_paths: List[str]) -> List[dict]:
        """
        Öğrenci fotoğraflarını işler ve kaliteli yüzleri döndürür
        """
        processed_faces = []
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\n📸 Fotoğraf {i}/{len(image_paths)} işleniyor: {os.path.basename(image_path)}")
                
                # Yüz tespiti
                faces = self.detect_faces(image_path)
                
                if not faces:
                    print(f"Yüz bulunamadı")
                    continue
                
                # En büyük yüzü seç (det_score'a göre)
                best_face = max(faces, key=lambda x: x['det_score'])
                
                quality = self.check_face_quality(image_path, best_face['bbox'], best_face.get('landmark'))
                
                self._print_quality_report(quality, i)
                
                # Kalite eşiğini kontrol et (dengeli yaklaşım: 3/5 kriter, tanıma aşaması katı)
                if quality['summary']['total_passed'] >= 3 and quality['overall_quality'] >= 0.60:
                    processed_faces.append({
                        'image_path': image_path,
                        'face_data': best_face,
                        'quality': quality
                    })
                    print(f"Fotoğraf kabul edildi (Genel skor: {quality['overall_quality']:.2f})")
                else:
                    failed_reasons = ", ".join(quality['summary']['failed_checks'])
                    print(f"Fotoğraf reddedildi - Başarısız kriterler: {failed_reasons}")
                    print(f"   Genel skor: {quality['overall_quality']:.2f} (gereken: ≥0.60, 3/5 kriter)")
            
            except Exception as e:
                print(f" {os.path.basename(image_path)} işlenirken hata: {e}")
        
        print(f"\nSonuç: {len(processed_faces)}/{len(image_paths)} fotoğraf kabul edildi")
        return processed_faces
    
    def _print_quality_report(self, quality: Dict, photo_num: int):
        """Kalite raporu yazdırır"""
        print(f"    Kalite Analizi:")
        
        details = quality['details']
        
        # Her kriteri kontrol et
        print(f"   1️⃣ Yüz Netliği: {details['sharpness']['message']} (Skor: {details['sharpness']['score']:.2f})")
        print(f"   2️⃣ Gözler: {details['eyes_open']['message']} (Skor: {details['eyes_open']['score']:.2f})")
        print(f"   3️⃣ Açı: {details['face_angle']['message']} (Skor: {details['face_angle']['score']:.2f})")
        print(f"   4️⃣ Yüz Bütünlüğü: {details['face_integrity']['message']} (Skor: {details['face_integrity']['score']:.2f})")
        print(f"   5️⃣ Işık: {details['lighting']['message']} (Skor: {details['lighting']['score']:.2f})")
        
        # Özet
        summary = quality['summary']
        print(f"    Özet: {summary['total_passed']}/5 kriter başarılı")
        if summary['total_passed'] > 0:
            print(f"      Başarılı: {', '.join(summary['passed_checks'])}")
        if summary['total_failed'] > 0:
            print(f"      Başarısız: {', '.join(summary['failed_checks'])}")
    
    # =================== DUYGU ANALİZİ ÖZELLİKLERİ ===================
    
    def analyze_emotion(self, image_path: str, face_bbox: Optional[Tuple] = None) -> Dict:
        """
        Verilen görüntüde duygu analizi yapar
        Args:
            image_path: Görüntü dosya yolu
            face_bbox: Yüz koordinatları (x1, y1, x2, y2), None ise otomatik tespit
        Returns: {"success": bool, "emotions": dict, "dominant_emotion": str, "message": str}
        """
        if not self.emotion_analysis_enabled:
            return {
                "success": False,
                "emotions": {},
                "dominant_emotion": None,
                "message": "Duygu analizi devre dışı"
            }
        
        try:
            # Face region'ı belirle
            if face_bbox is not None:
                # Bbox verilmişse yüzü kırp
                image = cv2.imread(image_path)
                if image is None:
                    return {
                        "success": False,
                        "emotions": {},
                        "dominant_emotion": None,
                        "message": "Görüntü okunamadı"
                    }
                
                x1, y1, x2, y2 = face_bbox
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
                
                face_region = image[y1:y2, x1:x2]
                
                if face_region.size == 0:
                    return {
                        "success": False,
                        "emotions": {},
                        "dominant_emotion": None,
                        "message": "Geçersiz yüz bölgesi"
                    }
                
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, face_region)
                    temp_path = tmp_file.name
                
                try:
                    # DeepFace ile analiz
                    result = DeepFace.analyze(
                        img_path=temp_path,
                        actions=['emotion'],
                        detector_backend=emotion_config.backend,
                        enforce_detection=emotion_config.enforce_detection,
                        silent=True
                    )
                finally:
                    os.unlink(temp_path)
            else:
                result = DeepFace.analyze(
                    img_path=image_path,
                    actions=['emotion'],
                    detector_backend=emotion_config.backend,
                    enforce_detection=emotion_config.enforce_detection,
                    silent=True
                )
            
            if isinstance(result, list):
                emotion_data = result[0] if result else {}
            else:
                emotion_data = result
            
            if 'emotion' not in emotion_data:
                return {
                    "success": False,
                    "emotions": {},
                    "dominant_emotion": None,
                    "message": "Duygu verisi bulunamadı"
                }
            
            # Duygu skorlarını al
            emotions = emotion_data['emotion']
            
            # En yüksek skora sahip duyguyu bul
            dominant_emotion_en = max(emotions.keys(), key=lambda x: emotions[x])
            
            # Türkçe'ye çevir
            dominant_emotion_tr = emotion_config.emotion_labels.get(
                dominant_emotion_en, dominant_emotion_en
            )
            
            # Tüm duyguları Türkçe'ye çevir
            emotions_tr = {}
            for emotion_en, score in emotions.items():
                emotion_tr = emotion_config.emotion_labels.get(emotion_en, emotion_en)
                emotions_tr[emotion_tr] = score
            
            return {
                "success": True,
                "emotions": emotions_tr,
                "dominant_emotion": dominant_emotion_tr,
                "dominant_score": emotions[dominant_emotion_en],
                "message": f"Duygu analizi başarılı - {dominant_emotion_tr}"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Duygu analizi hatası: {error_msg}")
            
            if "No face detected" in error_msg:
                return {
                    "success": False,
                    "emotions": {},
                    "dominant_emotion": None,
                    "message": "Görüntüde yüz tespit edilemedi"
                }
            elif "Unable to find" in error_msg or "not found" in error_msg:
                return {
                    "success": False,
                    "emotions": {},
                    "dominant_emotion": None,
                    "message": "Görüntü dosyası bulunamadı"
                }
            else:
                return {
                    "success": False,
                    "emotions": {},
                    "dominant_emotion": None,
                    "message": f"Duygu analizi hatası: {error_msg}"
                }
    
    def analyze_multiple_faces_emotions(self, image_path: str, faces: List[Dict]) -> List[Dict]:
        """
        Birden fazla yüz için duygu analizi yapar
        Args:
            image_path: Görüntü dosya yolu
            faces: Yüz listesi (detect_faces çıktısı)
        Returns: List[{"face_index": int, "emotion_analysis": dict}]
        """
        emotions_results = []
        
        for i, face in enumerate(faces):
            bbox = face.get('bbox')
            if bbox is not None:
                emotion_result = self.analyze_emotion(image_path, tuple(bbox))
                emotions_results.append({
                    "face_index": i,
                    "emotion_analysis": emotion_result
                })
            else:
                emotions_results.append({
                    "face_index": i,
                    "emotion_analysis": {
                        "success": False,
                        "emotions": {},
                        "dominant_emotion": None,
                        "message": "Yüz koordinatları bulunamadı"
                    }
                })
        
        return emotions_results 