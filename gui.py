import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import threading
import sys
from io import StringIO
from typing import List
from database import DatabaseManager
from face_processor import FaceProcessor

class FaceRecognitionGUI:
    def __init__(self):
        """Ana GUI sınıfını başlatır"""
        self.root = tk.Tk()
        self.root.title("🎓 OKULDAN Yüz Tanıma Sistemi - Öğrenci Takip")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        self.db_manager = DatabaseManager()
        self.face_processor = None
        
        self.selected_photos = [] 
        self.current_mode = "main"
        
        self.captured_photos = []  
        self.photo_qualities = []  
        self.recognition_ready = False  
        self.min_photos_captured = 1  
        self.target_accuracy = 0.65  
        self.max_photos_limit = 10   
        self.current_student_name = ""  
        self.current_student_id = ""  
        self.photo_count = 0  
        
        # Yüz tanıma için değişkenler
        self.current_recognition_photo = None  
        self.detected_faces = []  
        self.manual_face_buttons_frame = None  
        
        self.console_output = StringIO()
        
        self.setup_gui()
        self.init_face_processor()
    
    def init_face_processor(self):
        """Yüz işleme modülünü ayrı thread'de başlatır"""
        def load_models():
            try:

                old_stdout = sys.stdout
                sys.stdout = self.console_output
                
                self.face_processor = FaceProcessor()
                
                sys.stdout = old_stdout
                
                self.root.after(0, self.update_log_area)
                self.root.after(0, lambda: self.update_status(" Modeller hazır!"))
                
            except Exception as e:
                sys.stdout = old_stdout
                
                self.root.after(0, self.update_log_area)
                self.root.after(0, lambda: self.update_status(f" Model hatası: {e}"))
        
        self.update_status("Modeller yükleniyor... (Lütfen bekleyin)")
        threading.Thread(target=load_models, daemon=True).start()
    
    def update_log_area(self):
        """Log alanını console output ile günceller"""
        if hasattr(self, 'log_area'):
            output = self.console_output.getvalue()
            if output:
                self.log_area.insert(tk.END, output)
                self.log_area.see(tk.END)
                self.console_output = StringIO()  # Reset
    
    def setup_gui(self):
        """Ana GUI bileşenlerini oluşturur"""
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=(10, 0))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="OKULDAN YÜZ TANIMA SİSTEMİ",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        self.status_var = tk.StringVar(value="Sistem başlatılıyor...")
        status_frame = tk.Frame(self.root, bg='#34495e', height=40)
        status_frame.pack(fill='x', padx=10, pady=(5, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=('Arial', 10),
            fg='white',
            bg='#34495e'
        )
        self.status_label.pack(expand=True)
        
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.main_frame = tk.Frame(main_container, bg='#f0f0f0')
        self.main_frame.pack(side='left', fill='both', expand=True)
        
        log_frame = tk.Frame(main_container, bg='white', width=300)
        log_frame.pack(side='right', fill='y', padx=(10, 0))
        log_frame.pack_propagate(False)
        
        log_title = tk.Label(
            log_frame,
            text="istem Logları",
            font=('Arial', 12, 'bold'),
            bg='white',
            pady=10
        )
        log_title.pack(fill='x')
        
        self.log_area = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            width=40,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#333',
            wrap=tk.WORD
        )
        self.log_area.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.show_main_menu()
    
    def show_main_menu(self):
        """Ana menüyü gösterir"""
        self.clear_main_frame()
        self.current_mode = "main"
        
        cards_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        cards_frame.pack(expand=True)
        
        register_card = self.create_menu_card(
            cards_frame,
            "ÖĞRENCİ KAYIT",
            "Yeni öğrenci ekle ve\nfotoğraflarını kaydet",
            "#3498db",
            self.show_student_registration
        )
        register_card.pack(side='left', padx=20, pady=50)
        
        recognition_card = self.create_menu_card(
            cards_frame,
            "YÜZ TANIMA",
            "Fotoğraf yükleyerek\nöğrenci tanıma",
            "#e74c3c",
            self.show_face_recognition
        )
        recognition_card.pack(side='left', padx=20, pady=50)
        
        list_card = self.create_menu_card(
            cards_frame,
            "📋 ÖĞRENCİ LİSTESİ",
            "Kayıtlı öğrencileri\ngörüntüle",
            "#27ae60",
            self.show_student_list
        )
        list_card.pack(side='left', padx=20, pady=50)
        
        failed_card = self.create_menu_card(
            cards_frame,
            "BAŞARISIZ KAYITLAR",
            "Kalite analizi başarısız\nolan kayıtları görüntüle",
            "#f39c12",
            self.show_failed_registrations
        )
        failed_card.pack(side='left', padx=20, pady=50)
    
    def create_menu_card(self, parent, title, description, color, command):
        """Menü kartı oluşturur"""
        card_frame = tk.Frame(parent, bg=color, width=200, height=200)
        card_frame.pack_propagate(False)
        
        title_label = tk.Label(
            card_frame,
            text=title,
            font=('Arial', 14, 'bold'),
            fg='white',
            bg=color,
            wraplength=180
        )
        title_label.pack(pady=(20, 10))
        
        desc_label = tk.Label(
            card_frame,
            text=description,
            font=('Arial', 10),
            fg='white',
            bg=color,
            wraplength=180
        )
        desc_label.pack(pady=(0, 20))
        
        btn = tk.Button(
            card_frame,
            text="BAŞLAT",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg=color,
            cursor='hand2',
            command=command,
            relief='flat',
            padx=20
        )
        btn.pack(pady=10)
        
        return card_frame
    
    def show_student_registration(self):
        """Progressive öğrenci kayıt ekranını gösterir - sıralı fotoğraf alma sistemi"""
        self.clear_main_frame()
        self.current_mode = "registration"
        
        self.reset_progressive_system()
        
        back_btn = tk.Button(
            self.main_frame,
            text="← Ana Menü",
            command=self.show_main_menu,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        back_btn.pack(anchor='nw', pady=(0, 10))
        
        title = tk.Label(
            self.main_frame,
            text="👨‍🎓 YENİ ÖĞRENCİ KAYDI - AKILLI SİSTEM",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        )
        title.pack(pady=(0, 20))
        
        main_container = tk.Frame(self.main_frame, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20)
        
        left_panel = tk.Frame(main_container, bg='white', relief='solid', bd=1)
        left_panel.pack(side='left', fill='y', padx=(0, 10), pady=10)
        
        tk.Label(left_panel, text="Öğrenci Adı:", font=('Arial', 12, 'bold'), bg='white').pack(pady=(20, 5), anchor='w', padx=20)
        self.name_entry = tk.Entry(left_panel, font=('Arial', 12), width=25)
        self.name_entry.pack(padx=20, anchor='w')
        
        tk.Label(left_panel, text="Öğrenci ID:", font=('Arial', 12, 'bold'), bg='white').pack(pady=(10, 5), anchor='w', padx=20)
        self.student_id_entry = tk.Entry(left_panel, font=('Arial', 12), width=25)
        self.student_id_entry.pack(padx=20, anchor='w')
        
        tk.Label(left_panel, text="Öğrenci Sınıfı:", font=('Arial', 12, 'bold'), bg='white').pack(pady=(10, 5), anchor='w', padx=20)
        self.student_class_entry = tk.Entry(left_panel, font=('Arial', 12), width=25)
        self.student_class_entry.pack(padx=20, anchor='w')
        
        tk.Label(left_panel, text="Fotoğraf Ekleme:", font=('Arial', 12, 'bold'), bg='white').pack(pady=(20, 5), anchor='w', padx=20)
        
        self.add_photo_btn = tk.Button(
            left_panel,
            text="TEK FOTOĞRAF EKLE",
            command=self.capture_single_photo,
            bg='#3498db',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            width=20
        )
        self.add_photo_btn.pack(padx=20, pady=10, anchor='w')
        
        self.progress_label = tk.Label(
            left_panel,
            text="Durum: Henüz fotoğraf eklenmedi",
            font=('Arial', 10),
            bg='white',
            fg='#7f8c8d'
        )
        self.progress_label.pack(padx=20, anchor='w')
        
        self.recognition_rate_label = tk.Label(
            left_panel,
            text="Tanınma Oranı: Hesaplanmadı",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#e74c3c'
        )
        self.recognition_rate_label.pack(padx=20, pady=5, anchor='w')
        
        self.auto_register_label = tk.Label(
            left_panel,
            text="Sistem: En az 1 fotoğraf gerekli",
            font=('Arial', 10),
            bg='white',
            fg='#f39c12'
        )
        self.auto_register_label.pack(padx=20, pady=5, anchor='w')
        
        self.manual_register_btn = tk.Button(
            left_panel,
            text="ÖĞRENCİYİ KAYDET",
            command=self.manual_register_student,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            width=20,
            state='disabled' 
        )
        self.manual_register_btn.pack(padx=20, pady=10, anchor='w')
        
        right_panel = tk.Frame(main_container, bg='white', relief='solid', bd=1)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0), pady=10)
        
        tk.Label(right_panel, text="Eklenen Fotoğraflar ve Kalite Analizi", font=('Arial', 12, 'bold'), bg='white').pack(pady=(10, 5), padx=10)
        
        self.photos_analysis_area = scrolledtext.ScrolledText(
            right_panel,
            height=15,
            width=50,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#333',
            wrap=tk.WORD
        )
        self.photos_analysis_area.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        

        self.photos_analysis_area.insert(tk.END, "AKILLI FOTOĞRAF SİSTEMİ\n")
        self.photos_analysis_area.insert(tk.END, "=" * 50 + "\n\n")
        
        self.photos_analysis_area.insert(tk.END, "AKILLI KALİTE SİSTEMİ (%90+ DOĞRULUK İÇİN):\n")
        self.photos_analysis_area.insert(tk.END, "🔴 KRİTİK: 3/3 MUTLAKA GEÇMELİ\n")
        self.photos_analysis_area.insert(tk.END, "🟡 DESTEK: 2'sinden EN AZ 1'i GEÇMELİ\n")
        self.photos_analysis_area.insert(tk.END, "Genel kalite skoru EN AZ %60 olmalı\n\n")
        
        self.photos_analysis_area.insert(tk.END, "🔴 KRİTİK KRİTERLER (Olmazsa Olmaz):\n")
        self.photos_analysis_area.insert(tk.END, "1️⃣ Yüz Netliği - Bulanıksa tanınamaz\n")
        self.photos_analysis_area.insert(tk.END, "3️⃣ Açı - Görsel tabanlı akıllı tespit\n")  
        self.photos_analysis_area.insert(tk.END, "4️⃣ Yüz Bütünlüğü - Kesilmişse tanınamaz\n\n")
        
        self.photos_analysis_area.insert(tk.END, "🟡 DESTEK KRİTERLERİ (Esnek):\n")
        self.photos_analysis_area.insert(tk.END, "2️⃣ Gözler - Yarı kapalı tolere edilebilir\n")
        self.photos_analysis_area.insert(tk.END, "5️⃣ Işık - Modern AI kompanse edebilir\n\n")
        
        self.photos_analysis_area.insert(tk.END, "⚠️ ÖNEMLİ: Düşük kaliteli fotoğraflar sisteme EKLENMEYECEKTİR!\n\n")
        
        self.photos_analysis_area.insert(tk.END, "DİNAMİK SİSTEM nasıl çalışır:\n")
        self.photos_analysis_area.insert(tk.END, "1️⃣ 'FOTOĞRAF EKLE' butonuna tıklayın\n")
        self.photos_analysis_area.insert(tk.END, "2️⃣ Her fotoğraf anında analiz edilir\n")
        self.photos_analysis_area.insert(tk.END, "3️⃣ Kaliteli ise sisteme eklenir, değilse reddedilir\n")
        self.photos_analysis_area.insert(tk.END, "4️⃣ HER fotoğraf sonrası doğruluk oranı hesaplanır\n")
        self.photos_analysis_area.insert(tk.END, f"5️⃣ %{int(self.target_accuracy*100)}+ doğruluk ulaşılınca OTOMATIK KAYIT\n\n")
        self.photos_analysis_area.insert(tk.END, "Hedef: YETERLİ doğruluk oranı (kişiye göre değişir)\n")
        self.photos_analysis_area.insert(tk.END, f"Limit: Maksimum {self.max_photos_limit} deneme hakkı\n\n")
        self.photos_analysis_area.config(state='disabled')
    
    def _save_failed_registration(self, photo_path, quality, quality_report, failure_reason):
        """Başarısız kayıt bilgilerini veritabanına kaydeder"""
        try:
            student_name = getattr(self, 'current_student_name', 'Bilinmeyen')
            student_id = getattr(self, 'current_student_id', 'Bilinmeyen')
            student_class = getattr(self, 'current_student_class', '')
            
            quality_score = quality['overall_quality'] if quality else 0.0
            
            self.db_manager.add_failed_registration(
                student_name=student_name,
                student_id=student_id,
                student_class=student_class,
                photo_path=photo_path,
                quality_score=quality_score,
                quality_details=quality,
                quality_report=quality_report,
                failure_reason=failure_reason
            )
            
            print(f"Başarısız kayıt kaydedildi: {student_name} ({student_id}) - {failure_reason}")
            
        except Exception as e:
            print(f"Başarısız kayıt kaydetme hatası: {e}")
    
    def _check_smart_quality_criteria(self, quality):
        """
        Akıllı kalite kontrolü: Kritik + Destek kriter sistemi
        %90+ doğruluk için optimize 
        """
        details = quality['details']
        overall_quality = quality['overall_quality']

        if overall_quality < 0.60:
            return False
        
        critical_criteria = {
            'sharpness': details['sharpness']['is_sharp'],          
            'face_angle': details['face_angle']['is_suitable'],      
            'face_integrity': details['face_integrity']['is_complete'] 
        }
        
        critical_passed = sum(critical_criteria.values())
        
        support_criteria = {
            'eyes_open': details['eyes_open']['are_open'],        
            'lighting': details['lighting']['is_adequate']        
        }
        
        support_passed = sum(support_criteria.values())
        
        if critical_passed == 3 and support_passed >= 1:
            return True  # Mükemmel: Tüm kritik + en az 1 destek
        elif critical_passed == 3 and overall_quality >= 0.75:
            return True  # İyi: Tüm kritik + yüksek genel kalite
        else:
            return False  # Yetersiz
    
    def reset_progressive_system(self):
        """Dinamik Progressive fotoğraf sistemini resetler"""
        self.captured_photos = []
        self.photo_qualities = []
        self.recognition_ready = False
        self.current_student_name = ""
        self.current_student_id = ""
        self.current_student_class = ""
        self.photo_count = 0 
        
        # Manuel kayıt butonunu resetle - SAFE CHECK
        if hasattr(self, 'manual_register_btn') and self.manual_register_btn is not None:
            try:
                self.manual_register_btn.config(state='disabled', bg='#bdc3c7')
            except tk.TclError:
                # Widget destroyed, ignore
                self.manual_register_btn = None
        
    def capture_single_photo(self):
        """DİNAMİK TEK FOTOĞRAF - Her fotoğraf sonrası doğruluk analizi"""
        # Önce widget'ların mevcut olduğunu kontrol et - BUG FIX
        if not hasattr(self, 'name_entry') or self.name_entry is None:
            messagebox.showerror("Sistem Hatası", "Arayüz bozuk! Lütfen ekranı yeniden açın.")
            return
        if not hasattr(self, 'student_id_entry') or self.student_id_entry is None:
            messagebox.showerror("Sistem Hatası", "Arayüz bozuk! Lütfen ekranı yeniden açın.")
            return
            
        # Öğrenci bilgilerini kontrol et
        try:
            name = self.name_entry.get().strip()
            student_id = self.student_id_entry.get().strip()
            student_class = self.student_class_entry.get().strip()
        except tk.TclError:
            messagebox.showerror("Sistem Hatası", "Widget'lar bozuk! Lütfen ekranı yeniden açın.")
            return
        
        if not name or not student_id:
            messagebox.showerror("Eksik Bilgi", "Önce öğrenci adı ve ID'sini giriniz!")
            return
            
        self.current_student_name = name
        self.current_student_id = student_id
        self.current_student_class = student_class
        
        # GÜVENLIK: Maksimum deneme kontrolü
        if self.photo_count >= self.max_photos_limit:
            messagebox.showwarning("Limit", 
                f"Maksimum {self.max_photos_limit} fotoğraf deneme limitine ulaşıldı.\n"
                "Bu öğrenci için daha fazla fotoğraf eklenemez.")
            return
        
        # Tek fotoğraf seçim dialogu (çoklu seçim YOK) - GELİŞMİŞ FORMAT DESTEĞİ
        file_path = filedialog.askopenfilename(
            title=f"FOTOĞRAF SEÇİN - {self.photo_count + 1}. deneme (Hedef: %{int(self.target_accuracy*100)} doğruluk) - {name}",
            filetypes=[
                ("Tüm Desteklenen", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG dosyaları", "*.jpg *.jpeg"),
                ("PNG dosyaları", "*.png"),
                ("BMP dosyaları", "*.bmp"),
                ("TIFF dosyaları", "*.tiff"),
                ("WebP dosyaları", "*.webp"),
                ("Tüm dosyalar", "*.*")
            ],
            multiple=False  # Çoklu seçimi açıkça kapat
        )
        
        if file_path:
            # Sadece tek dosya seçildiğinden emin ol
            if isinstance(file_path, (list, tuple)):
                messagebox.showerror("Hata", "Lütfen sadece 1 fotoğraf seçin!\nÇoklu seçim yapılamaz.")
                return
            
            # Deneme sayısını artır
            self.photo_count += 1
                
            if self.face_processor:
                self.update_status(f"🔄 Fotoğraf #{self.photo_count} analiz ediliyor...")
                threading.Thread(target=self._analyze_single_photo, args=(file_path,), daemon=True).start()
            else:
                messagebox.showerror("Hata", "Yüz işleme modeli henüz hazır değil!\nLütfen bekleyiniz.")
    
    def _analyze_single_photo(self, photo_path):
        """Tek fotoğrafı analiz eder ve anında feedback verir"""
        try:
            photo_num = len(self.captured_photos) + 1
            filename = os.path.basename(photo_path)
            
            # Yüz tespiti
            faces = self.face_processor.detect_faces(photo_path)
            
            if not faces:
                # Başarısız kayıt - yüz bulunamadı
                self._save_failed_registration(photo_path, None, None, "Yüz tespit edilemedi")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, "no_face"
                ))
                return
            
            # En iyi yüzü seç 
            try:
                # Sadece geçerli bbox'a sahip yüzleri filtrele
                valid_faces = [face for face in faces if face.get('bbox') is not None and len(face.get('bbox', [])) >= 4]
                
                if not valid_faces:
                    # Başarısız kayıt - geçersiz yüz verisi
                    self._save_failed_registration(photo_path, None, None, "Geçersiz yüz verisi")
                    self.root.after(0, lambda: self._display_photo_feedback(
                        photo_num, filename, None, None, "invalid_face_data"
                    ))
                    return
                
                best_face = max(valid_faces, key=lambda x: x.get('det_score', 0))
                
                if best_face.get('bbox') is None or len(best_face.get('bbox', [])) < 4:
                    self._save_failed_registration(photo_path, None, None, "Geçersiz bbox verisi")
                    self.root.after(0, lambda: self._display_photo_feedback(
                        photo_num, filename, None, None, "invalid_bbox_data"
                    ))
                    return
                    
            except (ValueError, KeyError, TypeError) as e:
                print(f"Yüz veri hatası: {e}")
                # Başarısız kayıt - yüz veri hatası
                self._save_failed_registration(photo_path, None, None, f"Yüz veri hatası: {e}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, "face_data_error"
                ))
                return
            
            try:
                quality = self.face_processor.check_face_quality(
                    photo_path, best_face['bbox'], best_face.get('landmark')
                )
                
                if quality is None:
                    self._save_failed_registration(photo_path, None, None, "Kalite analizi başarısız")
                    self.root.after(0, lambda: self._display_photo_feedback(
                        photo_num, filename, None, None, "quality_analysis_failed"
                    ))
                    return
                    
            except Exception as quality_error:
                print(f"Kalite analiz hatası: {quality_error}")
                self._save_failed_registration(photo_path, None, None, f"Kalite analiz hatası: {quality_error}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, f"quality_error: {str(quality_error)}"
                ))
                return
            
            # AKILLI KALİTE KONTROLÜ - Kritik + Destek kriter sistemi
            if self._check_smart_quality_criteria(quality):
                # Fotoğraf kaliteli - sisteme ekle
                self.captured_photos.append({
                    'path': photo_path,
                    'face_data': best_face,
                    'quality': quality,
                    'filename': filename
                })
                
                # Feedback göster
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, quality, best_face, "accepted"
                ))
            else:
                # Fotoğraf kalitesiz - başarısız kayıt olarak kaydet
                quality_report = self.db_manager.generate_formatted_quality_report(photo_path, quality)
                self._save_failed_registration(photo_path, quality, quality_report, "Kalite kriterleri karşılanmadı")
                
                # Feedback göster
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, quality, best_face, "rejected"
                ))
            
            # DİNAMİK: Her fotoğraf sonrası doğruluk kontrolü yap
            self.root.after(0, self._check_recognition_readiness)
            
        except Exception as e:
            error_str = str(e)
            photo_num = len(self.captured_photos) + 1
            filename = os.path.basename(photo_path)
            
            print(f"Fotoğraf analiz hatası: {error_str}")
            
            # Hata tipine göre farklı mesajlar ve başarısız kayıt
            if "GÖRÜNTÜ OKUMA HATASI" in error_str or "Görüntü okunamadı" in error_str:
                # Detaylı görüntü okuma hatası
                self._save_failed_registration(photo_path, None, None, f"Görüntü okuma hatası: {error_str}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, f"image_error: {error_str}"
                ))
                self.root.after(0, lambda: self.update_status("Görüntü okuma hatası"))
            elif "Desteklenmeyen" in error_str:
                self._save_failed_registration(photo_path, None, None, f"Desteklenmeyen format: {error_str}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, f"format_error: {error_str}"
                ))
                self.root.after(0, lambda: self.update_status("Desteklenmeyen format"))
            elif "çok büyük" in error_str:
                self._save_failed_registration(photo_path, None, None, f"Dosya çok büyük: {error_str}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, f"size_error: {error_str}"
                ))
                self.root.after(0, lambda: self.update_status("Dosya çok büyük"))
            else:
                # Genel hata
                self._save_failed_registration(photo_path, None, None, f"Genel hata: {error_str}")
                self.root.after(0, lambda: self._display_photo_feedback(
                    photo_num, filename, None, None, f"error: {error_str}"
                ))
                self.root.after(0, lambda: self.update_status("Fotoğraf analizi başarısız"))
    
    def _display_photo_feedback(self, photo_num, filename, quality, face_data, status):
        """Fotoğraf için 5 soruyla detaylı feedback gösterir"""
        # Text alanını düzenlenebilir yap
        self.photos_analysis_area.config(state='normal')
        
        # Fotoğraf başlığı
        self.photos_analysis_area.insert(tk.END, f"\n📸 FOTOĞRAF #{photo_num}: {filename}\n")
        self.photos_analysis_area.insert(tk.END, "=" * 50 + "\n")
        
        if status == "no_face":
            self.photos_analysis_area.insert(tk.END, "YÜZ BULUNAMADI!\n")
            self.photos_analysis_area.insert(tk.END, "Lütfen yüzün net görüldüğü bir fotoğraf seçin.\n\n")
            self._update_progress_status("Son fotoğrafta yüz bulunamadı")
            
        elif status == "invalid_face_data":
            self.photos_analysis_area.insert(tk.END, "YÜZ VERİSİ GEÇERSİZ!\n")
            self.photos_analysis_area.insert(tk.END, "Yüz tespit edildi ama veri yapısı bozuk.\n")
            self.photos_analysis_area.insert(tk.END, "Farklı bir fotoğraf deneyin.\n\n")
            self._update_progress_status("Yüz verisi geçersiz")
            
        elif status == "invalid_bbox_data":
            self.photos_analysis_area.insert(tk.END, "YÜZ KOORDİNATLARI GEÇERSİZ!\n")
            self.photos_analysis_area.insert(tk.END, "Yüz sınırları belirlenemedi.\n")
            self.photos_analysis_area.insert(tk.END, "Daha net bir fotoğraf deneyin.\n\n")
            self._update_progress_status("Yüz koordinatları geçersiz")
            
        elif status == "face_data_error":
            self.photos_analysis_area.insert(tk.END, "YÜZ VERİ YAPISI HATASI!\n")
            self.photos_analysis_area.insert(tk.END, "AI modeli yüz verilerini düzgün işleyemedi.\n")
            self.photos_analysis_area.insert(tk.END, "Başka bir fotoğraf deneyin.\n\n")
            self._update_progress_status("Yüz veri yapısı hatası")
            
        elif status == "quality_analysis_failed":
            self.photos_analysis_area.insert(tk.END, "KALİTE ANALİZİ BAŞARISIZ!\n")
            self.photos_analysis_area.insert(tk.END, "Fotoğraf kalitesi analiz edilemedi.\n")
            self.photos_analysis_area.insert(tk.END, "Tekrar deneyin.\n\n")
            self._update_progress_status("Kalite analizi başarısız")
            
        elif status.startswith("error") or status.startswith("image_error") or status.startswith("format_error") or status.startswith("size_error") or status.startswith("quality_error"):
            # Hata mesajını ayrıştır
            if ':' in status:
                error_type, error_message = status.split(':', 1)
            else:
                error_type = "error"
                error_message = status
            
            if error_type == "image_error":
                # Detaylı görüntü okuma hatası
                self.photos_analysis_area.insert(tk.END, "GÖRÜNTÜ OKUMA HATASI!\n")
                self.photos_analysis_area.insert(tk.END, "=" * 30 + "\n")
                self.photos_analysis_area.insert(tk.END, f"{error_message}\n\n")
                self._update_progress_status("Görüntü okunamadı")
                
            elif error_type == "format_error":
                # Format hatası
                self.photos_analysis_area.insert(tk.END, "DOSYA FORMAT HATASI!\n")
                self.photos_analysis_area.insert(tk.END, "=" * 30 + "\n")
                self.photos_analysis_area.insert(tk.END, f"{error_message}\n\n")
                self._update_progress_status("Desteklenmeyen format")
                
            elif error_type == "size_error":
                # Boyut hatası
                self.photos_analysis_area.insert(tk.END, "DOSYA BOYUT HATASI!\n")
                self.photos_analysis_area.insert(tk.END, "=" * 30 + "\n")
                self.photos_analysis_area.insert(tk.END, f"{error_message}\n\n")
                self._update_progress_status("Dosya çok büyük")
                
            elif error_type == "quality_error":
                # Kalite analiz hatası
                self.photos_analysis_area.insert(tk.END, "KALİTE ANALİZ HATASI!\n")
                self.photos_analysis_area.insert(tk.END, "=" * 30 + "\n")
                self.photos_analysis_area.insert(tk.END, f"Teknik Detay: {error_message}\n")
                self.photos_analysis_area.insert(tk.END, "Bu fotoğrafın kalitesi analiz edilemedi.\n")
                self.photos_analysis_area.insert(tk.END, "Başka bir fotoğraf deneyin.\n\n")
                self._update_progress_status("Kalite analizi hatası")
                
            else:
                # Genel hata
                self.photos_analysis_area.insert(tk.END, f"🔴 HATA: {error_message}\n\n")
                self._update_progress_status("Fotoğraf analiz hatası")
            
        elif status in ["accepted", "rejected"] and quality:
            # 5 temel soru ve yanıtları
            details = quality['details']
            
            self.photos_analysis_area.insert(tk.END, "AKILLI KALİTE ANALİZİ:\n\n")
            
            # KRİTİK KRİTERLER
            self.photos_analysis_area.insert(tk.END, "🔴 KRİTİK KRİTERLER (3/3 MUTLAKA GEÇMELİ):\n")
            
            # 1. Yüz net mi? (KRİTİK)
            sharpness = details['sharpness']
            self.photos_analysis_area.insert(tk.END, "1️Yüz Netliği (KRİTİK):\n")
            if sharpness['is_sharp']:
                self.photos_analysis_area.insert(tk.END, f"   {sharpness['message']} (Skor: {sharpness['score']:.2f})\n")
            else:
                self.photos_analysis_area.insert(tk.END, f"   {sharpness['message']} (Skor: {sharpness['score']:.2f})\n")
            
            # 3. Açı uygun mu? (KRİTİK - GÖRSEL TABANLI)
            angle = details['face_angle']
            self.photos_analysis_area.insert(tk.END, "3️⃣ Açı Uygunluğu (KRİTİK - GÖRSEL TABANLI):\n")
            if angle['is_suitable']:
                self.photos_analysis_area.insert(tk.END, f"    {angle['message']} (Skor: {angle['score']:.2f})\n")
            else:
                self.photos_analysis_area.insert(tk.END, f"    {angle['message']} (Skor: {angle['score']:.2f})\n")
            
            # 4. Yüz bütünlüğü algılanabiliyor mu? (KRİTİK)
            integrity = details['face_integrity']
            self.photos_analysis_area.insert(tk.END, "4️⃣ Yüz Bütünlüğü (KRİTİK):\n")
            if integrity['is_complete']:
                self.photos_analysis_area.insert(tk.END, f"    {integrity['message']} (Skor: {integrity['score']:.2f})\n")
            else:
                self.photos_analysis_area.insert(tk.END, f"    {integrity['message']} (Skor: {integrity['score']:.2f})\n")
            
            # DESTEK KRİTERLERİ
            self.photos_analysis_area.insert(tk.END, "\n🟡 DESTEK KRİTERLERİ (2'sinden EN AZ 1'i GEÇMELİ):\n")
            
            # 2. Gözler açık mı? (DESTEK)
            eyes = details['eyes_open']
            self.photos_analysis_area.insert(tk.END, "2️⃣ Gözler (DESTEK):\n")
            if eyes['are_open']:
                self.photos_analysis_area.insert(tk.END, f"    {eyes['message']} (Skor: {eyes['score']:.2f})\n")
            else:
                self.photos_analysis_area.insert(tk.END, f"    {eyes['message']} (Skor: {eyes['score']:.2f})\n")
            
            # 5. Işık yeterli mi? (DESTEK)
            lighting = details['lighting']
            self.photos_analysis_area.insert(tk.END, "5️⃣ Işık (DESTEK):\n")
            if lighting['is_adequate']:
                self.photos_analysis_area.insert(tk.END, f"    {lighting['message']} (Skor: {lighting['score']:.2f})\n")
            else:
                self.photos_analysis_area.insert(tk.END, f"    {lighting['message']} (Skor: {lighting['score']:.2f})\n")
            
            # Genel sonuç
            summary = quality['summary']
            overall_quality = quality['overall_quality']
            
            self.photos_analysis_area.insert(tk.END, f"\n GENEL SONUÇ:\n")
            self.photos_analysis_area.insert(tk.END, f"   Başarılı kriterler: {summary['total_passed']}/5\n")
            self.photos_analysis_area.insert(tk.END, f"   Genel kalite skoru: {overall_quality:.2f}/1.00\n")
            
            # AKILLI KARAR ANALİZİ
            details = quality['details']
            
            # Kritik kriterler kontrolü
            critical_passed = [
                details['sharpness']['is_sharp'],
                details['face_angle']['is_suitable'], 
                details['face_integrity']['is_complete']
            ]
            critical_count = sum(critical_passed)
            
            # Destek kriterler kontrolü  
            support_passed = [
                details['eyes_open']['are_open'],
                details['lighting']['is_adequate']
            ]
            support_count = sum(support_passed)
            
            self.photos_analysis_area.insert(tk.END, f"\n📊 AKILLI KARAR ANALİZİ:\n")
            self.photos_analysis_area.insert(tk.END, f"   🔴 Kritik: {critical_count}/3 başarılı\n")
            self.photos_analysis_area.insert(tk.END, f"   🟡 Destek: {support_count}/2 başarılı\n")
            self.photos_analysis_area.insert(tk.END, f"   📊 Genel kalite: %{overall_quality*100:.0f}\n")
            
            # DEBUG: Detaylı skorlar
            self.photos_analysis_area.insert(tk.END, f"\n🔍 DETAYLI SKORLAR (Debug):\n")
            self.photos_analysis_area.insert(tk.END, f"   Netlik: {details['sharpness']['score']:.2f} ({'✅' if details['sharpness']['is_sharp'] else '❌'})\n")
            self.photos_analysis_area.insert(tk.END, f"   Açı: {details['face_angle']['score']:.2f} ({'✅' if details['face_angle']['is_suitable'] else '❌'})\n")
            self.photos_analysis_area.insert(tk.END, f"   Bütünlük: {details['face_integrity']['score']:.2f} ({'✅' if details['face_integrity']['is_complete'] else '❌'})\n")
            self.photos_analysis_area.insert(tk.END, f"   Gözler: {details['eyes_open']['score']:.2f} ({'✅' if details['eyes_open']['are_open'] else '❌'})\n")
            self.photos_analysis_area.insert(tk.END, f"   Işık: {details['lighting']['score']:.2f} ({'✅' if details['lighting']['is_adequate'] else '❌'})\n")
            
            if status == "accepted":
                self.photos_analysis_area.insert(tk.END, "\n    FOTOĞRAF SİSTEME EKLENDİ!\n")
                if critical_count == 3 and support_count >= 1:
                    self.photos_analysis_area.insert(tk.END, "   Neden: Tüm kritik kriterler + en az 1 destek başarılı\n")
                elif critical_count == 3 and overall_quality >= 0.75:
                    self.photos_analysis_area.insert(tk.END, "   Neden: Tüm kritik kriterler + yüksek genel kalite\n")
                self._update_progress_status(f"✅ Fotoğraf #{photo_num} kabul edildi")
                
            elif status == "rejected":
                self.photos_analysis_area.insert(tk.END, "\n   FOTOĞRAF REDDEDİLDİ - SİSTEME EKLENMEDİ!\n")
                
                # Neden reddedildiğini açıkla
                if critical_count < 3:
                    failed_critical = []
                    if not details['sharpness']['is_sharp']:
                        failed_critical.append("Yüz Netliği")
                    if not details['face_angle']['is_suitable']:
                        failed_critical.append("Açı Uygunluğu")
                    if not details['face_integrity']['is_complete']:
                        failed_critical.append("Yüz Bütünlüğü")
                    self.photos_analysis_area.insert(tk.END, f"    Neden: Kritik kriterler başarısız → {', '.join(failed_critical)}\n")
                
                elif support_count == 0 and overall_quality < 0.75:
                    self.photos_analysis_area.insert(tk.END, "    Neden: Hiç destek kriteri geçemiyor + düşük genel kalite\n")
                
                elif overall_quality < 0.60:
                    self.photos_analysis_area.insert(tk.END, f"   Neden: Genel kalite çok düşük (%{overall_quality*100:.0f} < %60)\n")
                
                self.photos_analysis_area.insert(tk.END, "\n ÖNERİ: Kritik sorunları çözün ve tekrar deneyin!\n")
                self._update_progress_status(f" Fotoğraf #{photo_num} reddedildi")
        
        self.photos_analysis_area.insert(tk.END, "\n" + "─" * 50 + "\n")
        self.photos_analysis_area.see(tk.END)
        self.photos_analysis_area.config(state='disabled')
        
        # Progress durumunu güncelle
        self._update_overall_progress()
    
    def _update_progress_status(self, message):
        """İlerleme durumunu günceller"""
        self.progress_label.config(text=f" Durum: {message}")
    
    def _update_overall_progress(self):
        """Genel ilerleme durumunu günceller"""
        accepted_photos = len(self.captured_photos)  # Artık sadece kaliteli fotoğraflar var
        
        if accepted_photos == 0:
            self.progress_label.config(text="Durum: Henüz kaliteli fotoğraf eklenmedi")
        else:
            self.progress_label.config(text=f"Durum: {accepted_photos} kaliteli fotoğraf eklendi")
        
        # DİNAMİK buton durumu - maksimum deneme sayısına göre
        if self.photo_count >= self.max_photos_limit:
            self.add_photo_btn.config(text=f"MAKSİMUM {self.max_photos_limit} DENEME", state='disabled')
        else:
            remaining = self.max_photos_limit - self.photo_count
            self.add_photo_btn.config(
                text=f"📸 FOTOĞRAF EKLE ({accepted_photos} başarılı, {remaining} deneme kaldı)", 
                state='normal'
            )
    
    def _check_recognition_readiness(self):
        """DİNAMİK DOĞRULUK KONTROLü - Her fotoğraf sonrası hesaplama"""
        quality_photos = self.captured_photos
        photo_count = len(quality_photos)
        
        if photo_count == 0:
            return
        
        try:
            if photo_count == 1:
                # TEK FOTOĞRAF: Temel kontrol
                self.recognition_rate_label.config(
                    text=" 1 Fotoğraf: Temel kayıt hazır",
                    fg='#f39c12'
                )
                # TEK FOTOĞRAF İÇİN MANUEL KAYIT AKTİF
                self.manual_register_btn.config(state='normal', bg='#27ae60')
                self.auto_register_label.config(
                    text=f"Tek fotoğraf ile temel kayıt yapılabilir - Manuel kayıt butonu aktif\n⏳ Hedef %{int(self.target_accuracy*100)} doğruluk için daha fazla fotoğraf ekleyin",
                    fg='#3498db'
                )
                return
            
            # ÇOKLU FOTOĞRAF: İç tutarlılık analizi
            embeddings = []
            for photo in quality_photos:
                embedding = photo['face_data']['embedding']
                embeddings.append(embedding)
            
            # Kendi aralarında benzerlik kontrolü
            total_comparisons = 0
            total_similarity = 0
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self.face_processor.compare_embeddings(embeddings[i], embeddings[j])
                    total_comparisons += 1
                    total_similarity += similarity
            
            avg_similarity = total_similarity / total_comparisons if total_comparisons > 0 else 0
            
            # DİNAMİK DOĞRULUK DEĞERLENDİRME
            is_sufficient = avg_similarity >= self.target_accuracy
            
            # Tanınma oranını güncelle
            color = '#27ae60' if is_sufficient else '#e74c3c'
            self.recognition_rate_label.config(
                text=f"🎯 {photo_count} Fotoğraf: {avg_similarity:.1%} doğruluk (Hedef: {self.target_accuracy:.1%})",
                fg=color
            )
            
            # MANUEL KAYIT BUTONU DURUMU
            if is_sufficient or photo_count == 1:
                # YETERLİ DOĞRULUK VEYA TEK FOTOĞRAF - MANUEL KAYIT AKTİF
                self.manual_register_btn.config(state='normal', bg='#27ae60')
                if is_sufficient:
                    self.auto_register_label.config(
                        text=f" %{avg_similarity:.1%} doğruluk YETERLİ! Manuel kayıt butonu aktif",
                        fg='#27ae60'
                    )
                else:
                    self.auto_register_label.config(
                        text="Tek fotoğraf ile temel kayıt yapılabilir - Manuel kayıt butonu aktif",
                        fg='#3498db'
                    )
            else:
                # YETERSİZ DOĞRULUK - MANUEL KAYIT PASİF
                self.manual_register_btn.config(state='disabled', bg='#bdc3c7')
                needed = self.target_accuracy - avg_similarity
                self.auto_register_label.config(
                    text=f"⏳ %{needed:.1%} daha yüksek doğruluk gerekli - Yeni fotoğraf ekleyin",
                    fg='#f39c12'
                )
                
        except Exception as e:
            self.recognition_rate_label.config(
                text=" Doğruluk Hesaplama Hatası",
                fg='#e74c3c'
            )
            print(f"Recognition check error: {e}")
    
    def draw_faces_on_image(self, image_path, faces, face_matches=None):
        """Fotoğraf üzerine tespit edilen yüzleri yeşil karelerle + DİNAMİK BÜYÜK İSİMLERLE çizer"""
        try:
            # PIL ile resmi yükle
            pil_image = Image.open(image_path)
            draw = ImageDraw.Draw(pil_image)
            image_width, image_height = pil_image.size
            
            # DİNAMİK FONT BOYUTU - Fotoğraf boyutuna göre ayarla
            def calculate_font_size(image_width):
                """Fotoğraf genişliğine göre optimal font boyutu hesaplar"""
                if image_width < 400:
                    return 16, 20  # Küçük fotoğraflar için (medium, large)
                elif image_width < 800:
                    return 24, 30  # Orta boyut fotoğraflar için 
                elif image_width < 1200:
                    return 32, 40  # Büyük fotoğraflar için
                else:
                    return 40, 50  # Çok büyük fotoğraflar için
            
            medium_size, large_size = calculate_font_size(image_width)
            
            # Font ayarları - DİNAMİK BOYUTLARLA
            try:
                # Font yüklemeye çalış (Windows için)
                font_large = ImageFont.truetype("arial.ttf", large_size)   # Numaralar için - BÜYÜK
                font_medium = ImageFont.truetype("arial.ttf", medium_size) # İsimler için - BÜYÜK
                font_small = ImageFont.truetype("arial.ttf", max(12, medium_size - 8)) # Score için
            except:
                # Varsayılan font kullan (fallback)
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default() 
                font_small = ImageFont.load_default()
            
            # Her yüz için yeşil kare + isim çiz
            for i, face in enumerate(faces, 1):
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Yeşil kare çiz (kalın çizgi)
                draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
                
                # Yüz numarası yaz (yeşil)
                draw.text((x1 + 5, y1 + 5), str(i), fill='lime', font=font_large)
                
                # Detection score'u yaz (yeşil) - KÜÇÜK FONT
                det_score = face.get('det_score', 0)
                score_text = f"{det_score:.2f}"
                draw.text((x1 + 5, y2 - 30), score_text, fill='lime', font=font_small)
                
                # İSİM ETİKETİ + DUYGU ANALİZİ - YENİ ÖZELLİK!
                if face_matches:
                    # Bu yüz için eşleşme var mı kontrol et
                    face_name = "Bilinmeyen"
                    emotion_text = ""
                    name_color = 'orange'  # Bilinmeyen için turuncu
                    
                    # Face matches içinde bu yüz indeksini ara
                    for match in face_matches:
                        if match['face_index'] == i:
                            face_name = match['name']
                            name_color = 'red'  # Tanınan için kırmızı
                            
                            # Duygu analizi varsa ekle
                            if 'emotion_analysis' in match:
                                emotion_data = match['emotion_analysis']
                                if emotion_data and emotion_data.get('success'):
                                    emotion_text = f" ({emotion_data.get('dominant_emotion', 'Bilinmeyen')})"
                                else:
                                    emotion_text = " (Duygu tespit edilemedi)"
                            break
                    
                    # İsim + Duygu etiketini AKILLI POZİSYONLAMA ile yaz
                    # Tam metin (isim + duygu)
                    full_text = face_name + emotion_text
                    
                    # Önce text boyutunu hesapla
                    temp_bbox = draw.textbbox((0, 0), full_text, font=font_medium)
                    text_width = temp_bbox[2] - temp_bbox[0]
                    text_height = temp_bbox[3] - temp_bbox[1]
                    
                    # Etiket pozisyonu hesapla - sağ kenarı kontrol et
                    name_x = x2 + 10  # Yüz karesinin sağ tarafı
                    name_y = y1 + (y2 - y1) // 2  # Yüzün ortası
                    
                    # Eğer etiket fotoğrafın dışına taşacaksa sol tarafa al
                    if name_x + text_width + max(6, medium_size // 4) * 2 > image_width:
                        name_x = x1 - text_width - 15  # Sol tarafa yerleştir
                        # Sol taraf da yetersizse, üst tarafa al
                        if name_x < 0:
                            name_x = x1
                            name_y = y1 - text_height - 10  # Üst tarafa
                            # Üst de yetersizse alt tarafa
                            if name_y < 0:
                                name_y = y2 + text_height + 10  # Alt tarafa
                    
                    # Arka plan için beyaz dikdörtgen çiz (okunabilirlik için) - DİNAMİK PADDING
                    text_bbox = draw.textbbox((name_x, name_y), full_text, font=font_medium)
                    # Font boyutuna göre padding ayarla
                    padding = max(6, medium_size // 4)  # Font boyutunun 1/4'ü kadar padding, minimum 6px
                    draw.rectangle([
                        text_bbox[0] - padding, 
                        text_bbox[1] - padding,
                        text_bbox[2] + padding, 
                        text_bbox[3] + padding
                    ], fill='white', outline='black', width=2)  # Çerçeve kalınlığı da artırıldı
                    
                    # İsim + Duygu etiketini çiz - BÜYÜK VE KALIN + GÖLGE EFEKTİ
                    # Önce gölge efekti için koyu gri yazı (1px offset)
                    shadow_color = 'gray'
                    draw.text((name_x + 1, name_y + 1), full_text, fill=shadow_color, font=font_medium)
                    # Sonra asıl renkte yazı
                    draw.text((name_x, name_y), full_text, fill=name_color, font=font_medium)
                
            return pil_image
        except Exception as e:
            print(f"Yüz çizme hatası: {e}")
            # Hata durumunda orijinal resmi dön
            return Image.open(image_path)
    
    def resize_image_for_display(self, pil_image, max_width=600, max_height=400):
        """Resmi ekranda gösterim için boyutlandırır"""
        try:
            original_width, original_height = pil_image.size
            
            # Oranı koru
            ratio = min(max_width / original_width, max_height / original_height)
            
            if ratio < 1:  # Sadece küçültme gerekiyorsa
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return pil_image
        except Exception as e:
            print(f"Resim boyutlandırma hatası: {e}")
            return pil_image
    
    def display_photo_with_faces(self, image_path, faces, face_matches=None):
        """Fotoğrafı yüz işaretlemeleri + İSİM ETİKETLERİYLE birlikte gösterir"""
        try:
            # Yüzleri ve isimleri çiz
            pil_image = self.draw_faces_on_image(image_path, faces, face_matches)
            
            # Ekran için boyutlandır
            pil_image = self.resize_image_for_display(pil_image)
            
            # Tkinter formatına çevir
            photo = ImageTk.PhotoImage(pil_image)
            
            # GUI'de göster
            self.photo_label.configure(image=photo, text="")
            self.photo_label.image = photo  # Referansı tut
            
            # Başlık güncelle - SAFE CHECK
            if hasattr(self, 'photo_title') and self.photo_title is not None:
                try:
                    filename = os.path.basename(image_path)
                    self.photo_title.configure(
                        text=f"📷 {filename}",
                        fg='#2c3e50'
                    )
                except tk.TclError:
                    # Widget destroyed
                    self.photo_title = None
            
            # Tespit bilgisi güncelle
            face_count = len(faces)
            if face_count > 0:
                recognized_count = len(face_matches) if face_matches else 0
                if recognized_count > 0:
                    self._safe_update_text_widget('detection_info',
                        f" {face_count} yüz tespit edildi → {recognized_count} tanındı\n\n🔴 Kırmızı: Tanınan öğrenciler\n🟠 Turuncu: Bilinmeyen yüzler\n\n📋 Detaylı sonuçlar aşağıda görüntülenecek...",
                        fg='#27ae60'
                    )
                else:
                    self._safe_update_text_widget('detection_info',
                        f" {face_count} yüz tespit edildi → Hiçbiri tanınamadı\n\n🟠 Tüm yüzler: Bilinmeyen\n\n💡 İpucu: Önce öğrencileri sisteme kaydedin",
                        fg='#f39c12'
                    )
            else:
                self._safe_update_text_widget('detection_info',
                    " Hiç yüz tespit edilemedi\n\n📸 Lütfen şunları kontrol edin:\n• Fotoğrafta yüz var mı?\n• Görüntü kalitesi yeterli mi?\n• Yüz açık şekilde görünüyor mu?",
                    fg='#e74c3c'
                )
                
        except Exception as e:
            print(f" Fotoğraf gösterim hatası: {e}")
            self._safe_update_text_widget('detection_info',
                f" Fotoğraf gösterim hatası: {e}\n\n🔧 Teknik detaylar:\n{str(e)}",
                fg='#e74c3c'
            )
    
    def manual_register_student(self):
        """MANUEL ÖĞRENCİ KAYDI - Kullanıcı butona tıkladığında"""
        try:
            if not self.captured_photos:
                messagebox.showerror("Hata", "Önce kaliteli fotoğraf ekleyin!")
                return
            
            # Doğruluk kontrolü
            if len(self.captured_photos) > 1:
                embeddings = []
                for photo in self.captured_photos:
                    embedding = photo['face_data']['embedding']
                    embeddings.append(embedding)
                
                total_comparisons = 0
                total_similarity = 0
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity = self.face_processor.compare_embeddings(embeddings[i], embeddings[j])
                        total_comparisons += 1
                        total_similarity += similarity
                
                avg_similarity = total_similarity / total_comparisons if total_comparisons > 0 else 0
                
                if avg_similarity < self.target_accuracy:
                    response = messagebox.askyesno(
                        "Düşük Doğruluk Uyarısı", 
                        f"Mevcut doğruluk: %{avg_similarity:.1%}\n"
                        f"Hedef doğruluk: %{self.target_accuracy:.1%}\n\n"
                        f"Yine de kaydetmek istiyor musunuz?"
                    )
                    if not response:
                        return
            
            # Kayıt işlemini yap
            self._perform_registration("MANUEL")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Manuel kayıt hatası: {e}")
    
    def _perform_registration(self, kayit_tipi="OTOMATIK"):
        """ORTAK KAYIT FONKSİYONU - Manuel ve otomatik kayıt için"""
        try:
            quality_photos = self.captured_photos
            
            if not quality_photos:
                messagebox.showerror("Hata", "Kaliteli fotoğraf bulunamadı!")
                return
            
            # Veritabanına kaydet
            student_pk = self.db_manager.add_student(self.current_student_name, self.current_student_id, self.current_student_class)
            
            for photo_data in quality_photos:
                embedding = photo_data['face_data']['embedding']
                quality_score = photo_data['quality']['overall_quality']
                quality_details = photo_data['quality']  # Tüm kalite detayları
                photo_path = photo_data['path']
                
                # Formatlanmış kalite raporu oluştur
                quality_report = self.db_manager.generate_formatted_quality_report(photo_path, quality_details)
                
                self.db_manager.add_face_embedding(student_pk, embedding, photo_path, quality_score, quality_details, quality_report)
            
            # Doğruluk hesaplama
            if len(quality_photos) > 1:
                embeddings = [p['face_data']['embedding'] for p in quality_photos]
                total_comparisons = 0
                total_similarity = 0
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity = self.face_processor.compare_embeddings(embeddings[i], embeddings[j])
                        total_comparisons += 1
                        total_similarity += similarity
                avg_similarity = total_similarity / total_comparisons if total_comparisons > 0 else 0
                dogruluk_mesaji = f" Doğruluk: {avg_similarity:.1%}"
            else:
                dogruluk_mesaji = " Doğruluk: Tek fotoğraf (temel kayıt)"
            
            # Başarı mesajı
            avg_quality = sum(p['quality']['overall_quality'] for p in quality_photos) / len(quality_photos)
            
            messagebox.showinfo(
                f" {kayit_tipi} KAYIT BAŞARILI!",
                f" {self.current_student_name} başarıyla kaydedildi!\n\n"
                f" Kullanılan fotoğraf: {len(quality_photos)} (tümü kaliteli)\n"
                f" Ortalama kalite: {avg_quality:.2%}\n"
                f"{dogruluk_mesaji}\n"
                f" Tanınma sistemi: Aktif\n\n"
                f" Sadece kaliteli fotoğraflar sisteme eklendi!"
            )
            
            # Ana menüye dön
            self.show_main_menu()
            
        except ValueError as e:
            messagebox.showerror("Veritabanı Hatası", str(e))
        except Exception as e:
            messagebox.showerror("Kayıt Hatası", f"Beklenmeyen hata: {e}")
            print(f"Registration error: {e}")
    
    def _auto_register_student(self):
        """Otomatik öğrenci kaydı yapar"""
        try:
            quality_photos = self.captured_photos
            
            if not quality_photos:
                messagebox.showwarning("Uyarı", "Kaliteli fotoğraf bulunamadı.")
                return
            
            # Ortak kayıt fonksiyonunu kullan
            self._perform_registration("OTOMATIK")
            
        except Exception as e:
            messagebox.showerror("Otomatik Kayıt Hatası", f"Beklenmeyen hata: {e}")
            print(f"Auto register error: {e}")
    
    def show_face_recognition(self):
        """Yüz tanıma ekranını scrollable olarak gösterir"""
        self.clear_main_frame()
        self.current_mode = "recognition"
        
        # SCROLLABLE MAIN CONTAINER
        # Ana canvas ve scrollbar container
        main_container = tk.Frame(self.main_frame, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True)
        
        # Canvas for scrolling
        self.recognition_canvas = tk.Canvas(
            main_container, 
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.recognition_canvas.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        recognition_scrollbar = tk.Scrollbar(
            main_container, 
            orient='vertical', 
            command=self.recognition_canvas.yview
        )
        recognition_scrollbar.pack(side='right', fill='y')
        self.recognition_canvas.configure(yscrollcommand=recognition_scrollbar.set)
        
        # Scrollable frame inside canvas
        self.scrollable_frame = tk.Frame(self.recognition_canvas, bg='#f0f0f0')
        self.canvas_window = self.recognition_canvas.create_window(
            (0, 0), 
            window=self.scrollable_frame, 
            anchor='nw'
        )
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            if self.recognition_canvas.winfo_exists():
                self.recognition_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to canvas
        self.recognition_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        # Configure scroll region when frame changes
        def _configure_scrollregion(event=None):
            if self.recognition_canvas.winfo_exists():
                self.recognition_canvas.configure(scrollregion=self.recognition_canvas.bbox("all"))
        
        def _configure_canvas_width(event=None):
            if self.recognition_canvas.winfo_exists():
                canvas_width = self.recognition_canvas.winfo_width()
                self.recognition_canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        self.scrollable_frame.bind("<Configure>", _configure_scrollregion)
        self.recognition_canvas.bind("<Configure>", _configure_canvas_width)
        
        # ========== ŞİMDİ İÇERİK SCROLLABLE_FRAME'E EKLENİYOR ==========
        
        # Geri dön butonu
        back_btn = tk.Button(
            self.scrollable_frame,
            text="← Ana Menü",
            command=self.show_main_menu,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        back_btn.pack(anchor='nw', pady=(10, 10), padx=10)
        
        # Başlık
        title = tk.Label(
            self.scrollable_frame,
            text="🔍 YÜZ TANIMA",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        )
        title.pack(pady=(0, 20))
        
        # Fotoğraf yükleme alanı
        upload_frame = tk.Frame(self.scrollable_frame, bg='white', relief='solid', bd=1)
        upload_frame.pack(pady=10, padx=100, fill='x')
        
        upload_btn = tk.Button(
            upload_frame,
            text="📷 Fotoğraf Yükle",
            command=self.upload_recognition_photo,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 14, 'bold'),
            cursor='hand2',
            pady=20
        )
        upload_btn.pack(pady=30)
        
        # FOTOĞRAF GÖSTERIM ALANI
        photo_display_frame = tk.Frame(self.scrollable_frame, bg='white', relief='solid', bd=1)
        photo_display_frame.pack(pady=10, padx=50, fill='x')
        
        # Fotoğraf başlığı
        self.photo_title = tk.Label(
            photo_display_frame,
            text="📷 Yüklenen fotoğraf burada görünecek",
            font=('Arial', 11),
            bg='white',
            fg='#7f8c8d'
        )
        self.photo_title.pack(pady=10)
        
        # Fotoğraf labelı (resim gösterim için)
        self.photo_label = tk.Label(
            photo_display_frame,
            bg='white',
            text="",
            compound='center'
        )
        self.photo_label.pack(pady=10)
        
        # Tespit bilgisi - SCROLLABLE TEXT WIDGET
        detection_frame = tk.Frame(photo_display_frame, bg='white')
        detection_frame.pack(pady=5, fill='x', padx=10)
        
        # Scrollable text widget for detection info
        self.detection_info = tk.Text(
            detection_frame,
            height=4,  # 4 satır yükseklik
            font=('Arial', 10),
            bg='#f8f9fa',
            fg='#3498db',
            wrap=tk.WORD,  # Kelime bazında sarma
            relief='flat',
            bd=1,
            state='disabled',  # Kullanıcı düzenleyemesin
            cursor='arrow'
        )
        self.detection_info.pack(side='left', fill='both', expand=True)
        
        # Scroll bar
        detection_scrollbar = tk.Scrollbar(detection_frame, command=self.detection_info.yview)
        detection_scrollbar.pack(side='right', fill='y')
        self.detection_info.config(yscrollcommand=detection_scrollbar.set)
        
        # Sonuç alanı - ŞİMDİ SCROLLABLE_FRAME İÇİNDE
        self.result_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0', height=400)
        self.result_frame.pack(pady=20, padx=10, fill='both', expand=True)
        
        # Başlangıç mesajı
        initial_result_label = tk.Label(
            self.result_frame,
            text="📋 TANIMA SONUÇLARI\n\n Yukarıdan fotoğraf yükleyerek tanıma işlemini başlatın.\nSonuçlar burada görüntülenecek ve scroll ile incelenebilir.",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#7f8c8d',
            justify='center'
        )
        initial_result_label.pack(expand=True, pady=50)
        
        # Scroll region'ı düzenli olarak güncelle
        def _auto_update_scroll():
            try:
                if hasattr(self, 'scrollable_frame') and self.scrollable_frame and self.scrollable_frame.winfo_exists():
                    self.scrollable_frame.update_idletasks()
                    _configure_scrollregion()
                    # 100ms sonra tekrar kontrol et
                    self.root.after(100, _auto_update_scroll)
            except Exception as e:
                # Scroll güncelleme hatası, sessizce devam et
                pass
        
        # İlk scroll region ayarla ve otomatik güncellemeyi başlat
        self.scrollable_frame.update_idletasks()
        _configure_scrollregion()
        _auto_update_scroll()
    
    def show_student_list(self):
        """Öğrenci listesi ekranını gösterir"""
        self.clear_main_frame()
        self.current_mode = "list"
        
        # Geri dön butonu
        back_btn = tk.Button(
            self.main_frame,
            text="← Ana Menü",
            command=self.show_main_menu,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        back_btn.pack(anchor='nw', pady=(0, 10))
        
        # Başlık
        title = tk.Label(
            self.main_frame,
            text="📋 KAYITLI ÖĞRENCİLER",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        )
        title.pack(pady=(0, 20))
        
        # Tablo frame
        table_frame = tk.Frame(self.main_frame, bg='white')
        table_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Treeview
        columns = ('ID', 'Ad', 'Sınıf', 'Fotoğraf Sayısı')
        self.student_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Sütun genişliklerini ayarla
        self.student_tree.heading('ID', text='ID')
        self.student_tree.column('ID', width=120)
        
        self.student_tree.heading('Ad', text='Ad')
        self.student_tree.column('Ad', width=200)
        
        self.student_tree.heading('Sınıf', text='Sınıf')
        self.student_tree.column('Sınıf', width=120)
        
        self.student_tree.heading('Fotoğraf Sayısı', text='Fotoğraf Sayısı')
        self.student_tree.column('Fotoğraf Sayısı', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.student_tree.yview)
        self.student_tree.configure(yscrollcommand=scrollbar.set)
        
        self.student_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # BUTON ALANI - Silme butonu
        button_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20, padx=20, fill='x')
        
        # Silme butonu
        delete_btn = tk.Button(
            button_frame,
            text="🗑️ Seçili Öğrenciyi Sil",
            command=self.delete_selected_student,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            width=20
        )
        delete_btn.pack(side='left', padx=10)
        
        # Yenile butonu
        refresh_btn = tk.Button(
            button_frame,
            text="🔄 Listeyi Yenile",
            command=self.load_student_list,
            bg='#3498db',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            width=15
        )
        refresh_btn.pack(side='left', padx=10)
        
        # Kalite raporu butonu
        quality_report_btn = tk.Button(
            button_frame,
            text="📊 Kalite Raporu",
            command=self.show_quality_report,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            width=15
        )
        quality_report_btn.pack(side='left', padx=10)
        
        # Bilgi etiketi
        info_label = tk.Label(
            button_frame,
            text="💡 Silmek için öğrenciyi seçin ve 'Sil' butonuna tıklayın",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        info_label.pack(side='right', padx=10)
        
        # Verileri yükle
        self.load_student_list()
    
    def select_photos(self):
        """Fotoğraf seçim dialogunu açar"""
        files = filedialog.askopenfilenames(
            title="5 adet öğrenci fotoğrafı seçin",
            filetypes=[("Resim dosyaları", "*.jpg *.jpeg *.png *.bmp")],
            multiple=True
        )
        
        if files:
            if len(files) > 5:
                messagebox.showwarning("Uyarı", "En fazla 5 fotoğraf seçebilirsiniz!")
                files = files[:5]
            
            self.selected_photos = list(files)
            self.update_photo_list()
            
            # Fotoğrafları seçtikten sonra kalite kontrolü yap
            if self.face_processor and len(self.selected_photos) > 0:
                self.update_status("🔄 Fotoğraf kalitesi kontrol ediliyor...")
                threading.Thread(target=self._analyze_selected_photos, daemon=True).start()
    
    def _analyze_selected_photos(self):
        """Seçilen fotoğrafları analiz eder"""
        try:
            photo_analyses = []
            
            for i, photo_path in enumerate(self.selected_photos, 1):
                self.root.after(0, lambda i=i: self.update_status(f"🔄 Fotoğraf {i}/{len(self.selected_photos)} analiz ediliyor..."))
                
                try:
                    # Yüz tespiti
                    faces = self.face_processor.detect_faces(photo_path)
                    
                    if not faces:
                        photo_analyses.append({
                            'path': photo_path,
                            'status': 'no_face',
                            'quality': None,
                            'message': 'Yüz bulunamadı'
                        })
                        continue
                    
                    # En iyi yüzü seç
                    best_face = max(faces, key=lambda x: x['det_score'])
                    
                    # Kalite kontrolü
                    quality = self.face_processor.check_face_quality(
                        photo_path, 
                        best_face['bbox'], 
                        best_face.get('landmark')
                    )
                    
                    # Durumu belirle (dengeli yaklaşım: kayıt kolaylaştır, tanıma katı)
                    if quality['summary']['total_passed'] >= 3 and quality['overall_quality'] >= 0.60:
                        status = 'good'
                        message = f"Kaliteli ({quality['summary']['total_passed']}/5 kriter başarılı)"
                    else:
                        status = 'poor'
                        failed = ", ".join(quality['summary']['failed_checks'])
                        message = f" Düşük kalite - Sorunlar: {failed}"
                    
                    photo_analyses.append({
                        'path': photo_path,
                        'status': status,
                        'quality': quality,
                        'message': message
                    })
                    
                except Exception as e:
                    photo_analyses.append({
                        'path': photo_path,
                        'status': 'error',
                        'quality': None,
                        'message': f'Hata: {str(e)}'
                    })
            
            # Sonuçları GUI'de göster
            self.root.after(0, lambda: self._show_photo_quality_results(photo_analyses))
            self.root.after(0, lambda: self.update_status(" Fotoğraf analizi tamamlandı"))
            
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f" Analiz hatası: {e}"))
    
    def _show_photo_quality_results(self, analyses):
        """Fotoğraf kalite sonuçlarını gösterir"""
        # Önceki sonuçları temizle
        self.photos_listbox.delete(0, tk.END)
        
        # Yeni sonuçları ekle
        good_count = 0
        for analysis in analyses:
            filename = os.path.basename(analysis['path'])
            
            if analysis['status'] == 'good':
                good_count += 1
                # Yeşil renkte göster
                self.photos_listbox.insert(tk.END, f" {filename}")
                # Detaylı bilgi için tooltip ekle
                self.photos_listbox.insert(tk.END, f"    {analysis['message']}")
            elif analysis['status'] == 'poor':
                # Sarı renkte göster
                self.photos_listbox.insert(tk.END, f" {filename}")
                self.photos_listbox.insert(tk.END, f"    {analysis['message']}")
            elif analysis['status'] == 'no_face':
                # Kırmızı renkte göster
                self.photos_listbox.insert(tk.END, f" {filename}")
                self.photos_listbox.insert(tk.END, f"    {analysis['message']}")
            else:  # error
                self.photos_listbox.insert(tk.END, f" {filename}")
                self.photos_listbox.insert(tk.END, f"    {analysis['message']}")
        
        # Özet bilgiyi güncelle
        total_photos = len(analyses)
        poor_count = total_photos - good_count
        
        if poor_count > 0:
            # Herhangi bir fotoğraf sorunluysa kayıt yapılamaz
            self.photo_count_label.config(
                text=f" {total_photos} fotoğraf - {poor_count} SORUNLU VAR! (KAYIT YAPILAMAZ!)",
                foreground='red'
            )
        elif good_count == total_photos and total_photos > 0:
            # Tüm fotoğraflar kaliteli - mükemmel!
            self.photo_count_label.config(
                text=f" {total_photos} fotoğraf - HEPSİ KALİTELİ! (KAYIT HAZİR!)",
                foreground='darkgreen'
            )
        else:
            # Hiç fotoğraf yok veya başka durum
            self.photo_count_label.config(
                text=f" Fotoğraf analizi tamamlanamadı",
                foreground='red'
            )
        
        # Kalite detaylarını ayrı pencerede göster
        if any(a['quality'] for a in analyses):
            self._show_quality_details_window(analyses)
    
    def _show_quality_details_window(self, analyses):
        """Kalite detaylarını ayrı pencerede gösterir"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(" Fotoğraf Kalite Detayları")
        detail_window.geometry("800x600")
        detail_window.configure(bg='#f0f0f0')
        
        # Başlık
        title_label = tk.Label(
            detail_window,
            text=" FOTOĞRAF KALİTE ANALİZİ",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)
        
        # Scroll edilebilir metin alanı
        text_frame = tk.Frame(detail_window, bg='white')
        text_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        text_area = scrolledtext.ScrolledText(
            text_frame,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#333',
            wrap=tk.WORD
        )
        text_area.pack(fill='both', expand=True)
        
        # Detaylı rapor oluştur
        report = " FOTOĞRAF KALİTE RAPORU\n"
        report += "=" * 60 + "\n\n"
        
        for i, analysis in enumerate(analyses, 1):
            filename = os.path.basename(analysis['path'])
            report += f"📸 FOTOĞRAF {i}: {filename}\n"
            report += "-" * 40 + "\n"
            
            if analysis['quality']:
                quality = analysis['quality']
                details = quality['details']
                
                report += f"Genel Skor: {quality['overall_quality']:.2f}/1.00\n"
                report += f"Başarılı Kriterler: {quality['summary']['total_passed']}/5\n\n"
                
                report += " Detaylı Analiz:\n"
                report += f"  1️⃣ Yüz Netliği: {details['sharpness']['message']}\n"
                report += f"     Skor: {details['sharpness']['score']:.2f}\n"
                if 'metrics' in details['sharpness']:
                    metrics = details['sharpness']['metrics']
                    report += f"     Laplacian: {metrics.get('laplacian_variance', 0):.1f}\n"
                
                report += f"\n  2️⃣ Gözler: {details['eyes_open']['message']}\n"
                report += f"     Skor: {details['eyes_open']['score']:.2f}\n"
                
                report += f"\n  3️⃣ Açı: {details['face_angle']['message']}\n"
                report += f"     Skor: {details['face_angle']['score']:.2f}\n"
                if 'metrics' in details['face_angle'] and 'yaw_angle' in details['face_angle']['metrics']:
                    metrics = details['face_angle']['metrics']
                    report += f"     Yatay açı: {metrics.get('yaw_angle', 0):.1f}°\n"
                    report += f"     Dikey açı: {metrics.get('pitch_angle', 0):.1f}°\n"
                
                report += f"\n  4️⃣ Yüz Bütünlüğü: {details['face_integrity']['message']}\n"
                report += f"     Skor: {details['face_integrity']['score']:.2f}\n"
                
                report += f"\n  5️⃣ Işık: {details['lighting']['message']}\n"
                report += f"     Skor: {details['lighting']['score']:.2f}\n"
                if 'metrics' in details['lighting']:
                    metrics = details['lighting']['metrics']
                    report += f"     Parlaklık: {metrics.get('mean_brightness', 0):.1f}\n"
                    report += f"     Kontrast: {metrics.get('contrast', 0):.1f}\n"
                
                # Sonuç (dengeli yaklaşım)
                if quality['summary']['total_passed'] >= 3:
                    report += "\nSONUÇ: Fotoğraf kayıt için uygun\n"
                else:
                    report += "\n  SONUÇ: Fotoğraf kalitesi yetersiz\n"
                    report += f"Sorunlar: {', '.join(quality['summary']['failed_checks'])}\n"
            else:
                report += f"Analiz yapılamadı: {analysis['message']}\n"
            
            report += "\n" + "=" * 60 + "\n\n"
        
        # Genel özet
        good_photos = sum(1 for a in analyses if a['status'] == 'good')
        total_photos = len(analyses)
        
        report += "GENEL ÖZET\n"
        report += "-" * 20 + "\n"
        report += f"Toplam Fotoğraf: {total_photos}\n"
        report += f"Kaliteli Fotoğraf: {good_photos}\n"
        report += f"Sorunlu Fotoğraf: {total_photos - good_photos}\n"
        report += f"Başarı Oranı: {(good_photos/total_photos)*100:.1f}%\n"
        
        poor_photos = total_photos - good_photos
        
        if poor_photos > 0:
            report += f"\nKAYIT YAPILAMAZ - {poor_photos} fotoğraf sorunlu!"
            report += "\n YENİ KURAL: TÜM fotoğraflar kaliteli olmalıdır!"
            report += "\n Sorunlu fotoğrafları değiştirip tekrar deneyin."
        elif good_photos == total_photos and total_photos > 0:
            report += "\nMÜKEMMEL - TÜM fotoğraflar kaliteli!"
            report += "\nÖğrenci kaydı için hazır!"
        else:
            report += "\nHATA - Fotoğraf analizi tamamlanamadı!"
        
        text_area.insert(tk.END, report)
        text_area.config(state='disabled')  # Sadece okunabilir yap
        
        # Kapat butonu
        close_btn = tk.Button(
            detail_window,
            text="Kapat",
            command=detail_window.destroy,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        close_btn.pack(pady=10)
    
    def update_photo_list(self):
        """Seçilen fotoğraf listesini günceller"""
        self.photos_listbox.delete(0, tk.END)
        for photo in self.selected_photos:
            filename = os.path.basename(photo)
            self.photos_listbox.insert(tk.END, filename)
        
        count = len(self.selected_photos)
        self.photo_count_label.config(text=f"{count}/5 fotoğraf seçildi")
    
    def register_student(self):
        """Öğrenci kaydını yapar"""
        name = self.name_entry.get().strip()
        student_id = self.student_id_entry.get().strip()
        student_class = self.student_class_entry.get().strip()
        
        if not name or not student_id:
            messagebox.showerror("Hata", "Öğrenci adı ve ID alanları zorunludur!")
            return
        
        if len(self.selected_photos) < 1:
            messagebox.showerror("Hata", "En az 1 fotoğraf seçmelisiniz!")
            return
        
        if len(self.selected_photos) < 3:
            result = messagebox.askyesno(
                "Az Sayıda Fotoğraf",
                f"Sadece {len(self.selected_photos)} fotoğraf seçildi!\n\n"
                "ÖNERİLEN: En az 3-4 fotoğraf (daha iyi doğruluk için)\n"
                "İdeal: 4-5 fotoğraf\n"
                "NOT: Tanıma aşamasında çoklu doğrulama kullanılır\n\n"
                "Az sayıda fotoğrafla devam etmek istediğinizden emin misiniz?\n"
                "(Daha fazla fotoğraf = daha iyi tanıma)"
            )
            if not result:
                return
        
        if not self.face_processor:
            messagebox.showerror("Hata", "Yüz işleme modeli henüz hazır değil!\nLütfen modellerin yüklenmesini bekleyin.")
            return
        
        # İşlemi ayrı thread'de yap
        self.update_status("🔄 Öğrenci kaydediliyor...")
        threading.Thread(target=self._process_student_registration, args=(name, student_id, student_class), daemon=True).start()
    
    def _process_student_registration(self, name, student_id, student_class):
        """Öğrenci kayıt işlemini yapar"""
        try:
            # Fotoğrafları işle
            processed_faces = self.face_processor.process_student_photos(self.selected_photos)
            
            # KATIT KONTROL: TÜM FOTOĞRAFLAR KALİTELİ OLMALI
            total_photos = len(self.selected_photos)
            accepted_photos = len(processed_faces)
            rejected_photos = total_photos - accepted_photos
            
            if rejected_photos > 0:
                # Herhangi bir fotoğraf sorunluysa kayıt yapma
                self.root.after(0, lambda: messagebox.showerror(
                    "Sorunlu Fotoğraf Tespit Edildi", 
                    f"{rejected_photos}/{total_photos} fotoğraf kalite kontrolünden geçemedi!\n\n"
                    f" Seçilen fotoğraf: {total_photos}\n"
                    f"Kaliteli fotoğraf: {accepted_photos}\n"
                    f"Sorunlu fotoğraf: {rejected_photos}\n\n"
                    "KAYIT ŞARTI: TÜM fotoğraflar kaliteli olmalıdır!\n\n"
                    "Lütfen aşağıdaki kriterlere uygun fotoğraflar seçin:\n"
                    "• Net ve keskin fotoğraflar\n"
                    "• Gözlerin açık olduğu pozlar\n"
                    "• Frontal açıdan çekilmiş (baş eğimi olmayan)\n"
                    "• Yüzün tamamen görünür olduğu\n"
                    "• Yeterli ışığa sahip\n\n"
                    "Sorunlu fotoğrafları değiştirip tekrar deneyin.\n"
                    "Detaylı analiz raporunu kontrol ederek hangi fotoğrafların\n"
                    "sorunlu olduğunu görebilirsiniz."
                ))
                self.root.after(0, lambda: self.update_status(f" Kayıt başarısız - {rejected_photos} sorunlu fotoğraf var"))
                return
            
            # Hiç kaliteli fotoğraf yoksa
            if accepted_photos == 0:
                self.root.after(0, lambda: messagebox.showerror(
                    "Hiç Kaliteli Fotoğraf Yok", 
                    " Hiçbir fotoğraf kalite kontrolünden geçemedi!\n\n"
                    "📋Lütfen daha kaliteli fotoğraflar seçin ve tekrar deneyin."
                ))
                self.root.after(0, lambda: self.update_status(" Kayıt başarısız - Hiç kaliteli fotoğraf yok"))
                return
            
            # Tüm fotoğraflar kaliteli - kayıt yap
            self.root.after(0, lambda: self.update_status(f" Tüm fotoğraflar kaliteli! Kayıt yapılıyor..."))
            self._complete_student_registration(name, student_id, student_class, processed_faces)
            
        except ValueError as e:
            self.root.after(0, lambda: messagebox.showerror("Hata", str(e)))
            self.root.after(0, lambda: self.update_status("Kayıt başarısız"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Hata", f"Beklenmeyen hata: {e}"))
            self.root.after(0, lambda: self.update_status("Kayıt başarısız"))
    
    def _complete_student_registration(self, name, student_id, student_class, processed_faces):
        """Öğrenci kayıt işlemini tamamlar (thread-safe)"""
        try:
            # Veritabanına kaydet
            student_pk = self.db_manager.add_student(name, student_id, student_class)
            
            for face_data in processed_faces:
                embedding = face_data['face_data']['embedding']
                quality_score = face_data['quality']['overall_quality']
                quality_details = face_data['quality']  # Tüm kalite detayları
                photo_path = face_data['image_path']
                
                # Formatlanmış kalite raporu oluştur
                quality_report = self.db_manager.generate_formatted_quality_report(photo_path, quality_details)
                
                self.db_manager.add_face_embedding(student_pk, embedding, photo_path, quality_score, quality_details, quality_report)
            
            # UI güncellemelerini main thread'de yap
            avg_quality = sum(f['quality']['overall_quality'] for f in processed_faces) / len(processed_faces)
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Başarılı Kayıt", 
                f" {name} başarıyla kaydedildi!\n\n"
                f" {len(processed_faces)} fotoğraf eklendi (TÜMÜ KALİTELİ!)\n"
                f" Ortalama kalite skoru: {avg_quality:.2f}\n"
                f" Tüm fotoğraflar kalite kontrolünden başarıyla geçti!\n\n"
                " Öğrenci artık yüz tanıma sistemi ile tanınabilir!"
            ))
            self.root.after(0, lambda: self.update_status(" Kayıt başarıyla tamamlandı"))
            self.root.after(0, self._clear_registration_form)
            
        except ValueError as e:
            self.root.after(0, lambda: messagebox.showerror("Veritabanı Hatası", str(e)))
            self.root.after(0, lambda: self.update_status(" Kayıt başarısız - Veritabanı hatası"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Beklenmeyen Hata", f"Kayıt sırasında hata oluştu: {e}"))
            self.root.after(0, lambda: self.update_status(" Kayıt başarısız"))
    
    def upload_recognition_photo(self):
        """FOTOĞRAF YÜKLE VE GÖSTER - Gelişmiş hata kontrolü ile"""
        file_path = filedialog.askopenfilename(
            title="Tanınacak fotoğrafı seçin",
            filetypes=[
                ("Tüm Desteklenen", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG dosyaları", "*.jpg *.jpeg"),
                ("PNG dosyaları", "*.png"),
                ("BMP dosyaları", "*.bmp"),
                ("TIFF dosyaları", "*.tiff"),
                ("WebP dosyaları", "*.webp"),
                ("Tüm dosyalar", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        if not self.face_processor:
            messagebox.showerror(
                "Model Hatası", 
                "🤖 Yüz tanıma modeli henüz yüklenmedi.\n\n"
                "⏳ Lütfen modellerin yüklenmesini bekleyin."
            )
            return
        
        # AĞLAMLANAN DOSYA KONTROLÜ
        try:
            # Dosya mevcut mu?
            if not os.path.exists(file_path):
                messagebox.showerror(
                    "Dosya Hatası",
                    f" Seçilen dosya bulunamadı:\n{file_path}\n\n"
                    " Dosyanın taşınmadığından emin olun."
                )
                return
            
            # Dosya boyutu kontrolü
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                messagebox.showerror(
                    "Dosya Hatası",
                    " Seçilen dosya boş.\n\n"
                    " Başka bir fotoğraf seçin."
                )
                return
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                response = messagebox.askyesno(
                    "Büyük Dosya Uyarısı",
                    f" Seçilen dosya çok büyük:\n"
                    f" Boyut: {file_size/1024/1024:.1f}MB\n\n"
                    f"⏱ İşlem uzun sürebilir.\n"
                    f"Devam etmek istiyor musunuz?"
                )
                if not response:
                    return
            
            # Dosya formatı kontrolü
            file_extension = os.path.splitext(file_path)[1].lower()
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            if file_extension not in supported_formats:
                messagebox.showerror(
                    "Format Hatası",
                    f" Desteklenmeyen dosya formatı: {file_extension}\n\n"
                    f" Desteklenen formatlar:\n"
                    f"• JPG/JPEG\n• PNG\n• BMP\n• TIFF\n• WebP\n\n"
                    f" Fotoğrafı desteklenen formatta kaydedin."
                )
                return
            
            # Fotoğrafı kaydet
            self.current_recognition_photo = file_path
            
            # İlk dosya bilgilerini göster
            self.update_status(f"📁 Dosya: {os.path.basename(file_path)} ({file_size/1024:.1f}KB)")
            
            # İlk önce fotoğrafı göster (yüz işaretleme olmadan)
            self.root.after(0, lambda: self.display_initial_photo(file_path))
            
            self.update_status("🔄 Yüzler tespit ediliyor...")
            threading.Thread(target=self._process_face_recognition, args=(file_path,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror(
                "Beklenmeyen Hata",
                f" Dosya işleme hatası:\n{str(e)}\n\n"
                " Başka bir fotoğraf deneyin."
            )
    
    def display_initial_photo(self, image_path):
        """İlk yükleme sırasında fotoğrafı gösterir (yüz işaretlemesi olmadan)"""
        try:
            # Orijinal fotoğrafı göster
            pil_image = Image.open(image_path)
            pil_image = self.resize_image_for_display(pil_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # GUI'de göster
            self.photo_label.configure(image=photo, text="")
            self.photo_label.image = photo
            
            # Başlık güncelle - SAFE CHECK
            if hasattr(self, 'photo_title') and self.photo_title is not None:
                try:
                    filename = os.path.basename(image_path)
                    self.photo_title.configure(
                        text=f"📷 {filename} - Yüzler tespit ediliyor...",
                        fg='#f39c12'
                    )
                except tk.TclError:
                    # Widget destroyed
                    self.photo_title = None
            
            # Bekleme mesajı - SAFE
            self._safe_update_text_widget('detection_info',
                "🔄 Yüz tespiti yapılıyor, lütfen bekleyin...\n\n⏳ İşlemler:\n• Fotoğraf analiz ediliyor\n• Yüzler tespit ediliyor\n• Veritabanı ile karşılaştırılıyor",
                fg='#f39c12'
            )
            
        except Exception as e:
            print(f"İlk fotoğraf gösterim hatası: {e}")
            self._safe_update_text_widget('detection_info',
                f" Fotoğraf yüklenemedi: {e}\n\n💡 Çözüm önerileri:\n• Dosya formatını kontrol edin (JPG, PNG)\n• Dosya boyutunu kontrol edin\n• Başka bir fotoğraf deneyin",
                fg='#e74c3c'
            )
    
    def _process_face_recognition(self, image_path):
        """ÇOKLU YÜZ + İSİM ETİKETLEME DESTEKLİ tanıma işlemi"""
        try:
            # Yüz tespit et
            faces = self.face_processor.detect_faces(image_path)
            
            # Tespit edilen yüzleri kaydet
            self.detected_faces = faces
            
            if not faces:
                # Yüz yoksa boş göster
                self.root.after(0, lambda: self.display_photo_with_faces(image_path, faces))
                self.root.after(0, lambda: self._show_recognition_result("❌ Fotoğrafta yüz bulunamadı", None, None))
                return
            
            face_count = len(faces)
            print(f" {face_count} yüz tespit edildi - tümü test ediliyor...")
            
            # Veritabanındaki yüzlerle karşılaştır
            db_embeddings = self.db_manager.get_all_embeddings()
            
            if not db_embeddings:
                # Veritabanı boşsa sadece yüzleri göster (isim olmadan)
                self.root.after(0, lambda: self.display_photo_with_faces(image_path, faces))
                self.root.after(0, lambda: self._show_recognition_result("ℹ️ Veritabanında kayıtlı öğrenci yok", None, None))
                return
            
            # TÜM YÜZLERİ TEST ET VE İSİM ETİKETLERİNİ HAZIRLA
            best_match = None
            best_similarity = 0.0
            face_matches = []
            
            # Veritabanı bilgisi
            unique_names = list(set([name for _, name, _ in db_embeddings]))
            status_msg = f"Tanıma için hazır: {len(unique_names)} kişi kayıtlı"
            print(status_msg)
            self.update_status(status_msg)
            
            for i, face in enumerate(faces, 1):
                embedding = face['embedding']
                det_score = face['det_score']
                
                print(f" Yüz {i}/{face_count} analiz ediliyor...")
                
                # AKILLI THRESHOLD SİSTEMİ - Grup fotoğrafları için özel threshold
                # Çoklu yüz tespit edildiğinde daha toleranslı threshold kullan
                if face_count >= 5:  # Grup fotoğrafı tespit edildi
                    adaptive_threshold = 0.25  # %25 - Grup fotoğrafları için
                    print(f"🎭 GRUP FOTOĞRAFI TESPİT EDİLDİ ({face_count} yüz) → Threshold: %25")
                else:
                    adaptive_threshold = 0.55  # %55 - Normal threshold
                    print(f"👤 TEK/AZ YÜZ TESPİT EDİLDİ ({face_count} yüz) → Threshold: %55")
                
                match = self.face_processor.find_best_match(embedding, db_embeddings, threshold=adaptive_threshold, face_count=face_count)
                
                if match:
                    student_id, name, similarity = match
                    print(f" Tanındı: {name} (%{similarity:.1%})")
                    
                    match_data = {
                        'face_index': i,
                        'student_id': student_id,
                        'name': name,
                        'similarity': similarity,
                        'det_score': det_score
                    }
                    
                    # Duygu analizi ekle (eğer etkinse)
                    if self.face_processor.emotion_analysis_enabled:
                        try:
                            emotion_result = self.face_processor.analyze_emotion(image_path, tuple(face["bbox"]))
                            match_data['emotion_analysis'] = emotion_result
                            if emotion_result['success']:
                                print(f" Duygu: {emotion_result['dominant_emotion']} (%{emotion_result['dominant_score']:.1f})")
                        except Exception as e:
                            print(f"  Duygu analizi hatası: {e}")
                            match_data['emotion_analysis'] = {'success': False, 'message': str(e)}
                    
                    face_matches.append(match_data)
                    
                    # En iyi eşleşmeyi güncelle
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = match
                else:
                    print(f" Yüz {i}: Tanınmadı")
            
            # FOTOĞRAFI İSİM ETİKETLERİYLE BİRLİKTE GÖSTER
            self.root.after(0, lambda: self.display_photo_with_faces(image_path, faces, face_matches))
            
            # SONUCU RAPOR ET
            if face_matches:  # Eğer herhangi bir eşleşme varsa
                if len(face_matches) > 1:
                    # Birden fazla eşleşme varsa detaylı rapor
                    # En iyi eşleşmeyi bul
                    best_face_match = max(face_matches, key=lambda x: x['similarity'])
                    
                    matches_info = "\n".join([
                        f"🔢 Yüz Numarası: {m['face_index']} → 👤 {m['name']} → 🎯 {m['similarity']:.1%}"
                        for i, m in enumerate(face_matches)
                    ])
                    
                    # Tanınamayan yüz numaralarını hesapla
                    recognized_face_numbers = [m['face_index'] for m in face_matches]
                    all_face_numbers = list(range(1, face_count + 1))
                    unrecognized_face_numbers = [str(num) for num in all_face_numbers if num not in recognized_face_numbers]
                    
                    # Tanınamayan yüzler varsa göster
                    unrecognized_section = ""
                    if unrecognized_face_numbers:
                        unrecognized_list = ", ".join(unrecognized_face_numbers)
                        unrecognized_section = f"🔢 TANINAMAYAN YÜZ NUMARALARI:\n{unrecognized_list}\n\n"
                        # Manuel kayıt butonları ekle
                        self.root.after(0, lambda: self._create_manual_registration_buttons(unrecognized_face_numbers))
                    
                    result_message = (
                        f"🎉 BAŞARILI TANIMA!\n"
                        f"════════════════════════════════════════\n\n"
                        f"📊 ÖZET:\n"
                        f"• Taranan yüz sayısı: {face_count}\n"
                        f"• Tanınan öğrenci sayısı: {len(face_matches)}\n"
                        f"• Bilinmeyen yüz sayısı: {face_count - len(face_matches)}\n\n"
                        f"🏆 EN İYİ EŞLEŞME:\n"
                        f"👤 İsim: {best_face_match['name']}\n"
                        f"🎯 Benzerlik: {best_face_match['similarity']:.2%}\n"
                        f"🔢 Yüz Numarası: {best_face_match['face_index']}\n"
                        f"📍 Konum: Fotoğrafta yeşil kare ile işaretli\n\n"
                        f"📋 TÜM EŞLEŞMELER:\n"
                        f"────────────────────────────────────────\n"
                        f"{matches_info}\n\n"
                        f"{unrecognized_section}"
                        f"💡 İpucu: Fotoğrafta kırmızı isimler tanınan,\n"
                        f"   turuncu 'Bilinmeyen' yazıları tanınamayan yüzleri gösterir.\n\n"
                        f"🔧 DEBUG INFO:\n"
                        f"• Base threshold: %55\n"
                        f"• Multi-match avg: %58, min: %52\n"
                        f"• Detaylı skorlar konsol'da ve log'da"
                    )
                    
                    self.root.after(0, lambda: self._show_recognition_result(result_message, best_face_match['name'], best_face_match['similarity']))
                else:
                    # Tek eşleşme için
                    single_match = face_matches[0]
                    face_number = single_match['face_index']
                    
                    # Eğer fotoğrafta birden fazla yüz varsa tanınamayan yüzleri de göster
                    unrecognized_section = ""
                    if face_count > 1:
                        all_face_numbers = list(range(1, face_count + 1))
                        unrecognized_face_numbers = [str(num) for num in all_face_numbers if num != face_number]
                        if unrecognized_face_numbers:
                            unrecognized_list = ", ".join(unrecognized_face_numbers)
                            unrecognized_section = f"🔢 Tanınamayan Yüz Numaraları: {unrecognized_list}\n"
                            # Manuel kayıt butonları ekle
                            self.root.after(0, lambda: self._create_manual_registration_buttons(unrecognized_face_numbers))
                    
                    result_message = (
                        f" BAŞARILI TANIMA!\n"
                        f"════════════════════════════════════════\n\n"
                        f" İsim: {single_match['name']}\n"
                        f" Benzerlik: {single_match['similarity']:.2%}\n"
                        f" Tanınan Yüz Numarası: {face_number}\n"
                        f"Taranan yüz sayısı: {face_count}\n"
                        f" Eşleşme: 1 öğrenci tanındı\n"
                        f"{unrecognized_section}\n"
                        f" Konum: Fotoğrafta yeşil kare içinde\n"
                        f" İsim: Kırmızı renkte gösterildi\n\n"
                        f"İpucu: Tek yüz tanıma işlemi tamamlandı."
                    )
                    
                    self.root.after(0, lambda: self._show_recognition_result(result_message, single_match['name'], single_match['similarity']))
            else:
                # Tanınamayan yüz numaralarını listele
                unrecognized_faces = [str(i) for i in range(1, face_count + 1)]
                unrecognized_list = ", ".join(unrecognized_faces)
                
                # Manuel kayıt butonları ekle
                self.root.after(0, lambda: self._create_manual_registration_buttons(unrecognized_faces))
                
                no_match_message = (
                    f" TANINAMAYAN SONUÇ\n"
                    f"════════════════════════════════════════\n\n"
                    f" ÖZET:\n"
                    f"• Taranan yüz sayısı: {face_count}\n"
                    f"• Tanınan öğrenci sayısı: 0\n"
                    f"• Bilinmeyen yüz sayısı: {face_count}\n\n"
                    f" TANINAMAYAN YÜZ NUMARALARI:\n"
                    f"{unrecognized_list}\n\n"
                    f"🟠 DURUM:\n"
                    f"Fotoğraftaki hiçbir yüz veritabanında\n"
                    f"kayıtlı öğrencilerle eşleşmedi.\n\n"
                    f" FOTOĞRAFTA:\n"
                    f"Tüm yüzler turuncu 'Bilinmeyen' etiketiyle\n"
                    f"yeşil kareler içinde gösterildi.\n\n"
                    f" ÖNERİLER:\n"
                    f"• Öğrencileri önce sisteme kaydedin\n"
                    f"• Fotoğraf kalitesini kontrol edin\n"
                    f"• Yüzlerin açık görünüp görünmediğini kontrol edin\n"
                    f"• Aşağıdaki butonlar ile tanınamayan yüzleri manuel olarak kaydedin"
                )
                self.root.after(0, lambda: self._show_recognition_result(no_match_message, None, None))
            
            self.root.after(0, lambda: self.update_status("✅ Tanıma tamamlandı"))
            
        except Exception as e:
            error_str = str(e)
            print(f" Yüz tanıma hatası: {error_str}")
            
            # Hata tipine göre farklı mesajlar
            if "Görüntü okunamadı" in error_str or "GÖRÜNTÜ OKUMA HATASI" in error_str:
                # Görüntü okuma hatası - detaylı mesaj zaten face_processor'da var
                self.root.after(0, lambda: self._show_recognition_result(error_str, None, None))
                self.root.after(0, lambda: self.update_status(" Görüntü okuma hatası"))
            elif "FileNotFoundError" in str(type(e)) or "bulunamadı" in error_str:
                self.root.after(0, lambda: self._show_recognition_result(
                    f" DOSYA BULUNAMADI\n\n"
                    f"Aranan dosya: {os.path.basename(image_path)}\n\n"
                    f"ÇÖZÜM ÖNERİLERİ:\n"
                    f"• Dosyanın hala aynı konumda olduğunu kontrol edin\n"
                    f"• Dosya adını değiştirmediyseniz kontrol edin\n"
                    f"• Başka bir fotoğraf seçin\n\n"
                    f"🔧 Teknik detay: {error_str}", 
                    None, None
                ))
                self.root.after(0, lambda: self.update_status(" Dosya bulunamadı"))
            elif "Desteklenmeyen" in error_str:
                self.root.after(0, lambda: self._show_recognition_result(error_str, None, None))
                self.root.after(0, lambda: self.update_status(" Desteklenmeyen format"))
            elif "çok büyük" in error_str:
                self.root.after(0, lambda: self._show_recognition_result(error_str, None, None))
                self.root.after(0, lambda: self.update_status(" Dosya çok büyük"))
            else:
                # Genel hata
                self.root.after(0, lambda: self._show_recognition_result(
                    f" BEKLENMEYEN HATA\n\n"
                    f" Hata detayı: {error_str}\n\n"
                    f" ÇÖZÜM ÖNERİLERİ:\n"
                    f"• Başka bir fotoğraf deneyin\n"
                    f"• Uygulamayı yeniden başlatın\n"
                    f"• Fotoğrafı farklı formatta kaydedin\n"
                    f"• Sistem yöneticisine başvurun", 
                    None, None
                ))
                self.root.after(0, lambda: self.update_status(" Tanıma başarısız"))
    
    def _show_recognition_result(self, message, name, similarity):
        """Tanıma sonucunu scrollable text widget ile gösterir"""
        # Önceki sonuçları temizle
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Scrollable text frame
        result_container = tk.Frame(self.result_frame, bg='#f0f0f0')
        result_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = tk.Label(
            result_container,
            text="📋 TANIMA SONUÇLARI",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 10))
        
        # Scrollable text widget
        text_frame = tk.Frame(result_container, bg='#f0f0f0')
        text_frame.pack(fill='both', expand=True)
        
        result_text = tk.Text(
            text_frame,
            font=('Arial', 12),
            bg='white',
            fg='#2c3e50',
            wrap=tk.WORD,
            relief='solid',
            bd=1,
            padx=15,
            pady=15,
            state='disabled',
            cursor='arrow'
        )
        result_text.pack(side='left', fill='both', expand=True)
        
        # Scroll bar
        result_scrollbar = tk.Scrollbar(text_frame, command=result_text.yview)
        result_scrollbar.pack(side='right', fill='y')
        result_text.config(yscrollcommand=result_scrollbar.set)
        
        # Text içeriğini ekle
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.insert(1.0, message)
        
        # Renk ayarları (sonuca göre)
        if name:  # Tanıma başarılı
            result_text.tag_add("success", "1.0", "end")
            result_text.tag_config("success", foreground='#27ae60')
        else:  # Tanıma başarısız
            result_text.tag_add("error", "1.0", "end") 
            result_text.tag_config("error", foreground='#e74c3c')
        
        result_text.config(state='disabled')
        result_text.see(1.0)  # Başlangıca odaklan
    
    def delete_selected_student(self):
        """Seçili öğrenciyi siler"""
        selected_item = self.student_tree.selection()
        
        if not selected_item:
            messagebox.showwarning("Uyarı", "Lütfen silmek istediğiniz öğrenciyi seçin!")
            return
        
        # Seçili öğrencinin bilgilerini al
        item = selected_item[0]
        values = self.student_tree.item(item, 'values')
        student_id = values[0]
        student_name = values[1]
        photo_count = values[2]
        
        # ONAY DIALOG'U - Çok önemli!
        response = messagebox.askyesno(
            " ÖĞRENCİ SİLME ONAYI",
            f"Bu öğrenciyi silmek istediğinizden emin misiniz?\n\n"
            f" Öğrenci: {student_name}\n"
            f" ID: {student_id}\n"
            f" Fotoğraf Sayısı: {photo_count}\n\n"
            f" DİKKAT: Bu işlem GERİ ALINAMAZ!\n"
            f"Öğrenciye ait tüm fotoğraflar ve veriler silinecek.",
            icon='warning'
        )
        
        if not response:
            return
        
        # İkinci onay (güvenlik için)
        second_confirm = messagebox.askyesno(
            " SON ONAY",
            f"SON UYARI!\n\n"
            f"'{student_name}' adlı öğrenci ve {photo_count} fotoğrafı "
            f"kalıcı olarak silinecek.\n\n"
            f"Bu işlem GERİ ALINAMAZ!\n\n"
            f"Devam etmek istiyor musunuz?",
            icon='warning'
        )
        
        if not second_confirm:
            return
        
        # Silme işlemini yap
        try:
            self.update_status(f"🗑️ {student_name} siliniyor...")
            
            success = self.db_manager.delete_student(student_id)
            
            if success:
                messagebox.showinfo(
                    " SİLME BAŞARILI",
                    f" {student_name} başarıyla silindi!\n\n"
                    f" Silinen veriler:\n"
                    f"• Öğrenci kaydı\n"
                    f"• {photo_count} adet fotoğraf\n"
                    f"• Tüm yüz verileri\n\n"
                    f" Veritabanı güncellendi."
                )
                
                # Listeyi yenile
                self.load_student_list()
                self.update_status("Öğrenci başarıyla silindi")
                
            else:
                messagebox.showerror(
                    "SİLME BAŞARISIZ",
                    f" {student_name} silinemedi!\n\n"
                    f"Possible sebpler:\n"
                    f"• Öğrenci bulunamadı\n"
                    f"• Veritabanı hatası\n"
                    f"• Sistem hatası\n\n"
                    f"Lütfen tekrar deneyin."
                )
                self.update_status(" Öğrenci silme işlemi başarısız")
                
        except Exception as e:
            messagebox.showerror(
                " SİSTEM HATASI",
                f"Beklenmeyen hata oluştu:\n\n{e}\n\n"
                f"Lütfen uygulamayı yeniden başlatın."
            )
            self.update_status(" Silme işleminde sistem hatası")
    
    def load_student_list(self):
        """Öğrenci listesini yükler"""
        try:
            # Mevcut verileri temizle
            for item in self.student_tree.get_children():
                self.student_tree.delete(item)
            
            # Yeni verileri ekle
            students = self.db_manager.get_all_students()
            
            if not students:
                # Hiç öğrenci yoksa bilgi göster
                self.student_tree.insert('', 'end', values=("", "📝 Henüz kayıtlı öğrenci yok", "", "0"))
            else:
                for student_id, name, student_class, photo_count in students:
                    # Sınıf bilgisi boşsa "-" göster
                    display_class = student_class if student_class else "-"
                    self.student_tree.insert('', 'end', values=(student_id, name, display_class, photo_count))
            
            # Durum güncelle
            student_count = len(students)
            self.update_status(f" {student_count} kayıtlı öğrenci listelendi")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Öğrenci listesi yüklenirken hata oluştu: {e}")
            self.update_status(" Öğrenci listesi yüklenemedi")
    
    def show_quality_report(self):
        """Kalite raporunu gösterir"""
        try:
            # Seçili öğrenci var mı kontrol et
            selected_item = self.student_tree.selection()
            
            if selected_item:
                # Seçili öğrencinin kalite raporunu göster
                student_data = self.student_tree.item(selected_item[0])['values']
                if len(student_data) >= 3 and student_data[0]:  # Geçerli öğrenci verisi var mı
                    student_id = student_data[0]
                    student_name = student_data[1]
                    self._show_student_quality_report(student_id, student_name)
                else:
                    messagebox.showwarning("Uyarı", "Geçerli bir öğrenci seçin!")
            else:
                # Hiç öğrenci seçilmemişse sistem genel istatistiklerini göster
                self._show_general_quality_statistics()
                
        except Exception as e:
            messagebox.showerror("Hata", f"Kalite raporu gösterilirken hata oluştu: {e}")
    
    def _show_student_quality_report(self, student_id: str, student_name: str):
        """Belirli bir öğrencinin kalite raporunu gösterir"""
        try:
            # Öğrenci kalite verilerini al
            quality_reports = self.db_manager.get_student_quality_report(student_id)
            
            if not quality_reports:
                messagebox.showinfo("Bilgi", f"{student_name} için kalite verisi bulunamadı!")
                return
            
            # Yeni pencere oluştur
            report_window = tk.Toplevel(self.root)
            report_window.title(f" {student_name} - Kalite Raporu")
            report_window.geometry("800x600")
            report_window.resizable(True, True)
            
            # Ana frame
            main_frame = tk.Frame(report_window, bg='white')
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Başlık
            title_label = tk.Label(
                main_frame,
                text=f" {student_name} ({student_id}) - Detaylı Kalite Raporu",
                font=('Arial', 14, 'bold'),
                bg='white',
                fg='#2c3e50'
            )
            title_label.pack(pady=(0, 20))
            
            # Özet bilgiler
            avg_quality = sum(r['quality_score'] for r in quality_reports if r['quality_score']) / len(quality_reports)
            summary_text = f" Toplam Fotoğraf: {len(quality_reports)}   |   🎯 Ortalama Kalite: {avg_quality:.3f}   |   📅 Son Kayıt: {quality_reports[0]['created_at'][:16]}"
            
            summary_label = tk.Label(
                main_frame,
                text=summary_text,
                font=('Arial', 11),
                bg='#ecf0f1',
                fg='#34495e',
                pady=10
            )
            summary_label.pack(fill='x', pady=(0, 15))
            
            # Scrollable text area
            text_frame = tk.Frame(main_frame)
            text_frame.pack(fill='both', expand=True)
            
            text_area = scrolledtext.ScrolledText(
                text_frame,
                font=('Consolas', 10),
                bg='#f8f9fa',
                fg='#2c3e50',
                wrap=tk.WORD
            )
            text_area.pack(fill='both', expand=True)
            
            # Rapor içeriğini oluştur
            report_content = self._generate_detailed_quality_report(quality_reports, student_name)
            text_area.insert('1.0', report_content)
            text_area.config(state='disabled')  # Sadece okuma
            
        except Exception as e:
            messagebox.showerror("Hata", f"Öğrenci kalite raporu oluşturulurken hata: {e}")
    
    def _show_general_quality_statistics(self):
        """Genel sistem kalite istatistiklerini gösterir"""
        try:
            # Sistem kalite istatistiklerini al
            stats = self.db_manager.get_quality_statistics()
            
            if not stats or stats.get('total_photos', 0) == 0:
                messagebox.showinfo("Bilgi", "Henüz kalite verisi bulunamadı!")
                return
            
            # Yeni pencere oluştur
            stats_window = tk.Toplevel(self.root)
            stats_window.title("📊 Sistem Kalite İstatistikleri")
            stats_window.geometry("600x500")
            stats_window.resizable(True, True)
            
            # Ana frame
            main_frame = tk.Frame(stats_window, bg='white')
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # Başlık
            title_label = tk.Label(
                main_frame,
                text="📊 SİSTEM KALİTE İSTATİSTİKLERİ",
                font=('Arial', 16, 'bold'),
                bg='white',
                fg='#2c3e50'
            )
            title_label.pack(pady=(0, 30))
            
            # İstatistik kartları
            stats_frame = tk.Frame(main_frame, bg='white')
            stats_frame.pack(fill='both', expand=True)
            
            # Genel bilgiler
            general_info = f"""
🎯 GENEL BİLGİLER
{'='*50}
📸 Toplam Fotoğraf Sayısı: {stats['total_photos']}
📊 Ortalama Kalite Skoru: {stats['average_quality']:.3f}
⬆️ En Yüksek Kalite: {stats['max_quality']:.3f}
⬇️ En Düşük Kalite: {stats['min_quality']:.3f}

🏆 KALİTE DAĞILIMI
{'='*50}
🥇 Mükemmel (≥0.80): {stats['excellent_photos']} fotoğraf ({stats['quality_distribution']['excellent']}%)
🥈 İyi (0.60-0.79): {stats['good_photos']} fotoğraf ({stats['quality_distribution']['good']}%)
🥉 Düşük (<0.60): {stats['poor_photos']} fotoğraf ({stats['quality_distribution']['poor']}%)

💡 SİSTEM ÖNERİLERİ
{'='*50}
• Mükemmel kalite oranı: %{stats['quality_distribution']['excellent']:.1f}
• Sistem kalite başarısı: {'🟢 YÜKSEK' if stats['quality_distribution']['excellent'] > 70 else '🟡 ORTA' if stats['quality_distribution']['excellent'] > 50 else '🔴 DÜŞÜK'}
• Önerilen iyileştirme: {'Fotoğraf kalitesini artırın' if stats['average_quality'] < 0.7 else 'Mevcut kalite standardı uygun'}
"""
            
            text_area = scrolledtext.ScrolledText(
                stats_frame,
                font=('Consolas', 11),
                bg='#f8f9fa',
                fg='#2c3e50',
                wrap=tk.WORD
            )
            text_area.pack(fill='both', expand=True)
            text_area.insert('1.0', general_info)
            text_area.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Hata", f"Sistem istatistikleri gösterilirken hata: {e}")
    
    def _generate_detailed_quality_report(self, quality_reports: list, student_name: str) -> str:
        """Detaylı kalite raporu metni oluşturur"""
        try:
            report_lines = []
            report_lines.append(f"📊 {student_name} - DETAYLI KALİTE ANALİZ RAPORU")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            for i, report in enumerate(quality_reports, 1):
                photo_path = report['photo_path']
                quality_score = report['quality_score']
                quality_details = report['quality_details']
                quality_report = report.get('quality_report')  # Formatlanmış rapor
                created_at = report['created_at']
                
                report_lines.append(f"🖼️ FOTOĞRAF {i}: {photo_path.split('/')[-1] if photo_path else 'Bilinmeyen dosya'}")
                report_lines.append(f"📅 Eklenme Tarihi: {created_at}")
                report_lines.append("-" * 80)
                
                # Eğer formatlanmış rapor varsa onu kullan, yoksa eski yöntemi kullan
                if quality_report:
                    # Formatlanmış raporu ekle (başlık ve ayırıcıları kaldır)
                    formatted_lines = quality_report.split('\n')
                    # İlk 3 satırı atla (başlık ve ayırıcılar)
                    for line in formatted_lines[3:]:
                        if line.strip():
                            report_lines.append(f"    {line}")
                    
                elif quality_details:
                    # Eski yöntem - JSON'dan manuel oluştur
                    overall = quality_details.get('overall_quality', 0)
                    report_lines.append(f"    🎯 Genel Kalite Skoru: {quality_score:.3f}")
                    report_lines.append(f"    📊 Genel Değerlendirme: {overall:.3f}")
                    
                    # Detaylar
                    details = quality_details.get('details', {})
                    if details:
                        report_lines.append("    🔍 DETAYLI KRİTER ANALİZİ:")
                        
                        # Her kriter için detaylı bilgi
                        criteria_info = {
                            'sharpness': ('🔍 Netlik', 'Fotoğrafın ne kadar keskin olduğu'),
                            'eyes_open': ('👁️ Göz Durumu', 'Gözlerin açık olma durumu'),
                            'face_angle': ('📐 Yüz Açısı', 'Yüzün kameraya doğru dönük olması'),
                            'face_integrity': ('🧩 Yüz Bütünlüğü', 'Yüzün tamamen görünür olması'),
                            'lighting': ('💡 Aydınlatma', 'Işık kalitesi ve parlaklık')
                        }
                        
                        for criterion, (icon_name, description) in criteria_info.items():
                            if criterion in details:
                                crit_data = details[criterion]
                                score = crit_data.get('score', 0)
                                is_adequate = crit_data.get('is_adequate', False)
                                message = crit_data.get('message', 'Bilgi yok')
                                
                                status = "✅ GEÇTİ" if is_adequate else "❌ BAŞARISIZ"
                                report_lines.append(f"      {icon_name}: {score:.3f} - {status}")
                                report_lines.append(f"         {description}")
                                report_lines.append(f"         💬 {message}")
                                report_lines.append("")
                    
                    # Özet
                    summary = quality_details.get('summary', {})
                    if summary:
                        passed = summary.get('total_passed', 0)
                        total = summary.get('total_criteria', 5)
                        failed_checks = summary.get('failed_checks', [])
                        
                        report_lines.append(f"    📋 ÖZET: {passed}/{total} kriter başarılı")
                        if failed_checks:
                            report_lines.append(f"    ⚠️ Başarısız kriterler: {', '.join(failed_checks)}")
                
                else:
                    report_lines.append(f"    🎯 Genel Kalite Skoru: {quality_score:.3f}")
                    report_lines.append(f"    ❌ Detaylı kalite verisi mevcut değil")
                
                report_lines.append("=" * 80)
                report_lines.append("")
            
            # Genel değerlendirme
            avg_quality = sum(r['quality_score'] for r in quality_reports if r['quality_score']) / len(quality_reports)
            excellent_count = len([r for r in quality_reports if r['quality_score'] and r['quality_score'] >= 0.8])
            
            report_lines.append("🎯 GENEL DEĞERLENDİRME")
            report_lines.append("=" * 80)
            report_lines.append(f"📊 Ortalama Kalite: {avg_quality:.3f}")
            report_lines.append(f"🏆 Mükemmel Fotoğraf Sayısı: {excellent_count}/{len(quality_reports)}")
            report_lines.append(f"📈 Kalite Yüzdesi: %{(avg_quality * 100):.1f}")
            
            if avg_quality >= 0.8:
                report_lines.append("🎉 SONUÇ: Tüm fotoğraflar mükemmel kalitede!")
            elif avg_quality >= 0.6:
                report_lines.append("✅ SONUÇ: Fotoğraflar uygun kalitede.")
            else:
                report_lines.append("⚠️ SONUÇ: Fotoğraf kalitesi artırılabilir.")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"❌ Rapor oluşturulurken hata: {e}"
    
    def _clear_registration_form(self):
        """Kayıt formunu temizler"""
        self.name_entry.delete(0, tk.END)
        self.student_id_entry.delete(0, tk.END)
        self.student_class_entry.delete(0, tk.END)
        self.selected_photos = []
        self.update_photo_list()
    
    def clear_main_frame(self):
        """Ana frame'i temizler ve widget referanslarını temizler"""
        # Önce widget'ları destroy et
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Widget referanslarını temizle
        self._clear_widget_references()
    
    def _clear_widget_references(self):
        """GUI widget referanslarını temizler - arayüz kayboluyor bug'ını önler"""
        # Face recognition ekranı widget'ları
        if hasattr(self, 'photo_title'):
            self.photo_title = None
        if hasattr(self, 'photo_label'):
            self.photo_label = None
        if hasattr(self, 'detection_info'):
            self.detection_info = None
        if hasattr(self, 'recognition_canvas'):
            self.recognition_canvas = None
        if hasattr(self, 'scrollable_frame'):
            self.scrollable_frame = None
        if hasattr(self, 'canvas_window'):
            self.canvas_window = None
        
        # Registration ekranı widget'ları
        if hasattr(self, 'name_entry'):
            self.name_entry = None
        if hasattr(self, 'student_id_entry'):
            self.student_id_entry = None
        if hasattr(self, 'photo_info_label'):
            self.photo_info_label = None
        if hasattr(self, 'status_label_detail'):
            self.status_label_detail = None
        if hasattr(self, 'manual_register_btn'):
            self.manual_register_btn = None
        
        # Student list ekranı widget'ları
        if hasattr(self, 'student_tree'):
            self.student_tree = None
        if hasattr(self, 'student_count_label'):
            self.student_count_label = None
    
    def _safe_configure_widget(self, widget_name, **kwargs):
        """Widget'ı güvenli şekilde configure eder - destroyed widget hatasını önler"""
        if hasattr(self, widget_name) and getattr(self, widget_name) is not None:
            try:
                widget = getattr(self, widget_name)
                widget.configure(**kwargs)
                return True
            except tk.TclError:
                # Widget destroyed, reference'ı temizle
                setattr(self, widget_name, None)
                return False
        return False
    
    def _safe_update_text_widget(self, widget_name, text, **text_config):
        """Text widget'ını güvenli şekilde günceller - tk.Text için özel"""
        if hasattr(self, widget_name) and getattr(self, widget_name) is not None:
            try:
                widget = getattr(self, widget_name)
                # Text widget'ı güncelle
                widget.config(state='normal')  # Düzenlemeye izin ver
                widget.delete(1.0, tk.END)     # Eski text'i sil
                widget.insert(1.0, text)       # Yeni text'i ekle
                
                # Text renk ayarları
                if 'fg' in text_config:
                    widget.tag_add("all", "1.0", "end")
                    widget.tag_config("all", foreground=text_config['fg'])
                
                widget.config(state='disabled') # Tekrar salt okunur yap
                widget.see(tk.END)              # Son satırı göster
                return True
            except tk.TclError:
                # Widget destroyed, reference'ı temizle
                setattr(self, widget_name, None)
                return False
        return False
    
    def show_failed_registrations(self):
        """Başarısız kayıtları görüntüler"""
        self.clear_main_frame()
        self.current_mode = "failed_registrations"
        
        # Başlık
        title_label = tk.Label(
            self.main_frame,
            text="❌ BAŞARISIZ KAYITLAR",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Başarısız kayıtları al
        failed_registrations = self.db_manager.get_failed_registrations()
        
        if not failed_registrations:
            # Başarısız kayıt yok
            no_data_label = tk.Label(
                self.main_frame,
                text="✅ Henüz başarısız kayıt bulunmuyor!\nTüm fotoğraflar kalite kriterlerini geçti.",
                font=('Arial', 14),
                fg='#27ae60',
                bg='#f0f0f0'
            )
            no_data_label.pack(pady=50)
        else:
            # Başarısız kayıtlar var - scrollable liste
            # Ana container
            main_container = tk.Frame(self.main_frame, bg='#f0f0f0')
            main_container.pack(fill='both', expand=True, padx=20, pady=10)
            
            # Canvas for scrolling
            canvas = tk.Canvas(main_container, bg='#f0f0f0', highlightthickness=0)
            scrollbar = tk.Scrollbar(main_container, orient='vertical', command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Başarısız kayıtları listele
            for i, registration in enumerate(failed_registrations):
                self._create_failed_registration_card(scrollable_frame, registration, i + 1)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
        
        # Geri dön butonu
        back_btn = tk.Button(
            self.main_frame,
            text="← Ana Menü",
            command=self.show_main_menu,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        back_btn.pack(pady=20)
    
    def _create_failed_registration_card(self, parent, registration, index):
        """Başarısız kayıt kartı oluşturur"""
        # Ana kart frame
        card_frame = tk.Frame(parent, bg='white', relief='raised', bd=2)
        card_frame.pack(fill='x', padx=10, pady=5)
        
        # Başlık satırı
        header_frame = tk.Frame(card_frame, bg='#e74c3c')
        header_frame.pack(fill='x')
        
        # Öğrenci bilgileri
        student_info = f"#{index} - {registration['student_name']} ({registration['student_id']})"
        if registration['student_class']:
            student_info += f" - {registration['student_class']}"
        
        student_label = tk.Label(
            header_frame,
            text=student_info,
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#e74c3c',
            anchor='w'
        )
        student_label.pack(side='left', padx=10, pady=5)
        
        # Tarih
        date_label = tk.Label(
            header_frame,
            text=registration['created_at'],
            font=('Arial', 10),
            fg='white',
            bg='#e74c3c',
            anchor='e'
        )
        date_label.pack(side='right', padx=10, pady=5)
        
        # İçerik frame
        content_frame = tk.Frame(card_frame, bg='white')
        content_frame.pack(fill='x', padx=10, pady=10)
        
        # Fotoğraf yolu
        photo_label = tk.Label(
            content_frame,
            text=f"📷 Fotoğraf: {os.path.basename(registration['photo_path'])}",
            font=('Arial', 10),
            fg='#2c3e50',
            bg='white',
            anchor='w'
        )
        photo_label.pack(anchor='w', pady=2)
        
        # Kalite skoru
        quality_label = tk.Label(
            content_frame,
            text=f"📊 Kalite Skoru: {registration['quality_score']:.2f}/1.00",
            font=('Arial', 10),
            fg='#2c3e50',
            bg='white',
            anchor='w'
        )
        quality_label.pack(anchor='w', pady=2)
        
        # Başarısızlık nedeni
        reason_label = tk.Label(
            content_frame,
            text=f"❌ Neden: {registration['failure_reason']}",
            font=('Arial', 10, 'bold'),
            fg='#e74c3c',
            bg='white',
            anchor='w',
            wraplength=600
        )
        reason_label.pack(anchor='w', pady=5)
        
        # Butonlar frame
        buttons_frame = tk.Frame(content_frame, bg='white')
        buttons_frame.pack(fill='x', pady=10)
        
        # Detayları görüntüle butonu
        if registration['quality_report']:
            details_btn = tk.Button(
                buttons_frame,
                text="📋 Detayları Görüntüle",
                command=lambda r=registration: self._show_failed_registration_details(r),
                bg='#3498db',
                fg='white',
                font=('Arial', 9, 'bold'),
                cursor='hand2',
                padx=10
            )
            details_btn.pack(side='left', padx=5)
        
        # Fotoğrafı görüntüle butonu
        if registration['photo_path'] and os.path.exists(registration['photo_path']):
            view_photo_btn = tk.Button(
                buttons_frame,
                text="🖼️ Fotoğrafı Görüntüle",
                command=lambda p=registration['photo_path']: self._view_failed_photo(p),
                bg='#27ae60',
                fg='white',
                font=('Arial', 9, 'bold'),
                cursor='hand2',
                padx=10
            )
            view_photo_btn.pack(side='left', padx=5)
    
    def _show_failed_registration_details(self, registration):
        """Başarısız kayıt detaylarını gösterir"""
        # Yeni pencere oluştur
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Başarısız Kayıt Detayları - {registration['student_name']}")
        details_window.geometry("800x600")
        details_window.configure(bg='#f0f0f0')
        
        # Başlık
        title_label = tk.Label(
            details_window,
            text=f"❌ BAŞARISIZ KAYIT DETAYLARI",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Öğrenci bilgileri
        student_frame = tk.Frame(details_window, bg='#e74c3c', relief='raised', bd=2)
        student_frame.pack(fill='x', padx=20, pady=10)
        
        student_info = f"👤 {registration['student_name']} ({registration['student_id']})"
        if registration['student_class']:
            student_info += f" - {registration['student_class']}"
        
        student_label = tk.Label(
            student_frame,
            text=student_info,
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#e74c3c'
        )
        student_label.pack(pady=10)
        
        # Kalite raporu
        if registration['quality_report']:
            report_frame = tk.Frame(details_window, bg='white', relief='raised', bd=2)
            report_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            report_label = tk.Label(
                report_frame,
                text="📊 KALİTE ANALİZ RAPORU",
                font=('Arial', 12, 'bold'),
                bg='white'
            )
            report_label.pack(pady=10)
            
            # Scrollable text area
            text_frame = tk.Frame(report_frame, bg='white')
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            text_widget = scrolledtext.ScrolledText(
                text_frame,
                wrap=tk.WORD,
                font=('Courier', 10),
                bg='#f8f9fa',
                fg='#2c3e50'
            )
            text_widget.pack(fill='both', expand=True)
            text_widget.insert(tk.END, registration['quality_report'])
            text_widget.config(state='disabled')
        
        # Kapat butonu
        close_btn = tk.Button(
            details_window,
            text="Kapat",
            command=details_window.destroy,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2',
            padx=20
        )
        close_btn.pack(pady=20)
    
    def _view_failed_photo(self, photo_path):
        """Başarısız kayıt fotoğrafını görüntüler"""
        try:
            # Fotoğrafı yükle ve göster
            image = Image.open(photo_path)
            
            # Boyutlandır
            max_width = 800
            max_height = 600
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Tkinter için dönüştür
            photo = ImageTk.PhotoImage(image)
            
            # Yeni pencere oluştur
            photo_window = tk.Toplevel(self.root)
            photo_window.title(f"Fotoğraf - {os.path.basename(photo_path)}")
            photo_window.configure(bg='#f0f0f0')
            
            # Fotoğrafı göster
            photo_label = tk.Label(photo_window, image=photo, bg='#f0f0f0')
            photo_label.image = photo  # Referansı koru
            photo_label.pack(pady=20)
            
            # Kapat butonu
            close_btn = tk.Button(
                photo_window,
                text="Kapat",
                command=photo_window.destroy,
                bg='#95a5a6',
                fg='white',
                font=('Arial', 10, 'bold'),
                cursor='hand2',
                padx=20
            )
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Fotoğraf görüntülenemedi: {e}")

    def _create_manual_registration_buttons(self, unrecognized_face_numbers):
        """Tanınamayan yüzler için manuel kayıt butonları oluşturur"""
        try:
            # Önceki butonları temizle
            if hasattr(self, 'manual_face_buttons_frame') and self.manual_face_buttons_frame:
                self.manual_face_buttons_frame.destroy()
            
            # Yeni frame oluştur - scrollable_frame'i kullan (yüz tanıma ekranı için)
            if hasattr(self, 'scrollable_frame') and self.scrollable_frame:
                parent_frame = self.scrollable_frame
            else:
                parent_frame = self.main_frame  # Fallback
            
            self.manual_face_buttons_frame = tk.Frame(parent_frame, bg='#f0f0f0')
            self.manual_face_buttons_frame.pack(pady=10, padx=20, fill='x')
            
            # Başlık
            title_label = tk.Label(
                self.manual_face_buttons_frame,
                text="🔧 MANUEL ÖĞRENCİ KAYDI",
                font=('Arial', 12, 'bold'),
                bg='#f0f0f0',
                fg='#e74c3c'
            )
            title_label.pack(pady=(0, 10))
            
            # Alt başlık
            subtitle_label = tk.Label(
                self.manual_face_buttons_frame,
                text="Tanınamayan yüzleri sisteme manuel olarak kaydetmek için butona tıklayın:",
                font=('Arial', 10),
                bg='#f0f0f0',
                fg='#2c3e50'
            )
            subtitle_label.pack(pady=(0, 10))
            
            # Her tanınamayan yüz için buton oluştur
            for face_num in unrecognized_face_numbers:
                face_num_int = int(face_num)
                
                btn = tk.Button(
                    self.manual_face_buttons_frame,
                    text=f"👤 Yüz {face_num} için Manuel Kayıt",
                    command=lambda fn=face_num_int: self._open_manual_registration_dialog(fn),
                    bg='#3498db',
                    fg='white',
                    font=('Arial', 10, 'bold'),
                    relief='raised',
                    bd=2,
                    padx=20,
                    pady=8
                )
                btn.pack(pady=5, padx=10, anchor='center')
            

            
            # Scroll region güncelle (yeni içerik eklendiği için)
            if hasattr(self, 'recognition_canvas') and self.recognition_canvas:
                self.root.after(100, self._update_scroll_region)
            
        except Exception as e:
            print(f"❌ Manuel kayıt butonları oluşturma hatası: {e}")
    
    def _open_manual_registration_dialog(self, face_number):
        """Belirli bir yüz için manuel kayıt dialog'u açar"""
        try:
            # Yüz verilerini kontrol et
            if not self.detected_faces or face_number > len(self.detected_faces):
                messagebox.showerror("Hata", f"Yüz {face_number} verisi bulunamadı!")
                return
            
            face_data = self.detected_faces[face_number - 1]  # 0-indexed
            
            # Dialog penceresi oluştur
            dialog = tk.Toplevel(self.root)
            dialog.title(f"👤 Yüz {face_number} için Manuel Öğrenci Kaydı")
            dialog.geometry("450x350")
            dialog.configure(bg='#f0f0f0')
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Ortalama
            dialog.geometry("+%d+%d" % (
                self.root.winfo_rootx() + 50,
                self.root.winfo_rooty() + 50
            ))
            
            # Başlık
            title_label = tk.Label(
                dialog,
                text=f"🎓 Yüz {face_number} için Öğrenci Bilgileri",
                font=('Arial', 14, 'bold'),
                bg='#f0f0f0',
                fg='#2c3e50'
            )
            title_label.pack(pady=20)
            
            # Form frame
            form_frame = tk.Frame(dialog, bg='#f0f0f0')
            form_frame.pack(pady=20, padx=30, fill='both', expand=True)
            
            # Öğrenci Adı
            tk.Label(form_frame, text="👤 Öğrenci Adı:", font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#2c3e50').grid(row=0, column=0, sticky='w', pady=5)
            name_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
            name_entry.grid(row=0, column=1, pady=5, padx=(10, 0))
            name_entry.focus()
            
            # Öğrenci ID
            tk.Label(form_frame, text="🆔 Öğrenci ID:", font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#2c3e50').grid(row=1, column=0, sticky='w', pady=5)
            id_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
            id_entry.grid(row=1, column=1, pady=5, padx=(10, 0))
            
            # Öğrenci Sınıfı
            tk.Label(form_frame, text="🏫 Öğrenci Sınıfı:", font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#2c3e50').grid(row=2, column=0, sticky='w', pady=5)
            class_entry = tk.Entry(form_frame, font=('Arial', 11), width=25)
            class_entry.grid(row=2, column=1, pady=5, padx=(10, 0))
            
            # Bilgi metni
            info_label = tk.Label(
                form_frame,
                text="ℹ️ Bu bilgiler yüz tanıma sistemine kaydedilecek\nve gelecekteki tanıma işlemlerinde kullanılacak.",
                font=('Arial', 9),
                bg='#f0f0f0',
                fg='#7f8c8d',
                justify='center'
            )
            info_label.grid(row=3, column=0, columnspan=2, pady=15)
            
            # Butonlar
            button_frame = tk.Frame(form_frame, bg='#f0f0f0')
            button_frame.grid(row=4, column=0, columnspan=2, pady=20)
            
            def save_student():
                name = name_entry.get().strip()
                student_id = id_entry.get().strip()
                student_class = class_entry.get().strip()
                
                if not name or not student_id:
                    messagebox.showerror("Hata", "Öğrenci adı ve ID alanları zorunludur!")
                    return
                
                try:
                    # Veritabanına kaydet
                    self._save_manual_registration(name, student_id, student_class, face_data, face_number)
                    dialog.destroy()
                    
                    # Başarı mesajı
                    messagebox.showinfo(
                        "✅ Başarılı!",
                        f"🎉 {name} ({student_id}) başarıyla kaydedildi!\n\n"
                        f"🔍 Bu yüz artık gelecekteki tanıma işlemlerinde\n"
                        f"otomatik olarak tanınacak."
                    )
                    
                except Exception as e:
                    messagebox.showerror("Hata", f"Kayıt sırasında hata: {e}")
            
            # Kaydet butonu
            save_btn = tk.Button(
                button_frame,
                text="💾 Kaydet",
                command=save_student,
                bg='#27ae60',
                fg='white',
                font=('Arial', 11, 'bold'),
                padx=20,
                pady=5
            )
            save_btn.pack(side='left', padx=10)
            
            # İptal butonu
            cancel_btn = tk.Button(
                button_frame,
                text="❌ İptal",
                command=dialog.destroy,
                bg='#e74c3c',
                fg='white',
                font=('Arial', 11, 'bold'),
                padx=20,
                pady=5
            )
            cancel_btn.pack(side='right', padx=10)
            
            # Enter ile kaydet
            def on_enter(event):
                save_student()
            
            dialog.bind('<Return>', on_enter)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Manuel kayıt dialog hatası: {e}")
    

    def _save_manual_registration(self, name, student_id, student_class, face_data, face_number):
        """Manuel kayıt verilerini sisteme kaydeder"""
        import time
        
        try:
            # Öğrenci zaten var mı kontrol et
            existing_student = self.db_manager.get_student_by_id(student_id)
            if existing_student:
                # Mevcut öğrenciye yeni yüz embedding'i ekle
                student_pk = existing_student[0]
                print(f"🔄 Mevcut öğrenciye ({name}) yeni yüz embedding'i ekleniyor...")
            else:
                # Yeni öğrenci oluştur
                student_pk = self.db_manager.add_student(name, student_id, student_class)
                print(f"✅ Yeni öğrenci oluşturuldu: {name} ({student_id})")
            
            # Yüz kalitesi analizi (basit)
            quality_analysis = {
                'overall_quality': 0.75,  # Manuel kayıt için varsayılan kalite
                'details': {
                    'manual_registration': True,
                    'face_number': face_number,
                    'source': 'recognition_photo_manual_save'
                },
                'summary': {
                    'total_passed': 3,
                    'total_failed': 2,
                    'passed_checks': ['Yüz Tespit Edildi', 'Manuel Doğrulama', 'Kullanıcı Onayı'],
                    'failed_checks': []
                }
            }
            
            # Embedding'i kaydet
            if hasattr(face_data, 'embedding'):
                embedding = face_data.embedding
            elif isinstance(face_data, dict) and 'embedding' in face_data:
                embedding = face_data['embedding']
            else:
                raise ValueError("Yüz embedding'i bulunamadı")
            
            # Kalite raporu oluştur
            quality_report = f"""📷 MANUEL KAYIT - YÜZ {face_number}
════════════════════════════════════════
👤 Öğrenci: {name}
🆔 ID: {student_id}
🏫 Sınıf: {student_class or 'Belirtilmedi'}
🔢 Yüz Numarası: {face_number}
📸 Kaynak: Yüz tanıma fotoğrafı (manuel kayıt)
📅 Kayıt Tarihi: {time.strftime('%Y-%m-%d %H:%M:%S')}

🔧 KALİTE ANALİZİ:
✅ Yüz tespit edildi
✅ Manuel doğrulama yapıldı  
✅ Kullanıcı tarafından onaylandı
⚠️ Otomatik kalite kontrolü yapılmadı
⚠️ Fotoğraf kalitesi manuel değerlendirme gerektiriyor

📊 GENEL SONUÇ:
Manuel kayıt olarak sisteme eklendi.
Bu kayıt gelecekteki otomatik tanıma işlemlerinde kullanılacak."""
            
            # Geçici dosya yolu (manuel kayıt için)
            temp_photo_path = f"manual_registration_face_{face_number}_{student_id}"
            
            self.db_manager.add_face_embedding(
                student_pk,
                embedding,
                temp_photo_path,
                quality_analysis['overall_quality'],
                quality_analysis,
                quality_report
            )
            
            print(f"✅ Manuel kayıt tamamlandı: {name} (Yüz {face_number})")
            
        except ValueError as ve:
            raise ve  # Veritabanı hatalarını yukarı taşı
        except Exception as e:
            raise Exception(f"Manuel kayıt hatası: {e}")
    
    def _update_scroll_region(self):
        """Scroll region'u günceller"""
        try:
            if hasattr(self, 'recognition_canvas') and self.recognition_canvas and self.recognition_canvas.winfo_exists():
                self.recognition_canvas.configure(scrollregion=self.recognition_canvas.bbox("all"))
        except Exception as e:
            print(f"⚠️ Scroll region güncelleme hatası: {e}")

    def update_status(self, message):
        """Durum mesajını günceller"""
        self.status_var.set(message)
        # Log alanına da ekle
        if hasattr(self, 'log_area'):
            self.log_area.insert(tk.END, f"[STATUS] {message}\n")
            self.log_area.see(tk.END)
    
    def run(self):
        """Uygulamayı başlatır"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.run() 