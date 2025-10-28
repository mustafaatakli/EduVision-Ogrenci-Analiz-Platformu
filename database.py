#!/usr/bin/env python3
"""
OKULDAN Yüz Tanıma Sistemi - MSSQL Veritabanı Yöneticisi
"""

import pyodbc
import pymssql
import numpy as np
import pickle
import os
import json
import logging
from typing import List, Tuple, Optional, Dict, Union
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

# Config sistemi import
try:
    from config import get_db_config
    db_config = get_db_config()
    CONFIG_AVAILABLE = True
except ImportError:
    print("Config sistemi bulunamadı, varsayılan veritabanı ayarları kullanılacak")
    CONFIG_AVAILABLE = False
    class DefaultDBConfig:
        db_type = "mssql"
        server = "localhost"
        database = "FaceRecognition_TestDB"
        username = ""
        password = ""
        port = 1433
        driver = "ODBC Driver 17 for SQL Server"
        timeout = 30
        backup_enabled = True
        backup_directory = "backups"
        connection_string = None
        trust_server_certificate = True
        encrypt = False
    
    db_config = DefaultDBConfig()

class DatabaseManager:
    def __init__(self, connection_params: Optional[Union[Dict, str]] = None):
        """MSSQL Veritabanı yöneticisini başlatır"""
        
        self.logger = logging.getLogger(__name__)
        
        if connection_params:
            if isinstance(connection_params, str):
                self.connection_params = self._get_connection_params_from_config()
                print(f"String db_path kullanımı deprecated: {connection_params}")
                print("Config sisteminden MSSQL ayarları kullanılıyor...")
            else:
                self.connection_params = connection_params
        else:
            self.connection_params = self._get_connection_params_from_config()
        
        self.engine = self._create_engine()
        
        print(f"Veritabanı: MSSQL ({self.connection_params['server']}/{self.connection_params['database']})")
        print(f"Timeout: {self.connection_params['timeout']}s")
        print(f"Backup: {'Etkin' if self.connection_params['backup_enabled'] else 'Devre dışı'}")
        
        self.init_database()
    
    def _get_connection_params_from_config(self) -> Dict:
        server = os.getenv('DB_SERVER', db_config.server if CONFIG_AVAILABLE else 'localhost')
        database = os.getenv('DB_DATABASE', db_config.database if CONFIG_AVAILABLE else 'FaceRecognitionDB')
        username = os.getenv('DB_USERNAME', db_config.username if CONFIG_AVAILABLE else '')
        password = os.getenv('DB_PASSWORD', db_config.password if CONFIG_AVAILABLE else '')
        port = int(os.getenv('DB_PORT', db_config.port if CONFIG_AVAILABLE else 1433))
        driver = os.getenv('DB_DRIVER', db_config.driver if CONFIG_AVAILABLE else 'ODBC Driver 17 for SQL Server')
        
        return {
            'server': server,
            'database': database,
            'username': username,
            'password': password,
            'port': port,
            'driver': driver,
            'timeout': db_config.timeout if CONFIG_AVAILABLE else 30,
            'backup_enabled': db_config.backup_enabled if CONFIG_AVAILABLE else True,
            'backup_directory': db_config.backup_directory if CONFIG_AVAILABLE else 'backups',
            'connection_string': db_config.connection_string if CONFIG_AVAILABLE else None,
            'trust_server_certificate': db_config.trust_server_certificate if CONFIG_AVAILABLE else True,
            'encrypt': db_config.encrypt if CONFIG_AVAILABLE else False
        }
    
    def _create_engine(self):
        """SQLAlchemy engine'i oluşturur"""
        try:
            if self.connection_params['connection_string']:
                connection_string = self.connection_params['connection_string']
            else:
                if self.connection_params['username'] and self.connection_params['password']:
                    connection_string = (
                        f"mssql+pyodbc://{quote_plus(self.connection_params['username'])}:"
                        f"{quote_plus(self.connection_params['password'])}@"
                        f"{self.connection_params['server']}:{self.connection_params['port']}/"
                        f"{self.connection_params['database']}"
                        f"?driver={quote_plus(self.connection_params['driver'])}"
                        f"&TrustServerCertificate={'yes' if self.connection_params['trust_server_certificate'] else 'no'}"
                        f"&Encrypt={'yes' if self.connection_params['encrypt'] else 'no'}"
                    )
                else:
                    connection_string = (
                        f"mssql+pyodbc://@{self.connection_params['server']}:{self.connection_params['port']}/"
                        f"{self.connection_params['database']}"
                        f"?driver={quote_plus(self.connection_params['driver'])}"
                        f"&Trusted_Connection=yes"
                        f"&TrustServerCertificate={'yes' if self.connection_params['trust_server_certificate'] else 'no'}"
                        f"&Encrypt={'yes' if self.connection_params['encrypt'] else 'no'}"
                    )
            
            engine = create_engine(
                connection_string,
                poolclass=StaticPool,
                echo=False  
            )
            
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"MSSQL bağlantısı başarılı: {self.connection_params['server']}/{self.connection_params['database']}")
            return engine
            
        except Exception as e:
            self.logger.error(f"MSSQL bağlantı hatası: {e}")
            raise
    
    def get_connection(self):
        """Yeni bağlantı döndürür"""
        return self.engine.connect()
    
    def init_database(self):
        """Veritabanını oluşturur ve tabloları hazırlar"""
        if self.connection_params['backup_enabled']:
            os.makedirs(self.connection_params['backup_directory'], exist_ok=True)
        
        with self.get_connection() as conn:
            # Öğrenci tablosu
            conn.execute(text('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='students' AND xtype='U')
                CREATE TABLE students (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    name NVARCHAR(255) NOT NULL,
                    student_id NVARCHAR(100) UNIQUE NOT NULL,
                    student_class NVARCHAR(50),
                    photo_count INT DEFAULT 0,
                    created_at DATETIME2 DEFAULT GETDATE()
                )
            '''))
            
            # Yüz embedding tablosu
            conn.execute(text('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='face_embeddings' AND xtype='U')
                CREATE TABLE face_embeddings (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    student_id INT,
                    embedding VARBINARY(MAX) NOT NULL,
                    photo_path NVARCHAR(500),
                    quality_score FLOAT,
                    quality_details NVARCHAR(MAX),
                    quality_report NVARCHAR(MAX),
                    created_at DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
                )
            '''))
            
            # Başarısız kayıtlar tablosu
            conn.execute(text('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='failed_registrations' AND xtype='U')
                CREATE TABLE failed_registrations (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    student_name NVARCHAR(255) NOT NULL,
                    student_id NVARCHAR(100) NOT NULL,
                    student_class NVARCHAR(50),
                    photo_path NVARCHAR(500),
                    quality_score FLOAT,
                    quality_details NVARCHAR(MAX),
                    quality_report NVARCHAR(MAX),
                    failure_reason NVARCHAR(MAX),
                    created_at DATETIME2 DEFAULT GETDATE()
                )
            '''))
            

            self._add_missing_columns(conn)
            
            conn.commit()
    
    def _add_missing_columns(self, conn):
        """Eksik olan alanları güvenli bir şekilde ekler"""
        try:
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'face_embeddings' AND COLUMN_NAME = 'quality_details'
            """))
            if result.scalar() == 0:
                conn.execute(text("ALTER TABLE face_embeddings ADD quality_details NVARCHAR(MAX)"))
                self.logger.info("quality_details alanı eklendi")
            
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'face_embeddings' AND COLUMN_NAME = 'quality_report'
            """))
            if result.scalar() == 0:
                conn.execute(text("ALTER TABLE face_embeddings ADD quality_report NVARCHAR(MAX)"))
                self.logger.info("quality_report alanı eklendi")
            
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'students' AND COLUMN_NAME = 'student_class'
            """))
            if result.scalar() == 0:
                conn.execute(text("ALTER TABLE students ADD student_class NVARCHAR(50)"))
                self.logger.info("student_class alanı eklendi")
                
        except Exception as e:
            self.logger.warning(f"Migration uyarısı: {e}")
    
    def add_student(self, name: str, student_id: str, student_class: str = None) -> int:
        """Yeni öğrenci ekler"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text("""
                    INSERT INTO students (name, student_id, student_class) 
                    OUTPUT INSERTED.id
                    VALUES (:name, :student_id, :student_class)
                """), {
                    'name': name,
                    'student_id': student_id,
                    'student_class': student_class
                })
                student_pk = result.scalar()
                conn.commit()
                return student_pk
            except Exception as e:
                conn.rollback()
                if "UNIQUE" in str(e) or "duplicate" in str(e).lower():
                    raise ValueError(f"Öğrenci ID '{student_id}' zaten mevcut!")
                raise
    
    def add_face_embedding(self, student_pk: int, embedding: np.ndarray, 
                          photo_path: str, quality_score: float, 
                          quality_details: Dict = None, quality_report: str = None):
        """Yüz embedding'i, detaylı kalite analizini ve formatlanmış raporu ekler"""
        with self.get_connection() as conn:
            try:
                embedding_blob = pickle.dumps(embedding)
                
                quality_details_json = None
                if quality_details:
                    try:
                        quality_details_json = json.dumps(quality_details, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Kalite detayları JSON'a çevrilemedi: {e}")
                        quality_details_json = None
                
                conn.execute(text('''
                    INSERT INTO face_embeddings 
                    (student_id, embedding, photo_path, quality_score, quality_details, quality_report) 
                    VALUES (:student_id, :embedding, :photo_path, :quality_score, :quality_details, :quality_report)
                '''), {
                    'student_id': student_pk,
                    'embedding': embedding_blob,
                    'photo_path': photo_path,
                    'quality_score': quality_score,
                    'quality_details': quality_details_json,
                    'quality_report': quality_report
                })
                
                conn.execute(text('''
                    UPDATE students 
                    SET photo_count = photo_count + 1 
                    WHERE id = :student_id
                '''), {'student_id': student_pk})
                
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Embedding ekleme hatası: {e}")
                raise
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Tüm embedding'leri döndürür"""
        with self.get_connection() as conn:
            result = conn.execute(text('''
                SELECT s.id, s.name, f.embedding 
                FROM students s 
                INNER JOIN face_embeddings f ON s.id = f.student_id
            '''))
            
            results = []
            for row in result:
                student_id, name, embedding_blob = row
                embedding = pickle.loads(embedding_blob)
                results.append((student_id, name, embedding))
            
            return results
    
    def get_student_by_id(self, student_id: str) -> Optional[Tuple[int, str, str]]:
        """Öğrenci ID'ye göre öğrenci bilgisini getirir"""
        with self.get_connection() as conn:
            result = conn.execute(text(
                "SELECT id, name, student_class FROM students WHERE student_id = :student_id"
            ), {'student_id': student_id})
            
            row = result.fetchone()
            return tuple(row) if row else None
    
    def get_all_students(self) -> List[Tuple[str, str, str, int]]:
        """Tüm öğrencileri listeler"""
        with self.get_connection() as conn:
            result = conn.execute(text(
                "SELECT student_id, name, student_class, photo_count FROM students ORDER BY name"
            ))
            
            return [tuple(row) for row in result]
    
    def delete_student(self, student_id: str) -> bool:
        """Öğrenciyi ve ilgili tüm verilerini siler"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text(
                    "SELECT id FROM students WHERE student_id = :student_id"
                ), {'student_id': student_id})
                
                row = result.fetchone()
                if not row:
                    return False
                
                internal_id = row[0]
                
                conn.execute(text(
                    "DELETE FROM face_embeddings WHERE student_id = :internal_id"
                ), {'internal_id': internal_id})
                
                result = conn.execute(text(
                    "DELETE FROM students WHERE id = :internal_id"
                ), {'internal_id': internal_id})
                
                deleted_count = result.rowcount
                conn.commit()
                
                return deleted_count > 0
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Öğrenci silme hatası: {e}")
                return False
    
    def get_student_photo_count(self, student_id: str) -> int:
        """Öğrencinin toplam fotoğraf sayısını döndürür"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text("""
                    SELECT COUNT(fe.id) 
                    FROM face_embeddings fe
                    INNER JOIN students s ON fe.student_id = s.id
                    WHERE s.student_id = :student_id
                """), {'student_id': student_id})
                
                count = result.scalar()
                return count if count else 0
                
            except Exception as e:
                self.logger.error(f"Fotoğraf sayısı sorgu hatası: {e}")
                return 0
    
    def get_student_quality_report(self, student_id: str) -> List[Dict]:
        """Öğrencinin tüm fotoğraflarının kalite raporunu döndürür"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text("""
                    SELECT fe.photo_path, fe.quality_score, fe.quality_details, 
                           fe.quality_report, fe.created_at
                    FROM face_embeddings fe
                    INNER JOIN students s ON fe.student_id = s.id
                    WHERE s.student_id = :student_id
                    ORDER BY fe.created_at DESC
                """), {'student_id': student_id})
                
                results = []
                for row in result:
                    photo_path, quality_score, quality_details_json, quality_report, created_at = row
                    
                    quality_details = None
                    if quality_details_json:
                        try:
                            quality_details = json.loads(quality_details_json)
                        except (json.JSONDecodeError, TypeError) as e:
                            self.logger.warning(f"JSON parse hatası: {e}")
                            quality_details = None
                    
                    results.append({
                        'photo_path': photo_path,
                        'quality_score': quality_score,
                        'quality_details': quality_details,
                        'quality_report': quality_report,
                        'created_at': created_at
                    })
                
                return results
                
            except Exception as e:
                self.logger.error(f"Kalite raporu sorgu hatası: {e}")
                return []
    
    def get_quality_statistics(self) -> Dict:
        """Tüm sistemin kalite istatistiklerini döndürür"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_photos,
                        AVG(quality_score) as avg_quality,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality,
                        SUM(CASE WHEN quality_score >= 0.80 THEN 1 ELSE 0 END) as excellent_photos,
                        SUM(CASE WHEN quality_score >= 0.60 AND quality_score < 0.80 THEN 1 ELSE 0 END) as good_photos,
                        SUM(CASE WHEN quality_score < 0.60 THEN 1 ELSE 0 END) as poor_photos
                    FROM face_embeddings
                    WHERE quality_score IS NOT NULL
                """))
                
                row = result.fetchone()
                
                if row and row[0] > 0:
                    total, avg, min_q, max_q, excellent, good, poor = row
                    return {
                        'total_photos': total,
                        'average_quality': round(float(avg), 3) if avg else 0,
                        'min_quality': round(float(min_q), 3) if min_q else 0,
                        'max_quality': round(float(max_q), 3) if max_q else 0,
                        'excellent_photos': excellent,
                        'good_photos': good,
                        'poor_photos': poor,
                        'quality_distribution': {
                            'excellent': round((excellent / total) * 100, 1) if total > 0 else 0,
                            'good': round((good / total) * 100, 1) if total > 0 else 0,
                            'poor': round((poor / total) * 100, 1) if total > 0 else 0
                        }
                    }
                else:
                    return {
                        'total_photos': 0,
                        'average_quality': 0,
                        'min_quality': 0,
                        'max_quality': 0,
                        'excellent_photos': 0,
                        'good_photos': 0,
                        'poor_photos': 0,
                        'quality_distribution': {'excellent': 0, 'good': 0, 'poor': 0}
                    }
                
            except Exception as e:
                self.logger.error(f"Kalite istatistikleri sorgu hatası: {e}")
                return {}
    
    def add_failed_registration(self, student_name: str, student_id: str, student_class: str, 
                               photo_path: str, quality_score: float, quality_details: Dict = None, 
                               quality_report: str = None, failure_reason: str = None):
        """Başarısız kayıt bilgilerini veritabanına kaydeder"""
        with self.get_connection() as conn:
            try:
                quality_details_json = None
                if quality_details:
                    try:
                        quality_details_json = json.dumps(quality_details, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Kalite detayları JSON'a çevrilemedi: {e}")
                        quality_details_json = None
                
                conn.execute(text('''
                    INSERT INTO failed_registrations 
                    (student_name, student_id, student_class, photo_path, quality_score, 
                     quality_details, quality_report, failure_reason) 
                    VALUES (:student_name, :student_id, :student_class, :photo_path, 
                            :quality_score, :quality_details, :quality_report, :failure_reason)
                '''), {
                    'student_name': student_name,
                    'student_id': student_id,
                    'student_class': student_class,
                    'photo_path': photo_path,
                    'quality_score': quality_score,
                    'quality_details': quality_details_json,
                    'quality_report': quality_report,
                    'failure_reason': failure_reason
                })
                
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Başarısız kayıt ekleme hatası: {e}")
                raise
    
    def get_student_id_by_pk(self, student_pk: int) -> Optional[str]:
        """Primary key'e göre student_id döndürür"""
        with self.get_connection() as conn:
            try:
                result = conn.execute(text(
                    "SELECT student_id FROM students WHERE id = :student_pk"
                ), {'student_pk': student_pk})
                
                row = result.fetchone()
                return row[0] if row else None
                
            except Exception as e:
                self.logger.error(f"Student ID sorgu hatası: {e}")
                return None
    
    @staticmethod
    def generate_formatted_quality_report(photo_path: str, quality_details: Dict) -> str:
        """Formatlanmış kalite raporu oluşturur (GUI'deki formatla aynı)"""
        try:
            if not quality_details or 'details' not in quality_details:
                return "Kalite verisi bulunamadı"
            
            filename = photo_path.split('/')[-1] if photo_path else 'Bilinmeyen dosya'
            report_lines = []
            
            # Başlık
            report_lines.append(f"📷 FOTOĞRAF: {filename}")
            report_lines.append("=" * 55)
            
            # Akıllı kalite analizi başlık
            report_lines.append("🔧 AKILLI KALİTE ANALİZİ:")
            report_lines.append("")
            
            details = quality_details['details']
            
            # KRİTİK KRİTERLER
            report_lines.append("● KRİTİK KRİTERLER (3/3 MUTLAKA GEÇMELI):")
            
            # 1. Yüz Netliği (KRİTİK)
            sharpness = details.get('sharpness', {})
            score = sharpness.get('score', 0)
            is_adequate = sharpness.get('is_adequate', False) or sharpness.get('is_sharp', False)
            message = sharpness.get('message', 'Bilgi yok')
            
            status_icon = "✓" if is_adequate else "✗"
            report_lines.append(f"1 □Yüz Netliği (KRİTİK):")
            report_lines.append(f"  {status_icon} {message} (Skor: {score:.2f})")
            
            # 3. Açı Uygunluğu (KRİTİK - GÖRSEL TABANLI)
            angle = details.get('face_angle', {})
            score = angle.get('score', 0)
            is_adequate = angle.get('is_adequate', False) or angle.get('is_suitable', False)
            message = angle.get('message', 'Bilgi yok')
            
            status_icon = "✓" if is_adequate else "✗"
            report_lines.append(f"3 □Açı_Uygunluğu (KRİTİK - GÖRSEL TABANLI):")
            report_lines.append(f"  {status_icon} {message} (Skor: {score:.2f})")
            
            # 4. Yüz Bütünlüğü (KRİTİK)
            integrity = details.get('face_integrity', {})
            score = integrity.get('score', 0)
            is_adequate = integrity.get('is_adequate', False) or integrity.get('is_complete', False)
            message = integrity.get('message', 'Bilgi yok')
            
            status_icon = "✓" if is_adequate else "✗"
            report_lines.append(f"4 □Yüz Bütünlüğü (KRİTİK):")
            report_lines.append(f"  {status_icon} {message} (Skor: {score:.2f})")
            
            report_lines.append("")
            
            # DESTEK KRİTERLERİ
            report_lines.append("● DESTEK KRİTERLERİ (2'sinden EN AZ 1'i GEÇMELI):")
            
            # 2. Gözler (DESTEK)
            eyes = details.get('eyes_open', {})
            score = eyes.get('score', 0)
            is_adequate = eyes.get('is_adequate', False) or eyes.get('are_open', False)
            message = eyes.get('message', 'Bilgi yok')
            
            status_icon = "✓" if is_adequate else "✗"
            report_lines.append(f"2 □Gözler (DESTEK):")
            report_lines.append(f"  {status_icon} {message} (Skor: {score:.2f})")
            
            # 5. Işık (DESTEK)
            lighting = details.get('lighting', {})
            score = lighting.get('score', 0)
            is_adequate = lighting.get('is_adequate', False)
            message = lighting.get('message', 'Bilgi yok')
            
            status_icon = "✓" if is_adequate else "✗"
            report_lines.append(f"5 □Işık (DESTEK):")
            report_lines.append(f"  {status_icon} {message} (Skor: {score:.2f})")
            
            report_lines.append("")
            
            # GENEL SONUÇ
            summary = quality_details.get('summary', {})
            overall_quality = quality_details.get('overall_quality', 0)
            total_passed = summary.get('total_passed', 0)
            
            report_lines.append("📊 GENEL SONUÇ:")
            report_lines.append(f"Başarılı kriterler: {total_passed}/5")
            report_lines.append(f"Genel kalite skoru: {overall_quality:.2f}/1.00")
            report_lines.append("")
            
            # AKILLI KARAR ANALİZİ
            critical_passed = [
                details.get('sharpness', {}).get('is_adequate', False) or details.get('sharpness', {}).get('is_sharp', False),
                details.get('face_angle', {}).get('is_adequate', False) or details.get('face_angle', {}).get('is_suitable', False),
                details.get('face_integrity', {}).get('is_adequate', False) or details.get('face_integrity', {}).get('is_complete', False)
            ]
            critical_count = sum(critical_passed)
            
            # Destek kriterler kontrolü  
            support_passed = [
                details.get('eyes_open', {}).get('is_adequate', False) or details.get('eyes_open', {}).get('are_open', False),
                details.get('lighting', {}).get('is_adequate', False)
            ]
            support_count = sum(support_passed)
            
            report_lines.append("AKILLI KARAR ANALİZİ:")
            report_lines.append(f"● Kritik: {critical_count}/3 başarılı")
            report_lines.append(f"● Destek: {support_count}/2 başarılı")
            report_lines.append(f"Genel kalite: %{int(overall_quality * 100)}")
            report_lines.append("")
            
            report_lines.append("🔧 DETAYLI SKORLAR (Debug):")
            sharpness_ok = details.get('sharpness', {}).get('is_adequate', False) or details.get('sharpness', {}).get('is_sharp', False)
            angle_ok = details.get('face_angle', {}).get('is_adequate', False) or details.get('face_angle', {}).get('is_suitable', False)
            integrity_ok = details.get('face_integrity', {}).get('is_adequate', False) or details.get('face_integrity', {}).get('is_complete', False)
            eyes_ok = details.get('eyes_open', {}).get('is_adequate', False) or details.get('eyes_open', {}).get('are_open', False)
            lighting_ok = details.get('lighting', {}).get('is_adequate', False)
            
            report_lines.append(f"Netlik: {details.get('sharpness', {}).get('score', 0):.2f} ({'✅' if sharpness_ok else '❌'})")
            report_lines.append(f"Açı: {details.get('face_angle', {}).get('score', 0):.2f} ({'✅' if angle_ok else '❌'})")
            report_lines.append(f"Bütünlük: {details.get('face_integrity', {}).get('score', 0):.2f} ({'✅' if integrity_ok else '❌'})")
            report_lines.append(f"Gözler: {details.get('eyes_open', {}).get('score', 0):.2f} ({'✅' if eyes_ok else '❌'})")
            report_lines.append(f"Işık: {details.get('lighting', {}).get('score', 0):.2f} ({'✅' if lighting_ok else '❌'})")
            report_lines.append("")
            
            if critical_count == 3 and support_count >= 1:
                report_lines.append("FOTOĞRAF SİSTEME EKLENDİ!")
                report_lines.append("✓ Neden: Tüm kritik kriterler + en az 1 destek başarılı")
            else:
                report_lines.append("FOTOĞRAF REDDEDİLDİ!")
                reasons = []
                if critical_count < 3:
                    reasons.append(f"Kritik kriterler yetersiz ({critical_count}/3)")
                if critical_count == 3 and support_count == 0:
                    reasons.append("Destek kriterleri yetersiz (0/2)")
                report_lines.append(f"✗ Neden: {', '.join(reasons)}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Formatlanmış rapor oluşturulurken hata: {e}"
    
    def backup_database(self) -> bool:
        """Veritabanı yedeği oluşturur (MSSQL için)"""
        if not self.connection_params['backup_enabled']:
            return False
        
        try:
            import datetime
            backup_filename = f"face_recognition_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            backup_path = os.path.join(self.connection_params['backup_directory'], backup_filename)
            
            with self.get_connection() as conn:
                conn.execute(text(f"""
                    BACKUP DATABASE [{self.connection_params['database']}] 
                    TO DISK = :backup_path
                    WITH FORMAT, INIT
                """), {'backup_path': backup_path})
                
                conn.commit()
            
            self.logger.info(f"Veritabanı yedeği oluşturuldu: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup hatası: {e}")
            return False
    
    def close(self):
        """Bağlantı havuzunu kapatır"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
