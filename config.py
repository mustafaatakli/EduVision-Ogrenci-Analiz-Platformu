#!/usr/bin/env python3
"""
OKULDAN Yüz Tanıma Sistemi - Merkezi Konfigürasyon Yöneticisi
Bu modül tüm sistem ayarlarını tek bir yerden yönetir.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

@dataclass
class DatabaseConfig:
    """Veritabanı konfigürasyonu"""
    # Veritabanı türü: 
    db_type: str = "mssql"
    
    # SQLite ayarları 
    path: str = "face_recognition_test.db"
    
    # MSSQL bağlantı ayarları
    server: str = "localhost"
    database: str = "Face_Recognition_Test_db"
    username: str = ""
    password: str = ""
    port: int = 1433
    driver: str = "ODBC Driver 17 for SQL Server"
    
    # Genel ayarlar
    timeout: int = 30
    pool_size: int = 5
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_directory: str = "backups"
    
    # MSSQL özel ayarları
    connection_string: Optional[str] = None
    trust_server_certificate: bool = True
    encrypt: bool = False

@dataclass
class APIConfig:
    """API sunucu konfigürasyonu"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    
    max_file_size_mb: int = 10
    allowed_file_types: list = field(default_factory=lambda: ["image0/jpeg", "image0/jpg", "image0/png", "image0/bmp"])
    
    request_timeout_seconds: int = 300
    connection_timeout_seconds: int = 60

@dataclass
class AIModelConfig:
    """AI model konfigürasyonu"""
    models_directory: str = "models"
    device: str = "auto"  # "cpu", "cuda", "auto"
    
    # InsightFace model ayarları
    face_detection_threshold: float = 0.5
    face_recognition_threshold: float = 0.55
    face_quality_threshold: float = 0.60
    
    # Model indirilecek URL'ler
    model_urls: Dict[str, str] = field(default_factory=lambda: {
        "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
    })
    
    batch_size: int = 1
    num_threads: int = 4
    memory_limit_gb: int = 4

@dataclass
class EmotionConfig:
    """Duygu analizi konfigürasyonu"""
    enabled: bool = True
    backend: str = "opencv"  # "opencv", "ssd", "dlib", "mtcnn", "retinaface"
    model_name: str = "VGG-Face"  # "VGG-Face", "Facenet", "OpenFace", "DeepFace"
    
    # Duygu etiketleri (DeepFace desteklediği 7 temel duygu)
    emotion_labels: Dict[str, str] = field(default_factory=lambda: {
        "angry": "Kızgın",
        "disgust": "Tiksinmiş", 
        "fear": "Korkmuş",
        "happy": "Mutlu",
        "sad": "Üzgün",
        "surprise": "Şaşkın",
        "neutral": "Nötr"
    })
    
    # Performans ayarları
    min_confidence: float = 0.1  # Minimum güven skoru
    enforce_detection: bool = False  # Yüz tespiti zorunlu mu
    
    # Cache ayarları
    cache_enabled: bool = True
    cache_models: bool = True
    
    # GUI görselleştirme ayarları
    show_confidence: bool = True  
    show_all_emotions: bool = False  
    emotion_color_map: Dict[str, str] = field(default_factory=lambda: {
        "Mutlu": "#2ecc71",      # Yeşil
        "Üzgün": "#3498db",      # Mavi  
        "Kızgın": "#e74c3c",     # Kırmızı
        "Şaşkın": "#f39c12",     # Turuncu
        "Korkmuş": "#9b59b6",    # Mor
        "Tiksinmiş": "#95a5a6",  # Gri
        "Nötr": "#34495e"        # Koyu gri
    })

@dataclass
class PhotoConfig:
    """Fotoğraf işleme konfigürasyonu"""
    photos_directory: str = "photos"
    temp_directory: str = "temp"
    max_photo_dimension: int = 1920
    jpeg_quality: int = 95
    
    auto_cleanup_temp: bool = True
    temp_file_max_age_hours: int = 24
    
    # Fotoğraf kalite kriterleri
    quality_criteria: Dict[str, Any] = field(default_factory=lambda: {
        "sharpness": {
            "min_threshold": 0.3,
            "weight": 0.3,
            "is_critical": True
        },
        "eyes_open": {
            "min_threshold": 0.5,
            "weight": 0.15,
            "is_critical": False
        },
        "face_angle": {
            "max_yaw": 30.0,
            "max_pitch": 30.0,
            "max_roll": 15.0,
            "weight": 0.2,
            "is_critical": True
        },
        "face_integrity": {
            "min_completion": 0.85,
            "weight": 0.2,
            "is_critical": True
        },
        "lighting": {
            "min_threshold": 0.4,
            "weight": 0.15,
            "is_critical": False
        }
    })
    
    # Akıllı karar sistemi
    smart_decision: Dict[str, Any] = field(default_factory=lambda: {
        "critical_criteria_required": 3,  # Tüm kritik kriterler geçmeli
        "support_criteria_minimum": 1,   # En az 1 destek kriter geçmeli
        "overall_quality_minimum": 0.50  # Genel kalite minimum
    })

@dataclass
class LoggingConfig:
    """Loglama konfigürasyonu"""
    log_directory: str = "logs"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    api_log_file: str = "api.log"
    error_log_file: str = "error.log"
    access_log_file: str = "access.log"
    
    max_log_size_mb: int = 10
    backup_count: int = 5
    
    console_enabled: bool = True
    file_enabled: bool = True

@dataclass
class SecurityConfig:
    """Güvenlik konfigürasyonu"""
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    
    jwt_secret_key: Optional[str] = None
    jwt_expire_hours: int = 24
    
    allowed_ips: list = field(default_factory=list)
    blocked_ips: list = field(default_factory=list)
    
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class SystemConfig:
    """Sistem konfigürasyonu"""
    environment: str = "development"  # "development", "production", "testing"
    debug_mode: bool = False
    
    base_directory: str = os.getcwd()
    data_directory: str = "data"
    cache_directory: str = "cache"
    
    # Performans ayarları
    max_concurrent_requests: int = 100
    cleanup_interval_minutes: int = 60
    
    health_check_enabled: bool = True
    metrics_enabled: bool = True
    metrics_port: int = 9090

class ConfigManager:
    """Merkezi konfigürasyon yöneticisi"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config = self._load_config()
        self._setup_directories()
        
    def _find_config_file(self) -> Optional[str]:
        """Config dosyasını otomatik bulur"""
        possible_paths = [
            "config.yaml",
            "config.yml", 
            "settings.yaml",
            "settings.yml",
            os.path.join("config", "config.yaml"),
            os.path.join("config", "config.yml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Konfigürasyonu yükler"""
        config = {}
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Environment variables ile config'i override eder"""
        
        env_mappings = {
            # Database
            "DB_PATH": ("database", "path"),
            "DB_TIMEOUT": ("database", "timeout"),
            "DB_BACKUP_ENABLED": ("database", "backup_enabled"),
            
            # API
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
            "API_DEBUG": ("api", "debug"),
            "API_WORKERS": ("api", "workers"),
            "CORS_ORIGINS": ("api", "cors_origins"),
            
            # AI Models
            "MODELS_DIR": ("ai_models", "models_directory"),
            "AI_DEVICE": ("ai_models", "device"),
            "FACE_DETECTION_THRESHOLD": ("ai_models", "face_detection_threshold"),
            "FACE_RECOGNITION_THRESHOLD": ("ai_models", "face_recognition_threshold"),
            
            # Photos
            "PHOTOS_DIR": ("photos", "photos_directory"),
            "MAX_FILE_SIZE_MB": ("api", "max_file_size_mb"),
            
            # Logging
            "LOG_LEVEL": ("logging", "log_level"),
            "LOG_DIR": ("logging", "log_directory"),
            
            # Security
            "API_KEY": ("security", "api_key"),
            "SECRET_KEY": ("security", "secret_key"),
            "JWT_SECRET": ("security", "jwt_secret_key"),
            
            # System
            "ENVIRONMENT": ("system", "environment"),
            "DEBUG_MODE": ("system", "debug_mode"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in config:
                    config[section] = {}
                
                if key in ["port", "timeout", "workers", "max_file_size_mb"]:
                    config[section][key] = int(value)
                elif key in ["debug", "backup_enabled", "debug_mode"]:
                    config[section][key] = value.lower() in ["true", "1", "yes", "on"]
                elif key in ["face_detection_threshold", "face_recognition_threshold"]:
                    config[section][key] = float(value)
                elif key == "cors_origins":
                    config[section][key] = value.split(",") if value else ["*"]
                else:
                    config[section][key] = value
        
        return config
    
    def _setup_directories(self):
        """Gerekli dizinleri oluşturur"""
        directories = [
            self.photos.photos_directory,
            self.photos.temp_directory,
            self.ai_models.models_directory,
            self.logging.log_directory,
            self.system.data_directory,
            self.system.cache_directory,
            os.path.dirname(self.database.path) or ".",
            self.database.backup_directory
        ]
        
        for directory in directories:
            if directory and directory != ".":
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def database(self) -> DatabaseConfig:
        """Veritabanı konfigürasyonu"""
        config = self._config.get("database", {})
        return DatabaseConfig(**config)
    
    @property
    def api(self) -> APIConfig:
        """API konfigürasyonu"""
        config = self._config.get("api", {})
        return APIConfig(**config)
    
    @property
    def ai_models(self) -> AIModelConfig:
        """AI model konfigürasyonu"""
        config = self._config.get("ai_models", {})
        return AIModelConfig(**config)
    
    @property
    def emotion(self) -> EmotionConfig:
        """Duygu analizi konfigürasyonu"""
        config = self._config.get("emotion", {})
        return EmotionConfig(**config)
    
    @property
    def photos(self) -> PhotoConfig:
        """Fotoğraf konfigürasyonu"""
        config = self._config.get("photos", {})
        return PhotoConfig(**config)
    
    @property
    def logging(self) -> LoggingConfig:
        """Loglama konfigürasyonu"""
        config = self._config.get("logging", {})
        return LoggingConfig(**config)
    
    @property 
    def security(self) -> SecurityConfig:
        """Güvenlik konfigürasyonu"""
        config = self._config.get("security", {})
        return SecurityConfig(**config)
    
    @property
    def system(self) -> SystemConfig:
        """Sistem konfigürasyonu"""
        config = self._config.get("system", {})
        return SystemConfig(**config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Herhangi bir config değerini alır"""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self):
        """Konfigürasyonu yeniden yükler"""
        self._config = self._load_config()
        self._setup_directories()
        print("Konfigürasyon yeniden yüklendi")
    
    def save_example_config(self, path: str = "config.example.yaml"):
        """Örnek config dosyası oluşturur"""
        example_config = {
            "# OKULDAN Yüz Tanıma Sistemi - Konfigürasyon Dosyası": None,
            "# Bu dosyayı config.yaml olarak kopyalayın": None,
            "": None,
            
            "database": {
                "path": "face_recognition.db",
                "timeout": 30,
                "pool_size": 5,
                "backup_enabled": True,
                "backup_interval_hours": 24,
                "backup_directory": "backups"
            },
            
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "reload": False,
                "workers": 1,
                "cors_origins": ["*"],
                "cors_methods": ["*"],
                "cors_headers": ["*"],
                "cors_credentials": True,
                "rate_limit_enabled": True,
                "rate_limit_requests_per_minute": 60,
                "max_file_size_mb": 10,
                "allowed_file_types": ["image/jpeg", "image/jpg", "image/png", "image/bmp"],
                "request_timeout_seconds": 300
            },
            
            "ai_models": {
                "models_directory": "models",
                "device": "auto",
                "face_detection_threshold": 0.5,
                "face_recognition_threshold": 0.55,
                "face_quality_threshold": 0.60,
                "batch_size": 1,
                "num_threads": 4,
                "memory_limit_gb": 4
            },
            
            "photos": {
                "photos_directory": "photos",
                "temp_directory": "temp",
                "max_photo_dimension": 1920,
                "jpeg_quality": 95,
                "auto_cleanup_temp": True,
                "temp_file_max_age_hours": 24,
                "quality_criteria": {
                    "sharpness": {"min_threshold": 0.3, "weight": 0.3, "is_critical": True},
                    "eyes_open": {"min_threshold": 0.5, "weight": 0.15, "is_critical": False},
                    "face_angle": {"max_yaw": 30.0, "max_pitch": 30.0, "max_roll": 15.0, "weight": 0.2, "is_critical": True},
                    "face_integrity": {"min_completion": 0.85, "weight": 0.2, "is_critical": True},
                    "lighting": {"min_threshold": 0.4, "weight": 0.15, "is_critical": False}
                },
                "smart_decision": {
                    "critical_criteria_required": 3,
                    "support_criteria_minimum": 1,
                    "overall_quality_minimum": 0.50
                }
            },
            
            "logging": {
                "log_directory": "logs",
                "log_level": "INFO",
                "api_log_file": "api.log",
                "error_log_file": "error.log",
                "max_log_size_mb": 10,
                "backup_count": 5,
                "console_enabled": True,
                "file_enabled": True
            },
            
            "security": {
                "# API anahtarları - production için environment variables kullanın": None,
                "api_key": None,
                "secret_key": None,
                "jwt_secret_key": None,
                "jwt_expire_hours": 24,
                "ssl_enabled": False,
                "allowed_ips": [],
                "blocked_ips": []
            },
            
            "system": {
                "environment": "development",
                "debug_mode": False,
                "data_directory": "data",
                "cache_directory": "cache",
                "max_concurrent_requests": 100,
                "cleanup_interval_minutes": 60,
                "health_check_enabled": True,
                "metrics_enabled": True
            }
        }
        
        yaml_content = []
        for key, value in example_config.items():
            if value is None:
                if key.startswith("#"):
                    yaml_content.append(key)
                elif key == "":
                    yaml_content.append("")
            else:
                yaml_content.append(f"{key}:")
                yaml_content.append(yaml.dump(value, default_flow_style=False, indent=2).rstrip())
                yaml_content.append("")
        
        with open(path, 'w', encoding='utf-8') as f:
            
            content = """# OKULDAN Yüz Tanıma Sistemi - Konfigürasyon Dosyası
# Bu dosyayı config.yaml olarak kopyalayın
def get_config() -> ConfigManager:
    """Global config instance'ını döndürür"""
    global config
    if config is None:
        config = ConfigManager()
    return config

def reload_config():
    """Global config'i yeniden yükler"""
    global config
    if config:
        config.reload()
    else:
        config = ConfigManager()

# Convenience functions
def get_db_config() -> DatabaseConfig:
    """Veritabanı config'ini döndürür"""
    return get_config().database

def get_api_config() -> APIConfig:
    """API config'ini döndürür"""
    return get_config().api

def get_ai_config() -> AIModelConfig:
    """AI model config'ini döndürür"""
    return get_config().ai_models

def get_emotion_config() -> EmotionConfig:
    """Duygu analizi config'ini döndürür"""
    return get_config().emotion

def get_photo_config() -> PhotoConfig:
    """Fotoğraf config'ini döndürür"""
    return get_config().photos

def get_logging_config() -> LoggingConfig:
    """Loglama config'ini döndürür"""
    return get_config().logging

def get_security_config() -> SecurityConfig:
    """Güvenlik config'ini döndürür"""
    return get_config().security

def get_system_config() -> SystemConfig:
    """Sistem config'ini döndürür"""
    return get_config().system

if __name__ == "__main__":
    # Test ve örnek dosya oluşturma
    config_manager = ConfigManager()
    config_manager.save_example_config()
    
    print("🔧 Config sistemi test ediliyor...")
    print(f"API Host: {config_manager.api.host}")
    print(f"API Port: {config_manager.api.port}")
    print(f"Database Path: {config_manager.database.path}")
    print(f"Models Directory: {config_manager.ai_models.models_directory}")
    print(f"Log Level: {config_manager.logging.log_level}")
    print("Config sistemi başarıyla çalışıyor!")
