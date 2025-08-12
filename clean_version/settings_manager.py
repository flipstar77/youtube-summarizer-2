import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from supabase_client import SupabaseDatabase
from database import Database
import hashlib
from cryptography.fernet import Fernet
import base64

class SettingsManager:
    """
    Manages application settings including API keys, preferences, and configurations
    """
    
    def __init__(self):
        """Initialize settings manager with same database as main app"""
        # For now, force SQLite for settings until Supabase tables are created
        self.use_supabase = False
        self.db = Database()
        print("[INFO] Using SQLite for settings management")
        
        # Initialize encryption
        self._init_encryption()
        self._init_settings_table()
    
    def _init_encryption(self):
        """Initialize encryption for sensitive settings like API keys"""
        # Generate or load encryption key
        key_file = "D:/mcp/.settings_key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            # Ensure directory exists
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            # Restrict file permissions (Windows)
            try:
                os.chmod(key_file, 0o600)
            except:
                pass
        
        self.cipher = Fernet(self.encryption_key)
    
    def _init_settings_table(self):
        """Initialize settings table in database"""
        if self.use_supabase:
            try:
                # For Supabase, we'd create via SQL if needed
                self.db.client.rpc('create_settings_table_if_not_exists').execute()
                print("[OK] Supabase settings table initialized")
            except Exception as e:
                print(f"[INFO] Supabase settings table may already exist: {str(e)}")
        else:
            # For SQLite, create table directly
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT UNIQUE NOT NULL,
                    setting_value TEXT,
                    setting_type TEXT DEFAULT 'text',
                    is_encrypted BOOLEAN DEFAULT FALSE,
                    is_sensitive BOOLEAN DEFAULT FALSE,
                    category TEXT DEFAULT 'general',
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("[OK] SQLite settings table initialized")
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value"""
        return self.cipher.decrypt(encrypted_value.encode()).decode()
    
    def set_setting(self, key: str, value: Any, setting_type: str = 'text', 
                   is_sensitive: bool = False, category: str = 'general', 
                   description: str = '') -> bool:
        """
        Set a setting value
        """
        try:
            # Convert value to string for storage
            if isinstance(value, (dict, list)):
                str_value = json.dumps(value)
                setting_type = 'json'
            elif isinstance(value, bool):
                str_value = str(value).lower()
                setting_type = 'boolean'
            elif isinstance(value, (int, float)):
                str_value = str(value)
                setting_type = 'number'
            else:
                str_value = str(value)
            
            # Encrypt sensitive values
            is_encrypted = False
            if is_sensitive and str_value:
                str_value = self._encrypt_value(str_value)
                is_encrypted = True
            
            setting_data = {
                'setting_key': key,
                'setting_value': str_value,
                'setting_type': setting_type,
                'is_encrypted': is_encrypted,
                'is_sensitive': is_sensitive,
                'category': category,
                'description': description,
                'updated_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                # Use upsert to insert or update
                result = self.db.client.table('app_settings').upsert(setting_data).execute()
                return len(result.data) > 0
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                # Check if setting exists
                cursor.execute('SELECT id FROM app_settings WHERE setting_key = ?', (key,))
                if cursor.fetchone():
                    # Update existing
                    cursor.execute('''
                        UPDATE app_settings 
                        SET setting_value = ?, setting_type = ?, is_encrypted = ?, 
                            is_sensitive = ?, category = ?, description = ?, updated_at = ?
                        WHERE setting_key = ?
                    ''', (str_value, setting_type, is_encrypted, is_sensitive, 
                         category, description, datetime.now().isoformat(), key))
                else:
                    # Insert new
                    cursor.execute('''
                        INSERT INTO app_settings 
                        (setting_key, setting_value, setting_type, is_encrypted, 
                         is_sensitive, category, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (key, str_value, setting_type, is_encrypted, 
                         is_sensitive, category, description))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            print(f"[ERROR] Error setting value for {key}: {str(e)}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value
        """
        try:
            if self.use_supabase:
                result = self.db.client.table('app_settings').select('*').eq('setting_key', key).execute()
                if not result.data:
                    return default
                setting = result.data[0]
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM app_settings WHERE setting_key = ?', (key,))
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return default
                
                columns = ['id', 'setting_key', 'setting_value', 'setting_type', 
                          'is_encrypted', 'is_sensitive', 'category', 'description',
                          'created_at', 'updated_at']
                setting = dict(zip(columns, row))
            
            value = setting['setting_value']
            
            # Decrypt if encrypted
            if setting['is_encrypted'] and value:
                try:
                    value = self._decrypt_value(value)
                except Exception as e:
                    print(f"[WARNING] Could not decrypt setting {key}: {str(e)}")
                    return default
            
            # Convert to appropriate type
            if not value:
                return default
                
            setting_type = setting['setting_type']
            
            if setting_type == 'boolean':
                return value.lower() in ('true', '1', 'yes', 'on')
            elif setting_type == 'number':
                try:
                    return float(value) if '.' in value else int(value)
                except:
                    return default
            elif setting_type == 'json':
                try:
                    return json.loads(value)
                except:
                    return default
            else:
                return value
                
        except Exception as e:
            print(f"[ERROR] Error getting setting {key}: {str(e)}")
            return default
    
    def get_settings_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get all settings in a category
        """
        try:
            if self.use_supabase:
                result = self.db.client.table('app_settings').select('*').eq('category', category).execute()
                settings_rows = result.data
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM app_settings WHERE category = ?', (category,))
                rows = cursor.fetchall()
                conn.close()
                
                if not rows:
                    return {}
                
                columns = ['id', 'setting_key', 'setting_value', 'setting_type', 
                          'is_encrypted', 'is_sensitive', 'category', 'description',
                          'created_at', 'updated_at']
                settings_rows = [dict(zip(columns, row)) for row in rows]
            
            result = {}
            for setting in settings_rows:
                key = setting['setting_key']
                # For sensitive settings, only return masked values in category view
                if setting['is_sensitive']:
                    result[key] = {
                        'value': '***MASKED***' if setting['setting_value'] else '',
                        'type': setting['setting_type'],
                        'description': setting['description'],
                        'is_sensitive': True,
                        'has_value': bool(setting['setting_value'])
                    }
                else:
                    # Get the actual value (with decryption if needed)
                    actual_value = self.get_setting(key)
                    result[key] = {
                        'value': actual_value,
                        'type': setting['setting_type'],
                        'description': setting['description'],
                        'is_sensitive': False,
                        'has_value': actual_value is not None
                    }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error getting settings for category {category}: {str(e)}")
            return {}
    
    def delete_setting(self, key: str) -> bool:
        """
        Delete a setting
        """
        try:
            if self.use_supabase:
                result = self.db.client.table('app_settings').delete().eq('setting_key', key).execute()
                return len(result.data) > 0
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM app_settings WHERE setting_key = ?', (key,))
                success = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return success
                
        except Exception as e:
            print(f"[ERROR] Error deleting setting {key}: {str(e)}")
            return False
    
    def initialize_default_settings(self):
        """
        Initialize default application settings
        """
        default_settings = [
            # API Keys
            {
                'key': 'openai_api_key',
                'value': os.getenv('OPENAI_API_KEY', ''),
                'category': 'api_keys',
                'description': 'OpenAI API key for GPT models and embeddings',
                'sensitive': True
            },
            {
                'key': 'elevenlabs_api_key',
                'value': os.getenv('ELEVENLABS_API_KEY', ''),
                'category': 'api_keys', 
                'description': 'ElevenLabs API key for text-to-speech',
                'sensitive': True
            },
            {
                'key': 'perplexity_api_key',
                'value': os.getenv('PERPLEXITY_API_KEY', ''),
                'category': 'api_keys',
                'description': 'Perplexity API key for enhanced AI responses',
                'sensitive': True
            },
            {
                'key': 'deepseek_api_key', 
                'value': os.getenv('DEEPSEEK_API_KEY', ''),
                'category': 'api_keys',
                'description': 'DeepSeek API key for alternative AI model',
                'sensitive': True
            },
            {
                'key': 'gemini_api_key',
                'value': os.getenv('GEMINI_API_KEY', ''),
                'category': 'api_keys',
                'description': 'Google Gemini API key',
                'sensitive': True
            },
            {
                'key': 'claude_api_key',
                'value': os.getenv('CLAUDE_API_KEY', ''),
                'category': 'api_keys',
                'description': 'Anthropic Claude API key',
                'sensitive': True
            },
            
            # Database Settings
            {
                'key': 'supabase_url',
                'value': os.getenv('SUPABASE_URL', ''),
                'category': 'database',
                'description': 'Supabase project URL',
                'sensitive': True
            },
            {
                'key': 'supabase_service_role_key',
                'value': os.getenv('SUPABASE_SERVICE_ROLE_KEY', ''),
                'category': 'database',
                'description': 'Supabase service role key',
                'sensitive': True
            },
            {
                'key': 'use_supabase',
                'value': os.getenv('USE_SUPABASE', 'false').lower() == 'true',
                'type': 'boolean',
                'category': 'database',
                'description': 'Use Supabase instead of local SQLite',
                'sensitive': False
            },
            
            # Application Settings
            {
                'key': 'app_debug_mode',
                'value': False,
                'type': 'boolean',
                'category': 'application',
                'description': 'Enable debug mode for development',
                'sensitive': False
            },
            {
                'key': 'auto_generate_audio',
                'value': False,
                'type': 'boolean',
                'category': 'application', 
                'description': 'Automatically generate audio for new summaries',
                'sensitive': False
            },
            {
                'key': 'default_ai_provider',
                'value': 'openai',
                'category': 'application',
                'description': 'Default AI provider for summaries',
                'sensitive': False
            },
            
            # Subscription Settings
            {
                'key': 'auto_process_subscriptions',
                'value': False,
                'type': 'boolean',
                'category': 'subscriptions',
                'description': 'Automatically process videos from subscribed channels',
                'sensitive': False
            },
            {
                'key': 'subscription_check_interval',
                'value': 3600,
                'type': 'number',
                'category': 'subscriptions',
                'description': 'Interval in seconds to check for new subscription videos',
                'sensitive': False
            },
            
            # Vector Search Settings
            {
                'key': 'vector_search_threshold',
                'value': 0.75,
                'type': 'number',
                'category': 'vector_search',
                'description': 'Similarity threshold for vector search (0.0-1.0, higher = more similar)',
                'sensitive': False
            },
            {
                'key': 'vector_search_limit',
                'value': 10,
                'type': 'number',
                'category': 'vector_search',
                'description': 'Maximum number of similar results to return',
                'sensitive': False
            },
            {
                'key': 'vector_search_enabled',
                'value': True,
                'type': 'boolean',
                'category': 'vector_search',
                'description': 'Enable semantic vector search functionality',
                'sensitive': False
            }
        ]
        
        for setting in default_settings:
            # Only set if not already exists
            existing = self.get_setting(setting['key'])
            if existing is None:
                self.set_setting(
                    key=setting['key'],
                    value=setting['value'],
                    setting_type=setting.get('type', 'text'),
                    is_sensitive=setting.get('sensitive', False),
                    category=setting['category'],
                    description=setting['description']
                )
        
        print("[OK] Default settings initialized")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider
        """
        key_map = {
            'openai': 'openai_api_key',
            'elevenlabs': 'elevenlabs_api_key',
            'perplexity': 'perplexity_api_key',
            'deepseek': 'deepseek_api_key',
            'gemini': 'gemini_api_key',
            'claude': 'claude_api_key'
        }
        
        setting_key = key_map.get(provider.lower())
        if not setting_key:
            return None
            
        return self.get_setting(setting_key)
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set API key for a specific provider
        """
        key_map = {
            'openai': 'openai_api_key',
            'elevenlabs': 'elevenlabs_api_key', 
            'perplexity': 'perplexity_api_key',
            'deepseek': 'deepseek_api_key',
            'gemini': 'gemini_api_key',
            'claude': 'claude_api_key'
        }
        
        setting_key = key_map.get(provider.lower())
        if not setting_key:
            return False
            
        return self.set_setting(
            key=setting_key,
            value=api_key,
            is_sensitive=True,
            category='api_keys',
            description=f'{provider} API key'
        )
    
    def validate_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """
        Validate an API key by making a test request
        """
        try:
            if provider.lower() == 'openai':
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Simple test request
                response = client.models.list()
                return {'valid': True, 'message': 'OpenAI API key is valid'}
                
            elif provider.lower() == 'elevenlabs':
                import requests
                headers = {'xi-api-key': api_key}
                response = requests.get('https://api.elevenlabs.io/v1/voices', headers=headers)
                if response.status_code == 200:
                    return {'valid': True, 'message': 'ElevenLabs API key is valid'}
                else:
                    return {'valid': False, 'message': 'ElevenLabs API key is invalid'}
                    
            # Add more provider validations as needed
            else:
                return {'valid': True, 'message': f'Validation not implemented for {provider}'}
                
        except Exception as e:
            return {'valid': False, 'message': f'API key validation failed: {str(e)}'}
    
    def export_settings(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Export all settings for backup purposes
        """
        try:
            if self.use_supabase:
                result = self.db.client.table('app_settings').select('*').execute()
                settings = result.data
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM app_settings')
                rows = cursor.fetchall()
                conn.close()
                
                columns = ['id', 'setting_key', 'setting_value', 'setting_type', 
                          'is_encrypted', 'is_sensitive', 'category', 'description',
                          'created_at', 'updated_at']
                settings = [dict(zip(columns, row)) for row in rows]
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'settings': {}
            }
            
            for setting in settings:
                key = setting['setting_key']
                
                if setting['is_sensitive'] and not include_sensitive:
                    continue
                
                # Get actual value (decrypted)
                value = self.get_setting(key)
                
                export_data['settings'][key] = {
                    'value': value,
                    'type': setting['setting_type'],
                    'category': setting['category'],
                    'description': setting['description'],
                    'is_sensitive': setting['is_sensitive']
                }
            
            return export_data
            
        except Exception as e:
            print(f"[ERROR] Error exporting settings: {str(e)}")
            return {'error': str(e)}