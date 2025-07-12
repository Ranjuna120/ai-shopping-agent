"""
Data Processing and Validation System for AI Shopping Agent
Handles data cleaning, validation, transformation, and batch processing
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import validators
from urllib.parse import urlparse, parse_qs
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
import aiohttp
import time
from collections import defaultdict, Counter
import unicodedata
from decimal import Decimal, InvalidOperation
import phonenumbers
from email_validator import validate_email, EmailNotValidError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NEEDS_CLEANING = "needs_cleaning"

@dataclass
class ValidationResult:
    """Data validation result"""
    status: ValidationStatus
    message: str
    original_value: Any
    cleaned_value: Any = None
    confidence: float = 1.0
    suggestions: List[str] = None

@dataclass
class ProcessingStats:
    """Data processing statistics"""
    total_records: int
    processed_records: int
    valid_records: int
    invalid_records: int
    cleaned_records: int
    warnings: int
    processing_time: float
    error_rate: float

class DataCleaner:
    """
    Comprehensive data cleaning utility
    """
    
    def __init__(self):
        self.price_patterns = {
            'currency': re.compile(r'[\$£€¥₹]'),
            'decimal': re.compile(r'\d+\.?\d*'),
            'range': re.compile(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)'),
            'comma_separated': re.compile(r'(\d{1,3}(?:,\d{3})*\.?\d*)'),
        }
        
        self.url_patterns = {
            'amazon': re.compile(r'amazon\.[a-z.]+/.*?/dp/([A-Z0-9]{10})'),
            'ebay': re.compile(r'ebay\.[a-z.]+/itm/(\d+)'),
            'walmart': re.compile(r'walmart\.com/ip/.*?/(\d+)'),
        }
        
        self.category_mappings = {
            'electronics': ['laptop', 'computer', 'phone', 'smartphone', 'tablet', 'camera', 
                          'headphone', 'speaker', 'mouse', 'keyboard', 'monitor'],
            'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat', 'sweater', 
                        'jeans', 'top', 'skirt', 'hat', 'socks'],
            'home': ['furniture', 'chair', 'table', 'bed', 'sofa', 'lamp', 'rug', 
                    'kitchen', 'bathroom', 'decor', 'curtain'],
            'sports': ['fitness', 'exercise', 'sport', 'gym', 'outdoor', 'bike', 
                      'yoga', 'running', 'swimming', 'football'],
            'books': ['book', 'novel', 'textbook', 'kindle', 'magazine', 'comic'],
            'beauty': ['makeup', 'skincare', 'beauty', 'cosmetic', 'perfume', 
                      'lotion', 'shampoo', 'cream'],
            'automotive': ['car', 'auto', 'tire', 'battery', 'oil', 'parts'],
            'gaming': ['game', 'gaming', 'console', 'controller', 'playstation', 
                      'xbox', 'nintendo'],
            'toys': ['toy', 'doll', 'puzzle', 'lego', 'children', 'kids'],
            'jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'bracelet', 'earring']
        }
    
    def clean_price(self, price_str: Union[str, float, int]) -> ValidationResult:
        """Clean and validate price data"""
        if price_str is None:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Price is None",
                original_value=price_str
            )
        
        # Convert to string for processing
        original_value = price_str
        price_str = str(price_str).strip()
        
        if not price_str:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Empty price value",
                original_value=original_value
            )
        
        # Remove currency symbols
        cleaned_price = re.sub(self.price_patterns['currency'], '', price_str)
        
        # Handle price ranges (take the lower value)
        range_match = self.price_patterns['range'].search(cleaned_price)
        if range_match:
            cleaned_price = range_match.group(1)
        
        # Remove commas from large numbers
        cleaned_price = cleaned_price.replace(',', '')
        
        # Extract decimal number
        decimal_match = self.price_patterns['decimal'].search(cleaned_price)
        if not decimal_match:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="No valid price number found",
                original_value=original_value
            )
        
        try:
            price_value = float(decimal_match.group())
            
            # Validate reasonable price range
            if price_value <= 0:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message="Price must be positive",
                    original_value=original_value
                )
            elif price_value > 1000000:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Unusually high price",
                    original_value=original_value,
                    cleaned_value=price_value,
                    confidence=0.7
                )
            elif price_value < 0.01:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Unusually low price",
                    original_value=original_value,
                    cleaned_value=price_value,
                    confidence=0.7
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    message="Price cleaned successfully",
                    original_value=original_value,
                    cleaned_value=round(price_value, 2),
                    confidence=1.0
                )
                
        except (ValueError, TypeError):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Could not convert to numeric value",
                original_value=original_value
            )
    
    def clean_product_name(self, name: str) -> ValidationResult:
        """Clean and validate product name"""
        if not name or not isinstance(name, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid or empty product name",
                original_value=name
            )
        
        original_name = name
        
        # Remove extra whitespace
        cleaned_name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove special characters but keep essential ones
        cleaned_name = re.sub(r'[^\w\s\-.,()&/]', '', cleaned_name)
        
        # Normalize unicode characters
        cleaned_name = unicodedata.normalize('NFKD', cleaned_name)
        
        # Check minimum length
        if len(cleaned_name) < 3:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Product name too short",
                original_value=original_name
            )
        
        # Check maximum length
        if len(cleaned_name) > 200:
            cleaned_name = cleaned_name[:200].rsplit(' ', 1)[0]
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Product name truncated",
                original_value=original_name,
                cleaned_value=cleaned_name,
                confidence=0.8
            )
        
        # Check for suspicious patterns
        if re.search(r'(\w)\1{5,}', cleaned_name):  # Repeated characters
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Suspicious repeated characters",
                original_value=original_name,
                cleaned_value=cleaned_name,
                confidence=0.6
            )
        
        confidence = 1.0 if cleaned_name == original_name else 0.9
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message="Product name cleaned successfully",
            original_value=original_name,
            cleaned_value=cleaned_name,
            confidence=confidence
        )
    
    def clean_url(self, url: str) -> ValidationResult:
        """Clean and validate URL"""
        if not url or not isinstance(url, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid or empty URL",
                original_value=url
            )
        
        original_url = url
        cleaned_url = url.strip()
        
        # Add protocol if missing
        if not cleaned_url.startswith(('http://', 'https://')):
            cleaned_url = 'https://' + cleaned_url
        
        # Validate URL format
        if not validators.url(cleaned_url):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid URL format",
                original_value=original_url
            )
        
        # Parse URL components
        try:
            parsed = urlparse(cleaned_url)
            
            # Check for supported platforms
            platform = self._identify_platform(cleaned_url)
            if platform == 'unknown':
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Unknown platform",
                    original_value=original_url,
                    cleaned_value=cleaned_url,
                    confidence=0.7
                )
            
            # Extract product ID for validation
            product_id = self._extract_product_id(cleaned_url, platform)
            if not product_id:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Could not extract product ID",
                    original_value=original_url,
                    cleaned_value=cleaned_url,
                    confidence=0.8
                )
            
            confidence = 1.0 if cleaned_url == original_url else 0.9
            
            return ValidationResult(
                status=ValidationStatus.VALID,
                message="URL cleaned successfully",
                original_value=original_url,
                cleaned_value=cleaned_url,
                confidence=confidence
            )
            
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"URL parsing error: {str(e)}",
                original_value=original_url
            )
    
    def clean_email(self, email: str) -> ValidationResult:
        """Clean and validate email address"""
        if not email or not isinstance(email, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid or empty email",
                original_value=email
            )
        
        original_email = email
        cleaned_email = email.strip().lower()
        
        try:
            validation = validate_email(cleaned_email)
            cleaned_email = validation.email
            
            return ValidationResult(
                status=ValidationStatus.VALID,
                message="Email cleaned successfully",
                original_value=original_email,
                cleaned_value=cleaned_email,
                confidence=1.0
            )
            
        except EmailNotValidError as e:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Invalid email: {str(e)}",
                original_value=original_email
            )
    
    def categorize_product(self, product_name: str) -> ValidationResult:
        """Categorize product based on name"""
        if not product_name or not isinstance(product_name, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid product name for categorization",
                original_value=product_name
            )
        
        product_lower = product_name.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_mappings.items():
            score = sum(1 for keyword in keywords if keyword in product_lower)
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Could not categorize product",
                original_value=product_name,
                cleaned_value='other',
                confidence=0.3
            )
        
        # Get best matching category
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]
        
        # Calculate confidence based on score
        confidence = min(max_score / 3, 1.0)  # Normalize to 0-1 range
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message=f"Product categorized as {best_category}",
            original_value=product_name,
            cleaned_value=best_category,
            confidence=confidence
        )
    
    def _identify_platform(self, url: str) -> str:
        """Identify e-commerce platform from URL"""
        url_lower = url.lower()
        
        if 'amazon' in url_lower:
            return 'amazon'
        elif 'ebay' in url_lower:
            return 'ebay'
        elif 'walmart' in url_lower:
            return 'walmart'
        elif 'target' in url_lower:
            return 'target'
        elif 'bestbuy' in url_lower:
            return 'bestbuy'
        else:
            return 'unknown'
    
    def _extract_product_id(self, url: str, platform: str) -> Optional[str]:
        """Extract product ID from URL"""
        if platform in self.url_patterns:
            match = self.url_patterns[platform].search(url)
            if match:
                return match.group(1)
        return None

class DataValidator:
    """
    Advanced data validation system
    """
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load validation rules for different data types"""
        return {
            'user': {
                'required_fields': ['username', 'email'],
                'field_types': {
                    'username': str,
                    'email': str,
                    'created_at': str
                },
                'constraints': {
                    'username': {'min_length': 3, 'max_length': 50},
                    'email': {'format': 'email'}
                }
            },
            'product': {
                'required_fields': ['name', 'price'],
                'field_types': {
                    'name': str,
                    'price': (int, float, str),
                    'url': str,
                    'platform': str
                },
                'constraints': {
                    'name': {'min_length': 3, 'max_length': 200},
                    'price': {'min_value': 0.01, 'max_value': 1000000},
                    'url': {'format': 'url'}
                }
            },
            'search_history': {
                'required_fields': ['user_id', 'query'],
                'field_types': {
                    'user_id': int,
                    'query': str,
                    'budget': (int, float, str),
                    'created_at': str
                },
                'constraints': {
                    'query': {'min_length': 1, 'max_length': 500},
                    'budget': {'min_value': 0}
                }
            },
            'favorites': {
                'required_fields': ['user_id', 'product_name'],
                'field_types': {
                    'user_id': int,
                    'product_name': str,
                    'price': (int, float, str),
                    'platform': str,
                    'product_url': str
                },
                'constraints': {
                    'product_name': {'min_length': 3, 'max_length': 200},
                    'price': {'min_value': 0.01},
                    'product_url': {'format': 'url'}
                }
            },
            'price_alert': {
                'required_fields': ['user_id', 'product_url', 'target_price'],
                'field_types': {
                    'user_id': int,
                    'product_url': str,
                    'target_price': (int, float, str),
                    'current_price': (int, float, str)
                },
                'constraints': {
                    'target_price': {'min_value': 0.01},
                    'current_price': {'min_value': 0.01},
                    'product_url': {'format': 'url'}
                }
            }
        }
    
    def validate_record(self, record: Dict[str, Any], record_type: str) -> Dict[str, ValidationResult]:
        """Validate a single record"""
        if record_type not in self.validation_rules:
            raise ValueError(f"Unknown record type: {record_type}")
        
        rules = self.validation_rules[record_type]
        results = {}
        
        # Check required fields
        for field in rules['required_fields']:
            if field not in record or record[field] is None:
                results[field] = ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Required field '{field}' is missing",
                    original_value=record.get(field)
                )
        
        # Validate each field
        for field, value in record.items():
            if field in results:  # Already validated as missing
                continue
            
            result = self._validate_field(field, value, rules, record_type)
            results[field] = result
        
        return results
    
    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any], record_type: str) -> ValidationResult:
        """Validate a single field"""
        # Check field type
        if field_name in rules['field_types']:
            expected_type = rules['field_types'][field_name]
            if not isinstance(value, expected_type):
                # Try type conversion for certain types
                if expected_type == str and value is not None:
                    value = str(value)
                elif expected_type in (int, float) and isinstance(value, str):
                    try:
                        value = float(value) if expected_type == float else int(value)
                    except (ValueError, TypeError):
                        return ValidationResult(
                            status=ValidationStatus.INVALID,
                            message=f"Invalid type for {field_name}",
                            original_value=value
                        )
        
        # Apply field-specific validation
        if field_name == 'email':
            return self.cleaner.clean_email(value)
        elif field_name in ['product_name', 'name']:
            return self.cleaner.clean_product_name(value)
        elif field_name in ['price', 'target_price', 'current_price', 'budget']:
            return self.cleaner.clean_price(value)
        elif field_name in ['url', 'product_url']:
            return self.cleaner.clean_url(value)
        elif field_name == 'query' and record_type == 'search_history':
            return self._validate_search_query(value)
        
        # Apply general constraints
        constraints = rules.get('constraints', {}).get(field_name, {})
        return self._apply_constraints(field_name, value, constraints)
    
    def _validate_search_query(self, query: str) -> ValidationResult:
        """Validate search query"""
        if not query or not isinstance(query, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid search query",
                original_value=query
            )
        
        original_query = query
        cleaned_query = query.strip()
        
        if len(cleaned_query) < 1:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Search query too short",
                original_value=original_query
            )
        
        if len(cleaned_query) > 500:
            cleaned_query = cleaned_query[:500]
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Search query truncated",
                original_value=original_query,
                cleaned_value=cleaned_query,
                confidence=0.9
            )
        
        # Check for suspicious patterns
        if re.search(r'[<>\"\'&]', cleaned_query):
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Potentially unsafe characters in query",
                original_value=original_query,
                cleaned_value=re.sub(r'[<>\"\'&]', '', cleaned_query),
                confidence=0.8
            )
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message="Search query valid",
            original_value=original_query,
            cleaned_value=cleaned_query,
            confidence=1.0
        )
    
    def _apply_constraints(self, field_name: str, value: Any, constraints: Dict[str, Any]) -> ValidationResult:
        """Apply general constraints to field value"""
        if not constraints:
            return ValidationResult(
                status=ValidationStatus.VALID,
                message="No constraints to validate",
                original_value=value,
                cleaned_value=value,
                confidence=1.0
            )
        
        # Length constraints
        if 'min_length' in constraints and hasattr(value, '__len__'):
            if len(value) < constraints['min_length']:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"{field_name} too short (min: {constraints['min_length']})",
                    original_value=value
                )
        
        if 'max_length' in constraints and hasattr(value, '__len__'):
            if len(value) > constraints['max_length']:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"{field_name} too long (max: {constraints['max_length']})",
                    original_value=value,
                    cleaned_value=value[:constraints['max_length']] if isinstance(value, str) else value,
                    confidence=0.8
                )
        
        # Value constraints
        if 'min_value' in constraints and isinstance(value, (int, float)):
            if value < constraints['min_value']:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"{field_name} below minimum value ({constraints['min_value']})",
                    original_value=value
                )
        
        if 'max_value' in constraints and isinstance(value, (int, float)):
            if value > constraints['max_value']:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"{field_name} above maximum value ({constraints['max_value']})",
                    original_value=value,
                    cleaned_value=constraints['max_value'],
                    confidence=0.7
                )
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message="Constraints satisfied",
            original_value=value,
            cleaned_value=value,
            confidence=1.0
        )

class DataProcessor:
    """
    Main data processing orchestrator
    """
    
    def __init__(self, db_path: str = 'shopping_agent.db'):
        self.db_path = db_path
        self.validator = DataValidator()
        self.batch_size = 1000
        self.max_workers = 4
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def process_table(self, table_name: str, record_type: str) -> ProcessingStats:
        """Process and validate entire table"""
        start_time = time.time()
        
        conn = self.get_db_connection()
        
        # Get total record count
        total_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        if total_count == 0:
            conn.close()
            return ProcessingStats(
                total_records=0,
                processed_records=0,
                valid_records=0,
                invalid_records=0,
                cleaned_records=0,
                warnings=0,
                processing_time=0,
                error_rate=0
            )
        
        # Process in batches
        stats = ProcessingStats(
            total_records=total_count,
            processed_records=0,
            valid_records=0,
            invalid_records=0,
            cleaned_records=0,
            warnings=0,
            processing_time=0,
            error_rate=0
        )
        
        # Get column names
        cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
        columns = [description[0] for description in cursor.description]
        
        # Process records in batches
        offset = 0
        while offset < total_count:
            batch_query = f"SELECT * FROM {table_name} LIMIT {self.batch_size} OFFSET {offset}"
            batch_data = conn.execute(batch_query).fetchall()
            
            batch_results = self._process_batch(batch_data, columns, record_type)
            
            # Update statistics
            stats.processed_records += len(batch_data)
            stats.valid_records += batch_results['valid']
            stats.invalid_records += batch_results['invalid']
            stats.cleaned_records += batch_results['cleaned']
            stats.warnings += batch_results['warnings']
            
            # Update cleaned records back to database
            if batch_results['updates']:
                self._apply_batch_updates(conn, table_name, batch_results['updates'])
            
            offset += self.batch_size
            
            # Progress logging
            progress = (offset / total_count) * 100
            logger.info(f"Processing {table_name}: {progress:.1f}% complete")
        
        conn.close()
        
        # Calculate final statistics
        end_time = time.time()
        stats.processing_time = end_time - start_time
        stats.error_rate = stats.invalid_records / stats.total_records if stats.total_records > 0 else 0
        
        logger.info(f"Completed processing {table_name}: {stats.valid_records}/{stats.total_records} valid records")
        
        return stats
    
    def _process_batch(self, batch_data: List[Tuple], columns: List[str], record_type: str) -> Dict[str, Any]:
        """Process a batch of records"""
        results = {
            'valid': 0,
            'invalid': 0,
            'cleaned': 0,
            'warnings': 0,
            'updates': []
        }
        
        for row_data in batch_data:
            record = dict(zip(columns, row_data))
            validation_results = self.validator.validate_record(record, record_type)
            
            # Analyze validation results
            has_invalid = any(r.status == ValidationStatus.INVALID for r in validation_results.values())
            has_warnings = any(r.status == ValidationStatus.WARNING for r in validation_results.values())
            has_cleaned = any(r.cleaned_value is not None for r in validation_results.values())
            
            if has_invalid:
                results['invalid'] += 1
            else:
                results['valid'] += 1
                
                if has_cleaned:
                    results['cleaned'] += 1
                    # Prepare update for cleaned data
                    cleaned_record = record.copy()
                    for field, result in validation_results.items():
                        if result.cleaned_value is not None:
                            cleaned_record[field] = result.cleaned_value
                    results['updates'].append(cleaned_record)
            
            if has_warnings:
                results['warnings'] += 1
        
        return results
    
    def _apply_batch_updates(self, conn: sqlite3.Connection, table_name: str, updates: List[Dict[str, Any]]):
        """Apply batch updates to database"""
        if not updates:
            return
        
        # Determine primary key (assume 'id' exists)
        for update in updates:
            if 'id' not in update:
                continue
            
            # Build update query
            set_clauses = []
            values = []
            
            for field, value in update.items():
                if field != 'id':
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
            
            if set_clauses:
                values.append(update['id'])
                query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = ?"
                conn.execute(query, values)
        
        conn.commit()
    
    def validate_single_record(self, record: Dict[str, Any], record_type: str) -> Dict[str, ValidationResult]:
        """Validate a single record and return results"""
        return self.validator.validate_record(record, record_type)
    
    def clean_and_validate_data(self, data: Any, data_type: str) -> ValidationResult:
        """Clean and validate specific data type"""
        if data_type == 'price':
            return self.validator.cleaner.clean_price(data)
        elif data_type == 'product_name':
            return self.validator.cleaner.clean_product_name(data)
        elif data_type == 'url':
            return self.validator.cleaner.clean_url(data)
        elif data_type == 'email':
            return self.validator.cleaner.clean_email(data)
        elif data_type == 'category':
            return self.validator.cleaner.categorize_product(data)
        else:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Unknown data type: {data_type}",
                original_value=data
            )
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'tables': {},
            'overall_stats': {},
            'issues': [],
            'recommendations': []
        }
        
        # Define table-to-record-type mapping
        table_mappings = {
            'users': 'user',
            'favorites': 'favorites',
            'search_history': 'search_history',
            'price_alert': 'price_alert'
        }
        
        total_records = 0
        total_valid = 0
        total_invalid = 0
        
        # Process each table
        for table_name, record_type in table_mappings.items():
            try:
                stats = self.process_table(table_name, record_type)
                report['tables'][table_name] = asdict(stats)
                
                total_records += stats.total_records
                total_valid += stats.valid_records
                total_invalid += stats.invalid_records
                
                # Add issues and recommendations
                if stats.error_rate > 0.1:  # More than 10% error rate
                    report['issues'].append(f"High error rate in {table_name}: {stats.error_rate:.1%}")
                    report['recommendations'].append(f"Review data entry processes for {table_name}")
                
            except Exception as e:
                logger.error(f"Error processing table {table_name}: {e}")
                report['issues'].append(f"Failed to process table {table_name}: {str(e)}")
        
        # Calculate overall statistics
        report['overall_stats'] = {
            'total_records': total_records,
            'total_valid': total_valid,
            'total_invalid': total_invalid,
            'overall_quality_score': total_valid / total_records if total_records > 0 else 0,
            'processed_at': datetime.now().isoformat()
        }
        
        return report
    
    def export_validation_results(self, results: Dict[str, ValidationResult], output_path: str):
        """Export validation results to file"""
        export_data = []
        
        for field, result in results.items():
            export_data.append({
                'field': field,
                'status': result.status.value,
                'message': result.message,
                'original_value': str(result.original_value),
                'cleaned_value': str(result.cleaned_value) if result.cleaned_value is not None else '',
                'confidence': result.confidence
            })
        
        df = pd.DataFrame(export_data)
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_path + '.csv', index=False)
        
        logger.info(f"Validation results exported to {output_path}")

# Flask route integration
def create_data_processing_routes(app, db):
    """Create data processing routes for Flask app"""
    processor = DataProcessor()
    
    @app.route('/api/data/validate', methods=['POST'])
    def validate_data():
        """Validate data through API"""
        try:
            from flask import request
            data = request.get_json()
            
            record = data.get('record', {})
            record_type = data.get('type', 'product')
            
            results = processor.validate_single_record(record, record_type)
            
            # Convert results to serializable format
            serialized_results = {}
            for field, result in results.items():
                serialized_results[field] = {
                    'status': result.status.value,
                    'message': result.message,
                    'original_value': result.original_value,
                    'cleaned_value': result.cleaned_value,
                    'confidence': result.confidence
                }
            
            return {
                'success': True,
                'validation_results': serialized_results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/data/clean', methods=['POST'])
    def clean_data():
        """Clean specific data through API"""
        try:
            from flask import request
            data = request.get_json()
            
            value = data.get('value')
            data_type = data.get('type', 'product_name')
            
            result = processor.clean_and_validate_data(value, data_type)
            
            return {
                'success': True,
                'result': {
                    'status': result.status.value,
                    'message': result.message,
                    'original_value': result.original_value,
                    'cleaned_value': result.cleaned_value,
                    'confidence': result.confidence
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/data/quality-report', methods=['GET'])
    def get_quality_report():
        """Get data quality report"""
        try:
            report = processor.generate_data_quality_report()
            return {
                'success': True,
                'report': report
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500
    
    @app.route('/api/data/process-table/<table_name>', methods=['POST'])
    def process_table(table_name):
        """Process and validate specific table"""
        try:
            from flask import request
            
            # Map table names to record types
            table_mappings = {
                'users': 'user',
                'favorites': 'favorites',
                'search_history': 'search_history',
                'price_alert': 'price_alert'
            }
            
            if table_name not in table_mappings:
                return {'success': False, 'error': 'Unknown table'}, 400
            
            record_type = table_mappings[table_name]
            stats = processor.process_table(table_name, record_type)
            
            return {
                'success': True,
                'processing_stats': asdict(stats)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

if __name__ == "__main__":
    # Test the data processing system
    processor = DataProcessor()
    
    # Test individual data cleaning
    print("Testing data cleaning:")
    
    price_result = processor.clean_and_validate_data("$99.99", "price")
    print(f"Price cleaning: {price_result.cleaned_value} (Status: {price_result.status.value})")
    
    name_result = processor.clean_and_validate_data("  Gaming Laptop RTX 4060  ", "product_name")
    print(f"Name cleaning: {name_result.cleaned_value} (Status: {name_result.status.value})")
    
    url_result = processor.clean_and_validate_data("amazon.com/dp/B08N5WRWNW", "url")
    print(f"URL cleaning: {url_result.cleaned_value} (Status: {url_result.status.value})")
    
    # Generate quality report
    print("\nGenerating data quality report...")
    report = processor.generate_data_quality_report()
    print(f"Overall quality score: {report['overall_stats']['overall_quality_score']:.2%}")
