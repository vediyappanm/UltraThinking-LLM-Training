"""Data quality checks and validation"""
from typing import Dict, Any, Optional, List
import logging
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate training data quality"""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 100000,
        check_duplicates: bool = True,
        check_special_chars: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.check_duplicates = check_duplicates
        self.check_special_chars = check_special_chars
        
        self.stats = {
            'total_samples': 0,
            'filtered_too_short': 0,
            'filtered_too_long': 0,
            'filtered_invalid': 0,
            'filtered_duplicate': 0,
            'filtered_special_chars': 0,
            'valid_samples': 0
        }
        
        self.seen_hashes = set() if check_duplicates else None
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a single sample
        
        Args:
            sample: Dictionary containing sample data
        
        Returns:
            True if sample is valid, False otherwise
        """
        self.stats['total_samples'] += 1
        
        # Check required fields
        text = None
        if 'text' in sample:
            text = sample['text']
        elif 'input_ids' in sample:
            # Already tokenized, assume valid
            self.stats['valid_samples'] += 1
            return True
        else:
            self.stats['filtered_invalid'] += 1
            logger.debug("Sample missing 'text' or 'input_ids' field")
            return False
        
        # Check if text is string
        if not isinstance(text, str):
            self.stats['filtered_invalid'] += 1
            logger.debug(f"Text is not string: {type(text)}")
            return False
        
        # Check length
        text_len = len(text)
        if text_len < self.min_length:
            self.stats['filtered_too_short'] += 1
            return False
        
        if text_len > self.max_length:
            self.stats['filtered_too_long'] += 1
            logger.debug(f"Sample too long: {text_len} chars (max: {self.max_length})")
            return False
        
        # Check for duplicates
        if self.check_duplicates:
            text_hash = hash(text)
            if text_hash in self.seen_hashes:
                self.stats['filtered_duplicate'] += 1
                return False
            self.seen_hashes.add(text_hash)
        
        # Check special characters ratio
        if self.check_special_chars:
            if not self._check_special_chars(text):
                self.stats['filtered_special_chars'] += 1
                return False
        
        self.stats['valid_samples'] += 1
        return True
    
    def _check_special_chars(self, text: str) -> bool:
        """Check if text has too many special characters"""
        if not text:
            return False
        
        # Count alphanumeric characters
        alphanumeric = sum(c.isalnum() or c.isspace() for c in text)
        ratio = alphanumeric / len(text)
        
        # Text should be at least 50% alphanumeric + spaces
        return ratio >= 0.5
    
    def validate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a batch of samples
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            List of valid samples
        """
        valid_samples = []
        for sample in batch:
            if self.validate_sample(sample):
                valid_samples.append(sample)
        return valid_samples
    
    def print_stats(self):
        """Print validation statistics"""
        logger.info("=" * 60)
        logger.info("Data Validation Statistics")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {self.stats['total_samples']}")
        logger.info(f"Valid samples: {self.stats['valid_samples']}")
        logger.info(f"Filtered (too short): {self.stats['filtered_too_short']}")
        logger.info(f"Filtered (too long): {self.stats['filtered_too_long']}")
        logger.info(f"Filtered (invalid format): {self.stats['filtered_invalid']}")
        
        if self.check_duplicates:
            logger.info(f"Filtered (duplicates): {self.stats['filtered_duplicate']}")
        
        if self.check_special_chars:
            logger.info(f"Filtered (special chars): {self.stats['filtered_special_chars']}")
        
        if self.stats['total_samples'] > 0:
            valid_pct = 100 * self.stats['valid_samples'] / self.stats['total_samples']
            logger.info(f"Validation rate: {valid_pct:.2f}%")
        
        logger.info("=" * 60)
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics"""
        for key in self.stats:
            self.stats[key] = 0
        if self.seen_hashes is not None:
            self.seen_hashes.clear()


class TokenValidator:
    """Validate tokenized data"""
    
    def __init__(self, vocab_size: int, pad_token_id: int = 0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
    
    def validate_tokens(self, input_ids: List[int]) -> bool:
        """Validate token IDs are within vocabulary"""
        if not input_ids:
            return False
        
        # Check all tokens are in vocabulary
        for token_id in input_ids:
            if not (0 <= token_id < self.vocab_size):
                logger.warning(f"Invalid token ID: {token_id} (vocab_size: {self.vocab_size})")
                return False
        
        # Check not all padding
        if all(t == self.pad_token_id for t in input_ids):
            return False
        
        return True
    
    def get_token_stats(self, input_ids: List[int]) -> Dict[str, Any]:
        """Get statistics about tokens"""
        if not input_ids:
            return {}
        
        unique_tokens = len(set(input_ids))
        pad_tokens = sum(1 for t in input_ids if t == self.pad_token_id)
        
        return {
            'total_tokens': len(input_ids),
            'unique_tokens': unique_tokens,
            'pad_tokens': pad_tokens,
            'vocab_coverage': unique_tokens / self.vocab_size,
            'pad_ratio': pad_tokens / len(input_ids)
        }
