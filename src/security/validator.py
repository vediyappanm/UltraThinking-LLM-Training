"""Security validation for inputs and configs"""
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation utilities"""
    
    # Dangerous patterns to check
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'os\.system',
        r'subprocess',
        r'open\s*\(',
    ]
    
    @staticmethod
    def check_code_injection(text: str) -> bool:
        """
        Check for potential code injection
        
        Args:
            text: String to check
        
        Returns:
            True if safe, False if dangerous patterns detected
        """
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        return True


def validate_model_path(path: str, allowed_dirs: List[str] = None) -> bool:
    """
    Validate model path to prevent directory traversal
    
    Args:
        path: Path to validate
        allowed_dirs: List of allowed base directories
    
    Returns:
        True if path is safe, False otherwise
    
    Raises:
        ValueError: If path is dangerous
        FileNotFoundError: If path doesn't exist
    """
    # Resolve to absolute path
    try:
        abs_path = Path(path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {path}") from e
    
    # Check for directory traversal
    if '..' in str(abs_path):
        raise ValueError("Directory traversal detected in path")
    
    # Check if path contains suspicious characters
    suspicious_chars = ['|', ';', '&', '$', '`']
    if any(char in str(path) for char in suspicious_chars):
        raise ValueError(f"Suspicious characters in path: {path}")
    
    # Check allowed directories if specified
    if allowed_dirs:
        allowed_dirs_resolved = [Path(d).resolve() for d in allowed_dirs]
        if not any(abs_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs_resolved):
            raise ValueError(f"Path not in allowed directories: {path}")
    
    # Check file exists (optional - comment out if creating new files)
    # if not abs_path.exists():
    #     raise FileNotFoundError(f"Path not found: {path}")
    
    logger.debug(f"Path validated: {abs_path}")
    return True


def sanitize_config(config: dict, max_depth: int = 10) -> dict:
    """
    Sanitize configuration to prevent code injection
    
    Args:
        config: Configuration dictionary
        max_depth: Maximum nesting depth to check
    
    Returns:
        Sanitized configuration
    
    Raises:
        ValueError: If dangerous configuration detected
    """
    dangerous_keys = ['__import__', 'eval', 'exec', 'compile', 'open', 'input']
    
    def check_dict(d: Dict[str, Any], depth: int = 0):
        """Recursively check dictionary"""
        if depth > max_depth:
            raise ValueError(f"Configuration nesting too deep: {depth}")
        
        for key, value in d.items():
            # Check key names
            key_lower = str(key).lower()
            if any(danger in key_lower for danger in dangerous_keys):
                raise ValueError(f"Dangerous configuration key: {key}")
            
            # Check for code in string values
            if isinstance(value, str):
                if not SecurityValidator.check_code_injection(value):
                    raise ValueError(f"Potential code injection in config value: {key}")
            
            # Recursively check nested dicts
            elif isinstance(value, dict):
                check_dict(value, depth + 1)
            
            # Check lists
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        check_dict(item, depth + 1)
                    elif isinstance(item, str):
                        if not SecurityValidator.check_code_injection(item):
                            raise ValueError(f"Potential code injection in config list: {key}")
    
    # Create a copy and validate
    sanitized = config.copy()
    check_dict(sanitized)
    
    logger.debug("Configuration sanitized successfully")
    return sanitized


def validate_file_size(file_path: str, max_size_mb: int = 1000) -> bool:
    """
    Validate file size to prevent resource exhaustion
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
    
    Returns:
        True if file size is acceptable
    
    Raises:
        ValueError: If file is too large
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_mb = path.stat().st_size / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise ValueError(
            f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
        )
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    dangerous_chars = ['..', '/', '\\', '\0', '|', ';', '&', '$', '`', '<', '>']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename
