"""
Utility functions for ResuMatch AI
Common helper functions and utilities
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level
        log_file (str, optional): Log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_file_upload(filename: str, file_size: int, max_size: int = 16 * 1024 * 1024) -> Dict[str, Any]:
    """
    Validate uploaded file
    
    Args:
        filename (str): Name of the uploaded file
        file_size (int): Size of the file in bytes
        max_size (int): Maximum allowed file size
        
    Returns:
        Dict[str, Any]: Validation result
    """
    result = {
        'valid': True,
        'errors': [],
        'file_type': None
    }
    
    # Check if file has extension
    if not filename or '.' not in filename:
        result['valid'] = False
        result['errors'].append('No file extension found')
        return result
    
    # Get file extension
    file_ext = filename.rsplit('.', 1)[1].lower()
    result['file_type'] = file_ext
    
    # Check file extension
    allowed_extensions = {'pdf', 'docx', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}
    if file_ext not in allowed_extensions:
        result['valid'] = False
        result['errors'].append(f'Invalid file type: {file_ext}. Allowed types: {", ".join(allowed_extensions)}')
    
    # Check file size
    if file_size > max_size:
        result['valid'] = False
        result['errors'].append(f'File too large: {file_size / 1024 / 1024:.2f}MB. Maximum size: {max_size / 1024 / 1024:.2f}MB')
    
    return result

def format_similarity_score(score: float) -> str:
    """
    Format similarity score for display
    
    Args:
        score (float): Similarity score (0-1)
        
    Returns:
        str: Formatted score string
    """
    return f"{score * 100:.1f}%"

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis
    
    Args:
        text (str): Input text
        max_keywords (int): Maximum number of keywords to return
        
    Returns:
        List[str]: List of keywords
    """
    import re
    from collections import Counter
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man'
    }
    
    words = [word for word in words if word not in stop_words]
    
    # Count frequency and return top keywords
    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(max_keywords)]

def create_response(success: bool, data: Any = None, message: str = None, error: str = None) -> Dict[str, Any]:
    """
    Create standardized API response
    
    Args:
        success (bool): Whether the operation was successful
        data (Any, optional): Response data
        message (str, optional): Success message
        error (str, optional): Error message
        
    Returns:
        Dict[str, Any]: Standardized response
    """
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    if message:
        response['message'] = message
    
    if error:
        response['error'] = error
    
    return response

def handle_exception(e: Exception, context: str = None) -> Dict[str, Any]:
    """
    Handle exceptions and create error response
    
    Args:
        e (Exception): The exception that occurred
        context (str, optional): Context where the exception occurred
        
    Returns:
        Dict[str, Any]: Error response
    """
    error_info = {
        'type': type(e).__name__,
        'message': str(e),
        'traceback': traceback.format_exc()
    }
    
    if context:
        error_info['context'] = context
    
    return create_response(
        success=False,
        error=f"{error_info['type']}: {error_info['message']}",
        data=error_info
    )

def save_analysis_result(result: Dict[str, Any], filename: str = None) -> str:
    """
    Save analysis result to file
    
    Args:
        result (Dict[str, Any]): Analysis result
        filename (str, optional): Custom filename
        
    Returns:
        str: Path to saved file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_result_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath

def load_analysis_result(filepath: str) -> Dict[str, Any]:
    """
    Load analysis result from file
    
    Args:
        filepath (str): Path to the result file
        
    Returns:
        Dict[str, Any]: Analysis result
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dict[str, Any]: System information
    """
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'node': platform.node(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version()
    }

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_text_for_display(text: str, max_length: int = 200) -> str:
    """
    Clean text for display purposes
    
    Args:
        text (str): Input text
        max_length (int): Maximum length for display
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text
