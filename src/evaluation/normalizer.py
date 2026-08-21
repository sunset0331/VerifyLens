import re
from typing import Optional, Any

def normalize_string(value: Any) -> Optional[str]:
    """
    Apply shared, strict text normalization layer before comparison.
    """
    if value is None:
        return None
        
    s = str(value)
    
    # Lowercase
    s = s.lower()
    
    # Replace common date separators with slash for consistency
    # (assuming DD/MM/YYYY is the ground truth format in our dataset)
    s = re.sub(r'(\d{2})-(\d{2})-(\d{4})', r'\1/\2/\3', s)
    s = re.sub(r'(\d{2})\.(\d{2})\.(\d{4})', r'\1/\2/\3', s)
    
    # Strip whitespace
    s = s.strip()
    
    # Normalize repeated spaces to a single space
    s = re.sub(r'\s+', ' ', s)
    
    # Optional: strip punctuation if it causes noise, but we keep it minimal
    # s = re.sub(r'[^\w\s/]', '', s)
    
    if s == "" or s == "none" or s == "null":
        return None
        
    return s

def normalize_dict(data: dict) -> dict:
    """Normalize all string values in a dictionary."""
    normalized = {}
    for k, v in data.items():
        normalized[k] = normalize_string(v)
    return normalized
