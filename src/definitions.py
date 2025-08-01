from pathlib import Path
import os
from typing import Optional

def find_project_root(current_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by looking for key project files/directories.
    Traverses up the directory tree until it finds the root.
    
    Args:
        current_path: Path to start searching from (defaults to current file's directory)
        
    Returns:
        Path to project root directory
    """
    if current_path is None:
        # Start from the directory this file is in
        current_path = Path(__file__).resolve().parent.parent
    
    # Look for key project indicators
    indicators = [
        'requirements.txt',  # Project dependencies file
        'src',              # Source code directory
        '.git'              # Git repository (if using version control)
    ]
    
    # Check if any of the indicators are in current directory
    if any((current_path / indicator).exists() for indicator in indicators):
        return current_path
    
    # If we hit the root directory without finding indicators, raise an error
    if current_path.parent == current_path:
        raise RuntimeError(
            "Could not find project root directory. "
            "Make sure you're running from within the project directory."
        )
    
    # Recursively check parent directory
    return find_project_root(current_path.parent)

# Project root directory (detected automatically)
PROJECT_ROOT = find_project_root()

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_DIR = DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'

# Source code directory
SRC_DIR = PROJECT_ROOT / 'src'

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Common file paths
#SEPSIS_LOG_PATH = INPUT_DIR / 'sepsis.xes'

def get_output_path(filename: str, log_name: str) -> Path:
    log_output_path = OUTPUT_DIR / log_name
    log_output_path.mkdir(parents=True, exist_ok=True)
    """Get absolute path for an output file."""
    return log_output_path / filename

def get_input_path(filename: str) -> Path:
    """Get absolute path for an input file."""
    return INPUT_DIR / filename 