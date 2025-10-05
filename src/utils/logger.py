"""
Logging utilities for WSL2 Ubuntu environment.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """Set up logger for WSL2 environment."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_logger(component: str) -> logging.Logger:
    """Get project-specific logger."""
    try:
        from ..config import get_config
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = config.LOGS_DIR / f"{component}_{timestamp}.log"
        return setup_logger(
            name=f"mental_health_ai.{component}",
            log_file=str(log_file),
            level=config.LOG_LEVEL
        )
    except:
        return setup_logger(name=f"mental_health_ai.{component}")
