"""
Schema Loader Utility

Loads and caches medical field schemas from YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
import logging

from app.models.schema_validation import FieldSchema

logger = logging.getLogger(__name__)


class SchemaLoader:
    """Loads and manages medical field schemas"""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize schema loader
        
        Args:
            schema_path: Path to field_schemas.yaml file
        """
        if schema_path is None:
            # Default to prompts/medical_terms/field_schemas.yaml
            base_path = Path(__file__).parent.parent
            schema_path = base_path / "prompts" / "medical_terms" / "field_schemas.yaml"
        
        self.schema_path = schema_path
        self._schemas: Dict[str, FieldSchema] = {}
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load schemas from YAML file"""
        try:
            if not self.schema_path.exists():
                logger.error(f"Schema file not found: {self.schema_path}")
                raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
            
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'fields' not in data:
                logger.error("Invalid schema file format: missing 'fields' key")
                raise ValueError("Invalid schema file format: missing 'fields' key")
            
            # Parse each field schema
            for field_key, field_data in data['fields'].items():
                try:
                    schema = FieldSchema(**field_data)
                    self._schemas[field_key] = schema
                    logger.info(f"Loaded schema for field: {field_key}")
                except Exception as e:
                    logger.error(f"Failed to parse schema for field '{field_key}': {e}")
                    raise
            
            logger.info(f"Successfully loaded {len(self._schemas)} field schemas")
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise ValueError(f"Failed to parse schema YAML: {e}")
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")
            raise
    
    def get_field_schema(self, field_type: str) -> Optional[FieldSchema]:
        """
        Get schema for a specific field type
        
        Args:
            field_type: Field type (e.g., 'hbsag', 'treatment_location', 'gender')
        
        Returns:
            FieldSchema if found, None otherwise
        """
        return self._schemas.get(field_type.lower())
    
    def get_all_schemas(self) -> Dict[str, FieldSchema]:
        """Get all loaded schemas"""
        return self._schemas.copy()
    
    def reload_schemas(self) -> None:
        """Reload schemas from file"""
        logger.info("Reloading schemas from file")
        self._schemas.clear()
        self._load_schemas()
    
    def validate_schema_file(self) -> bool:
        """
        Validate that the schema file is properly formatted
        
        Returns:
            True if valid, False otherwise
        """
        try:
            self._load_schemas()
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False


@lru_cache(maxsize=1)
def get_schema_loader() -> SchemaLoader:
    """
    Get cached schema loader instance
    
    Returns:
        SchemaLoader singleton instance
    """
    return SchemaLoader()
