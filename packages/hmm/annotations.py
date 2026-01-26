"""
Annotation loaders and adapters for HMM search results.

Responsibilities:
- Parse and load annotation sources (PHROGs, PFAM)
- Normalize heterogeneous formats into fast-lookup dictionaries
- Lazy-load data to minimize memory footprint and startup time
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

import pandas as pd
import utils.consol_log as log

logger = log.get_logger()


class AnnotationSource(ABC):
    """Abstract base class for defining how to parse specific annotation files."""

    @abstractmethod
    def load_to_dict(self, path: Path) -> Dict[str, str]:
        """Parse file and return a mapping of ID -> Annotation String."""
        pass

    @abstractmethod
    def normalize_id(self, domain_id: str) -> str:
        """Standardize the incoming domain ID to match the dictionary keys."""
        pass


class PhrogSource(AnnotationSource):
    def load_to_dict(self, path: Path) -> Dict[str, str]:
        # usecols=[0, 6] ensures we only load 'phrog' and 'annot' columns into memory
        try:
            df = pd.read_csv(path, sep="\t", usecols=["phrog", "annot"])
            # Convert numeric IDs to strings to ensure consistent lookup
            df["phrog"] = df["phrog"].astype(str)
            return df.set_index("phrog")["annot"].fillna("-").to_dict()
        except Exception as e:
            logger.error(f"Failed to load PHROG annotations from {path}: {e}")
            return {}

    def normalize_id(self, domain_id: str) -> str:
        # Extracts "123" from "phrog_123"
        if domain_id.startswith("phrog_"):
            return domain_id[len("phrog_") :]
        return domain_id


class PfamSource(AnnotationSource):
    def load_to_dict(self, path: Path) -> Dict[str, str]:
        records = {}
        try:
            with path.open("r") as f:
                current_id = None
                current_desc = None
                for line in f:
                    if line.startswith("#=GF ID"):
                        current_id = line.split(maxsplit=2)[-1].strip()
                    elif line.startswith("#=GF DE"):
                        current_desc = line.split(maxsplit=2)[-1].strip()
                    elif line.startswith("//"):
                        if current_id:
                            records[current_id] = current_desc or "-"
                        current_id, current_desc = None, None
            return records
        except (IOError, ValueError, pd.errors.ParserError) as e:
            logger.error(f"Failed to load Pfam annotations from {path}: {e}")
            return {}

    def normalize_id(self, domain_id: str) -> str:
        # Pfam IDs are usually used as-is
        return domain_id


class AnnotationProvider:
    """
    Unified provider that lazy-loads annotation data.
    """

    _registry: Dict[str, Type[AnnotationSource]] = {"phrogs": PhrogSource, "pfam": PfamSource}

    def __init__(self, source_type: str, file_path: str):
        self.path = Path(file_path)
        self._cache: Optional[Dict[str, str]] = None

        source_class = self._registry.get(source_type.lower())
        if not source_class:
            raise ValueError(
                f"Unsupported source: {source_type}. Supported: {list(self._registry.keys())}"
            )

        self.engine = source_class()

        if not self.path.exists():
            logger.warning(f"Annotation file not found at startup: {file_path}")

    @property
    def data(self) -> Dict[str, str]:
        """Lazy loader: only reads the file from disk on the first access."""
        if self._cache is None:
            if not self.path.exists():
                logger.error(f"Cannot load annotations: {self.path} does not exist.")
                self._cache = {}
            else:
                logger.info(f"Lazy-loading {self.engine.__class__.__name__} from {self.path}")
                self._cache = self.engine.load_to_dict(self.path)
        return self._cache

    def get(self, domain: str) -> str:
        """Return annotation string for a given domain with O(1) complexity."""
        lookup_id = self.engine.normalize_id(domain)
        return self.data.get(lookup_id, "-")
