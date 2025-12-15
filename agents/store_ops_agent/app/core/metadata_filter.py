"""Metadata filter parser and applicator for context-aware search.

This module provides:
- MetadataFilter: Data class for filter parameters
- MetadataFilterParser: Validates and parses filter inputs
- FilterRelaxationStrategy: Strategies for relaxing filters when no results found
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class StoreType(str, Enum):
    """Store type enumeration for filtering documents."""

    CAFE = "cafe"
    CONVENIENCE = "convenience"
    APPAREL = "apparel"
    RESTAURANT = "restaurant"
    RETAIL = "retail"


class Category(str, Enum):
    """Document category enumeration."""

    REFUND = "refund"
    PROMO = "promo"
    INVENTORY = "inventory"
    CS = "cs"
    OPERATION = "operation"
    HR = "hr"


class Language(str, Enum):
    """Supported language codes."""

    KO = "ko"
    EN = "en"
    JA = "ja"
    ZH = "zh"


@dataclass
class MetadataFilter:
    """Filter parameters for context-aware document search.

    Attributes:
        store_type: Type of store to filter by.
        category: Document category to filter by.
        effective_date: Date for which documents should be valid.
        language: Language code for document filtering.
    """

    store_type: Optional[StoreType] = None
    category: Optional[Category] = None
    effective_date: Optional[date] = None
    language: Optional[Language] = None

    def is_empty(self) -> bool:
        """Check if all filter fields are None."""
        return all(
            v is None
            for v in [self.store_type, self.category, self.effective_date, self.language]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary, excluding None values."""
        result: dict[str, Any] = {}
        if self.store_type is not None:
            result["store_type"] = self.store_type.value
        if self.category is not None:
            result["category"] = self.category.value
        if self.effective_date is not None:
            result["effective_date"] = self.effective_date.isoformat()
        if self.language is not None:
            result["language"] = self.language.value
        return result

    def active_filter_count(self) -> int:
        """Count the number of active (non-None) filters."""
        return sum(
            1
            for v in [self.store_type, self.category, self.effective_date, self.language]
            if v is not None
        )


class MetadataFilterParser:
    """Parser and validator for metadata filter inputs."""

    @staticmethod
    def parse(
        store_type: Optional[str] = None,
        category: Optional[str] = None,
        effective_date: Optional[str] = None,
        language: Optional[str] = None,
    ) -> MetadataFilter:
        """Parse and validate filter parameters.

        Args:
            store_type: Store type string (e.g., "cafe", "convenience").
            category: Category string (e.g., "refund", "promo").
            effective_date: Date string in ISO format (YYYY-MM-DD).
            language: Language code (e.g., "ko", "en").

        Returns:
            MetadataFilter object with validated values.

        Raises:
            ValueError: If any parameter has an invalid value.
        """
        parsed_store_type = MetadataFilterParser._parse_store_type(store_type)
        parsed_category = MetadataFilterParser._parse_category(category)
        parsed_effective_date = MetadataFilterParser._parse_effective_date(effective_date)
        parsed_language = MetadataFilterParser._parse_language(language)

        return MetadataFilter(
            store_type=parsed_store_type,
            category=parsed_category,
            effective_date=parsed_effective_date,
            language=parsed_language,
        )

    @staticmethod
    def _parse_store_type(value: Optional[str]) -> Optional[StoreType]:
        """Parse store type string to enum."""
        if value is None or value == "":
            return None
        try:
            return StoreType(value.lower())
        except ValueError as e:
            valid_values = [t.value for t in StoreType]
            raise ValueError(
                f"Invalid store_type: '{value}'. Must be one of: {valid_values}"
            ) from e

    @staticmethod
    def _parse_category(value: Optional[str]) -> Optional[Category]:
        """Parse category string to enum."""
        if value is None or value == "":
            return None
        try:
            return Category(value.lower())
        except ValueError as e:
            valid_values = [c.value for c in Category]
            raise ValueError(
                f"Invalid category: '{value}'. Must be one of: {valid_values}"
            ) from e

    @staticmethod
    def _parse_effective_date(value: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if value is None or value == "":
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(
                f"Invalid effective_date: '{value}'. Must be in YYYY-MM-DD format."
            ) from e

    @staticmethod
    def _parse_language(value: Optional[str]) -> Optional[Language]:
        """Parse language code string to enum."""
        if value is None or value == "":
            return None
        try:
            return Language(value.lower())
        except ValueError as e:
            valid_values = [lang.value for lang in Language]
            raise ValueError(
                f"Invalid language: '{value}'. Must be one of: {valid_values}"
            ) from e


@dataclass
class RelaxationResult:
    """Result of filter relaxation.

    Attributes:
        relaxed_filter: The relaxed filter (or None if cannot relax further).
        removed_field: Name of the field that was relaxed.
        original_value: Original value of the relaxed field.
        can_relax_further: Whether further relaxation is possible.
    """

    relaxed_filter: Optional[MetadataFilter]
    removed_field: Optional[str] = None
    original_value: Optional[str] = None
    can_relax_further: bool = False


class FilterRelaxationStrategy:
    """Strategies for relaxing filters when no results found.

    Relaxation priority (least to most important):
    1. language - Remove first (least specific)
    2. effective_date - Remove second
    3. category - Remove third
    4. store_type - Remove last (most important for relevance)
    """

    RELAXATION_ORDER = ["language", "effective_date", "category", "store_type"]

    @classmethod
    def relax_once(cls, filter_obj: MetadataFilter) -> RelaxationResult:
        """Relax the filter by removing one constraint.

        Removes constraints in order of least to most importance.

        Args:
            filter_obj: Current filter to relax.

        Returns:
            RelaxationResult with relaxed filter and metadata.
        """
        if filter_obj.is_empty():
            return RelaxationResult(
                relaxed_filter=None,
                can_relax_further=False,
            )

        # Try to relax each field in order
        for field_name in cls.RELAXATION_ORDER:
            current_value = getattr(filter_obj, field_name)
            if current_value is not None:
                # Create new filter with this field removed
                new_filter = MetadataFilter(
                    store_type=filter_obj.store_type if field_name != "store_type" else None,
                    category=filter_obj.category if field_name != "category" else None,
                    effective_date=filter_obj.effective_date if field_name != "effective_date" else None,
                    language=filter_obj.language if field_name != "language" else None,
                )

                # Format original value for logging
                if hasattr(current_value, "value"):
                    original_str = current_value.value
                elif hasattr(current_value, "isoformat"):
                    original_str = current_value.isoformat()
                else:
                    original_str = str(current_value)

                return RelaxationResult(
                    relaxed_filter=new_filter,
                    removed_field=field_name,
                    original_value=original_str,
                    can_relax_further=not new_filter.is_empty(),
                )

        return RelaxationResult(
            relaxed_filter=None,
            can_relax_further=False,
        )

    @classmethod
    def get_relaxation_message(cls, result: RelaxationResult) -> str:
        """Get a user-friendly message about the filter relaxation.

        Args:
            result: The relaxation result.

        Returns:
            Message describing the relaxation.
        """
        if result.removed_field is None:
            return "필터를 더 이상 완화할 수 없습니다."

        field_names_ko = {
            "language": "언어",
            "effective_date": "유효일자",
            "category": "카테고리",
            "store_type": "매장 유형",
        }

        field_name_ko = field_names_ko.get(result.removed_field, result.removed_field)
        return f"검색 결과가 없어 '{field_name_ko}' 필터를 제외하고 재검색합니다."
