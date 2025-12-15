"""Tests for metadata filter parser and relaxation strategy."""

from datetime import date

import pytest

from app.core.metadata_filter import (
    Category,
    FilterRelaxationStrategy,
    Language,
    MetadataFilter,
    MetadataFilterParser,
    StoreType,
)


class TestMetadataFilter:
    """Tests for MetadataFilter dataclass."""

    def test_empty_filter(self):
        """Test that a filter with all None values is considered empty."""
        filter_obj = MetadataFilter()
        assert filter_obj.is_empty() is True

    def test_non_empty_filter(self):
        """Test that a filter with any value is not empty."""
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)
        assert filter_obj.is_empty() is False

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=None,
            effective_date=date(2024, 1, 15),
            language=None,
        )
        result = filter_obj.to_dict()
        assert result == {
            "store_type": "cafe",
            "effective_date": "2024-01-15",
        }

    def test_to_dict_all_values(self):
        """Test to_dict with all values set."""
        filter_obj = MetadataFilter(
            store_type=StoreType.CONVENIENCE,
            category=Category.REFUND,
            effective_date=date(2024, 6, 1),
            language=Language.KO,
        )
        result = filter_obj.to_dict()
        assert result == {
            "store_type": "convenience",
            "category": "refund",
            "effective_date": "2024-06-01",
            "language": "ko",
        }

    def test_active_filter_count(self):
        """Test counting active filters."""
        filter_obj = MetadataFilter()
        assert filter_obj.active_filter_count() == 0

        filter_obj = MetadataFilter(store_type=StoreType.CAFE)
        assert filter_obj.active_filter_count() == 1

        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=Category.PROMO,
            language=Language.EN,
        )
        assert filter_obj.active_filter_count() == 3


class TestMetadataFilterParser:
    """Tests for MetadataFilterParser."""

    def test_parse_all_none(self):
        """Test parsing with all None values."""
        result = MetadataFilterParser.parse()
        assert result.is_empty() is True

    def test_parse_store_type_valid(self):
        """Test parsing valid store type."""
        result = MetadataFilterParser.parse(store_type="cafe")
        assert result.store_type == StoreType.CAFE

        result = MetadataFilterParser.parse(store_type="CONVENIENCE")
        assert result.store_type == StoreType.CONVENIENCE

    def test_parse_store_type_invalid(self):
        """Test parsing invalid store type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MetadataFilterParser.parse(store_type="invalid_type")
        assert "Invalid store_type" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_parse_category_valid(self):
        """Test parsing valid category."""
        result = MetadataFilterParser.parse(category="refund")
        assert result.category == Category.REFUND

        result = MetadataFilterParser.parse(category="PROMO")
        assert result.category == Category.PROMO

    def test_parse_category_invalid(self):
        """Test parsing invalid category raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MetadataFilterParser.parse(category="unknown")
        assert "Invalid category" in str(exc_info.value)

    def test_parse_effective_date_valid(self):
        """Test parsing valid effective date."""
        result = MetadataFilterParser.parse(effective_date="2024-01-15")
        assert result.effective_date == date(2024, 1, 15)

    def test_parse_effective_date_invalid_format(self):
        """Test parsing invalid date format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MetadataFilterParser.parse(effective_date="15-01-2024")
        assert "Invalid effective_date" in str(exc_info.value)
        assert "YYYY-MM-DD" in str(exc_info.value)

    def test_parse_effective_date_invalid_date(self):
        """Test parsing invalid date raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MetadataFilterParser.parse(effective_date="2024-13-01")
        assert "Invalid effective_date" in str(exc_info.value)

    def test_parse_language_valid(self):
        """Test parsing valid language."""
        result = MetadataFilterParser.parse(language="ko")
        assert result.language == Language.KO

        result = MetadataFilterParser.parse(language="EN")
        assert result.language == Language.EN

    def test_parse_language_invalid(self):
        """Test parsing invalid language raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MetadataFilterParser.parse(language="fr")
        assert "Invalid language" in str(exc_info.value)

    def test_parse_empty_strings_treated_as_none(self):
        """Test that empty strings are treated as None."""
        result = MetadataFilterParser.parse(
            store_type="",
            category="",
            effective_date="",
            language="",
        )
        assert result.is_empty() is True

    def test_parse_full_filter(self):
        """Test parsing a complete filter."""
        result = MetadataFilterParser.parse(
            store_type="apparel",
            category="inventory",
            effective_date="2024-12-01",
            language="ja",
        )
        assert result.store_type == StoreType.APPAREL
        assert result.category == Category.INVENTORY
        assert result.effective_date == date(2024, 12, 1)
        assert result.language == Language.JA


class TestFilterRelaxationStrategy:
    """Tests for FilterRelaxationStrategy."""

    def test_relax_empty_filter(self):
        """Test relaxing an empty filter returns None."""
        filter_obj = MetadataFilter()
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.relaxed_filter is None
        assert result.can_relax_further is False

    def test_relax_single_field_language(self):
        """Test relaxing filter with only language."""
        filter_obj = MetadataFilter(language=Language.KO)
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.relaxed_filter is not None
        assert result.relaxed_filter.is_empty() is True
        assert result.removed_field == "language"
        assert result.original_value == "ko"
        assert result.can_relax_further is False

    def test_relax_order_language_first(self):
        """Test that language is relaxed first."""
        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=Category.REFUND,
            effective_date=date(2024, 1, 1),
            language=Language.EN,
        )
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.removed_field == "language"
        assert result.relaxed_filter.language is None
        assert result.relaxed_filter.store_type == StoreType.CAFE
        assert result.relaxed_filter.category == Category.REFUND
        assert result.relaxed_filter.effective_date == date(2024, 1, 1)
        assert result.can_relax_further is True

    def test_relax_order_effective_date_second(self):
        """Test that effective_date is relaxed second."""
        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=Category.REFUND,
            effective_date=date(2024, 1, 1),
        )
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.removed_field == "effective_date"
        assert result.relaxed_filter.effective_date is None
        assert result.can_relax_further is True

    def test_relax_order_category_third(self):
        """Test that category is relaxed third."""
        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=Category.REFUND,
        )
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.removed_field == "category"
        assert result.relaxed_filter.category is None
        assert result.can_relax_further is True

    def test_relax_order_store_type_last(self):
        """Test that store_type is relaxed last."""
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        assert result.removed_field == "store_type"
        assert result.relaxed_filter.store_type is None
        assert result.can_relax_further is False

    def test_relaxation_message(self):
        """Test getting relaxation message."""
        filter_obj = MetadataFilter(language=Language.KO)
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        message = FilterRelaxationStrategy.get_relaxation_message(result)
        assert "언어" in message
        assert "재검색" in message

    def test_relaxation_message_category(self):
        """Test relaxation message for category."""
        filter_obj = MetadataFilter(category=Category.PROMO)
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        message = FilterRelaxationStrategy.get_relaxation_message(result)
        assert "카테고리" in message

    def test_relaxation_message_no_relaxation(self):
        """Test relaxation message when cannot relax."""
        filter_obj = MetadataFilter()
        result = FilterRelaxationStrategy.relax_once(filter_obj)
        message = FilterRelaxationStrategy.get_relaxation_message(result)
        assert "완화할 수 없습니다" in message


class TestEnumValues:
    """Tests for enum values."""

    def test_store_type_values(self):
        """Test all store type enum values."""
        assert StoreType.CAFE.value == "cafe"
        assert StoreType.CONVENIENCE.value == "convenience"
        assert StoreType.APPAREL.value == "apparel"
        assert StoreType.RESTAURANT.value == "restaurant"
        assert StoreType.RETAIL.value == "retail"

    def test_category_values(self):
        """Test all category enum values."""
        assert Category.REFUND.value == "refund"
        assert Category.PROMO.value == "promo"
        assert Category.INVENTORY.value == "inventory"
        assert Category.CS.value == "cs"
        assert Category.OPERATION.value == "operation"
        assert Category.HR.value == "hr"

    def test_language_values(self):
        """Test all language enum values."""
        assert Language.KO.value == "ko"
        assert Language.EN.value == "en"
        assert Language.JA.value == "ja"
        assert Language.ZH.value == "zh"
