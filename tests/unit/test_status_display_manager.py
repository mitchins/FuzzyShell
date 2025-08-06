"""
Tests for StatusDisplayManager and StatusInfo classes.
"""

import pytest
import urwid
from unittest.mock import patch

from fuzzyshell.status_display_manager import StatusDisplayManager, StatusInfo, StatusFooter


class TestStatusInfo:
    """Test StatusInfo data class."""
    
    def test_init_default_values(self):
        """Test StatusInfo initialization with default values."""
        status = StatusInfo()
        
        assert status.item_count == 0
        assert status.embedding_model == "unknown"
        assert status.search_time == 0.0
        assert status.show_scores is False
        assert status.query == ""
        assert status.results_count == 0
    
    def test_update_single_field(self):
        """Test updating a single field."""
        status = StatusInfo()
        
        status.update(item_count=1500)
        assert status.item_count == 1500
        assert status.embedding_model == "unknown"  # Should remain unchanged
        
        status.update(embedding_model="terminal-minilm")
        assert status.embedding_model == "terminal-minilm"
        assert status.item_count == 1500  # Should remain unchanged
    
    def test_update_multiple_fields(self):
        """Test updating multiple fields at once."""
        status = StatusInfo()
        
        status.update(
            item_count=2500,
            search_time=0.123,
            show_scores=True,
            query="test query"
        )
        
        assert status.item_count == 2500
        assert status.search_time == 0.123
        assert status.show_scores is True
        assert status.query == "test query"
        assert status.embedding_model == "unknown"  # Should remain default
    
    def test_update_with_none_values(self):
        """Test that None values don't update fields."""
        status = StatusInfo()
        original_count = status.item_count
        
        status.update(item_count=None, search_time=0.456)
        
        assert status.item_count == original_count  # Should not change
        assert status.search_time == 0.456  # Should change
    
    def test_update_invalid_field(self):
        """Test updating with invalid field name."""
        status = StatusInfo()
        
        # Should not raise an error, just ignore invalid fields
        status.update(invalid_field="value", item_count=100)
        
        assert status.item_count == 100
        assert not hasattr(status, 'invalid_field')
    
    def test_to_dict(self):
        """Test converting StatusInfo to dictionary."""
        status = StatusInfo()
        status.update(
            item_count=1000,
            embedding_model="test-model",
            search_time=0.25,
            show_scores=True,
            query="search query",
            results_count=42
        )
        
        result_dict = status.to_dict()
        
        expected = {
            'item_count': 1000,
            'embedding_model': 'test-model',
            'search_time': 0.25,
            'show_scores': True,
            'query': 'search query',
            'results_count': 42
        }
        
        assert result_dict == expected


class TestStatusDisplayManager:
    """Test StatusDisplayManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StatusDisplayManager()
    
    def test_init(self):
        """Test StatusDisplayManager initialization."""
        assert isinstance(self.manager.status_info, StatusInfo)
        assert self.manager._footer_widget is None
    
    def test_update_status(self):
        """Test updating status information."""
        self.manager.update_status(item_count=1500, search_time=0.123)
        
        assert self.manager.status_info.item_count == 1500
        assert self.manager.status_info.search_time == 0.123
    
    def test_get_status_info(self):
        """Test getting status information."""
        self.manager.update_status(embedding_model="test-model")
        
        status_info = self.manager.get_status_info()
        assert isinstance(status_info, StatusInfo)
        assert status_info.embedding_model == "test-model"
        assert status_info is self.manager.status_info  # Should be same instance
    
    def test_create_footer_widget(self):
        """Test creating footer widget."""
        footer_widget = self.manager.create_footer_widget()
        
        assert isinstance(footer_widget, urwid.AttrMap)
        assert hasattr(self.manager, '_left_text_widget')
        assert hasattr(self.manager, '_right_text_widget')
        assert isinstance(self.manager._left_text_widget, urwid.Text)
        assert isinstance(self.manager._right_text_widget, urwid.Text)
    
    def test_format_status_text_default(self):
        """Test formatting status text with default template."""
        self.manager.update_status(item_count=1000, search_time=0.123)
        
        result = self.manager.format_status_text()
        
        assert "FuzzyShell" in result
        assert "1,000 items" in result
        assert "123ms" in result
    
    def test_format_status_text_custom_template(self):
        """Test formatting status text with custom template."""
        self.manager.update_status(
            item_count=500,
            embedding_model="custom-model",
            query="test search"
        )
        
        template = "Model: {embedding_model}, Items: {item_count}, Query: {query}"
        result = self.manager.format_status_text(template)
        
        assert result == "Model: custom-model, Items: 500, Query: test search"
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        self.manager.update_status(
            item_count=2000,
            search_time=0.0567,  # Will be converted to 56.7ms
            show_scores=True,
            query="summary test",
            results_count=25
        )
        
        stats = self.manager.get_summary_stats()
        
        expected_keys = [
            'version', 'total_items', 'embedding_model', 
            'last_search_time_ms', 'scores_visible',
            'current_query', 'current_results'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_items'] == 2000
        assert stats['last_search_time_ms'] == 56.7
        assert stats['scores_visible'] is True
        assert stats['current_query'] == "summary test"
        assert stats['current_results'] == 25
    
    def test_reset_search_status(self):
        """Test resetting search-related status."""
        # Set some search status
        self.manager.update_status(
            search_time=0.123,
            query="test query",
            results_count=10
        )
        
        # Reset search status
        self.manager.reset_search_status()
        
        # Search-related fields should be reset
        assert self.manager.status_info.search_time == 0.0
        assert self.manager.status_info.query == ""
        assert self.manager.status_info.results_count == 0
        
        # Non-search fields should remain
        # (assuming item_count was set previously)
    
    def test_generate_key_bindings(self):
        """Test generating key binding text."""
        key_bindings = self.manager._generate_key_bindings()
        
        assert isinstance(key_bindings, list)
        assert len(key_bindings) > 0
        
        # Convert the key_bindings list to text by extracting strings from tuples
        text_parts = []
        for item in key_bindings:
            if isinstance(item, tuple) and len(item) >= 2:
                text_parts.append(item[1])  # Extract text from (style, text) tuple
            elif isinstance(item, str):
                text_parts.append(item)
        
        combined_text = " ".join(text_parts)
        
        assert "tips" in combined_text
        assert "expert" in combined_text
        assert "quit" in combined_text
        assert "navigate" in combined_text
        assert "select" in combined_text
        assert "scores" in combined_text
    
    def test_generate_key_bindings_scores_on(self):
        """Test key bindings with scores enabled."""
        self.manager.update_status(show_scores=True)
        
        key_bindings = self.manager._generate_key_bindings()
        
        # Extract text from tuples
        text_parts = []
        for item in key_bindings:
            if isinstance(item, tuple) and len(item) >= 2:
                text_parts.append(item[1])
            elif isinstance(item, str):
                text_parts.append(item)
        
        combined_text = " ".join(text_parts)
        assert "scores(ON)" in combined_text
    
    def test_generate_key_bindings_scores_off(self):
        """Test key bindings with scores disabled."""
        self.manager.update_status(show_scores=False)
        
        key_bindings = self.manager._generate_key_bindings()
        
        # Extract text from tuples
        text_parts = []
        for item in key_bindings:
            if isinstance(item, tuple) and len(item) >= 2:
                text_parts.append(item[1])
            elif isinstance(item, str):
                text_parts.append(item)
        
        combined_text = " ".join(text_parts)
        assert "scores(OFF)" in combined_text
    
    @patch('fuzzyshell.status_display_manager.__version__', '1.2.3')
    def test_version_display(self):
        """Test that version is displayed correctly."""
        result = self.manager.format_status_text()
        assert "FuzzyShell v1.2.3" in result
        
        stats = self.manager.get_summary_stats()
        assert stats['version'] == '1.2.3'


class TestStatusFooterLegacyCompatibility:
    """Test StatusFooter legacy compatibility wrapper."""
    
    def test_init(self):
        """Test StatusFooter initialization."""
        footer = StatusFooter()
        
        # Should have all the legacy properties
        assert footer.item_count == 0
        assert footer.embedding_model == "unknown"
        assert footer.search_time == 0.0
        assert footer.show_scores is False
        
        # Should be a urwid widget
        assert isinstance(footer, urwid.WidgetWrap)
    
    def test_update_backward_compatibility(self):
        """Test that update method works like the original."""
        footer = StatusFooter()
        
        footer.update(item_count=1500)
        assert footer.item_count == 1500
        
        footer.update(embedding_model="terminal-minilm")
        assert footer.embedding_model == "terminal-minilm"
        
        footer.update(search_time=0.123)
        assert footer.search_time == 0.123
        
        footer.update(show_scores=True)
        assert footer.show_scores is True
    
    def test_update_multiple_values_legacy(self):
        """Test updating multiple values at once (legacy compatibility)."""
        footer = StatusFooter()
        
        footer.update(
            item_count=2000,
            embedding_model="stock-minilm-l6",
            search_time=0.456,
            show_scores=True
        )
        
        assert footer.item_count == 2000
        assert footer.embedding_model == "stock-minilm-l6"
        assert footer.search_time == 0.456
        assert footer.show_scores is True
    
    def test_internal_display_manager(self):
        """Test that internal display manager is properly managed."""
        footer = StatusFooter()
        
        # Update through legacy interface
        footer.update(item_count=999, search_time=0.111)
        
        # Check that internal display manager is updated
        status_info = footer._display_manager.get_status_info()
        assert status_info.item_count == 999
        assert status_info.search_time == 0.111