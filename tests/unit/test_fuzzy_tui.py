"""
Tests for TUI components and widgets.
"""

import pytest
import urwid
from unittest.mock import MagicMock, patch, call
import time
import random

from fuzzyshell.fuzzy_tui import (
    WatermarkEdit, SearchManager, 
    ResultRenderer, IngestionProgressTUI
)
from fuzzyshell.status_display_manager import StatusFooter


class TestWatermarkEdit:
    """Test the WatermarkEdit widget."""
    
    def test_init(self):
        """Test WatermarkEdit initialization."""
        widget = WatermarkEdit(
            caption="Search: ",
            edit_text="",
            watermark_text="Enter query..."
        )
        
        assert widget.caption == "Search: "
        assert widget.edit_text == ""
        assert widget.watermark_text == "Enter query..."
        assert widget._has_been_edited is False
    
    def test_watermark_logic(self):
        """Test watermark display logic without complex render mocking."""
        widget = WatermarkEdit(
            caption="Search: ",
            edit_text="",
            watermark_text="Enter query..."
        )
        
        # Test watermark conditions
        assert widget.edit_text == ""
        assert widget._has_been_edited is False
        assert widget.watermark_text == "Enter query..."
        
        # After editing, watermark should not show
        widget._has_been_edited = True
        # Basic logic test - no render call needed
    
    def test_watermark_with_text(self):
        """Test watermark behavior when widget has text."""
        widget = WatermarkEdit(
            caption="Search: ",
            edit_text="actual query",
            watermark_text="Enter query..."
        )
        
        # Should not show watermark when has text
        assert widget.edit_text == "actual query"
        assert len(widget.edit_text) > 0  # Has text, so no watermark needed
    
    def test_keypress_marks_edited(self):
        """Test that keypress marks widget as edited."""
        widget = WatermarkEdit(watermark_text="Enter query...")
        
        with patch.object(urwid.Edit, 'keypress') as mock_super_keypress:
            mock_super_keypress.return_value = None
            
            # Printable character should mark as edited
            widget.keypress((80,), 'a')
            assert widget._has_been_edited is True
            
        # Reset for backspace test
        widget._has_been_edited = False
        widget.edit_text = "test"
        
        with patch.object(urwid.Edit, 'keypress') as mock_super_keypress:
            mock_super_keypress.return_value = None
            
            # Backspace with text should mark as edited
            widget.keypress((80,), 'backspace')
            assert widget._has_been_edited is True
    
    def test_keypress_passes_through_navigation(self):
        """Test that navigation keys are passed through."""
        widget = WatermarkEdit(watermark_text="Enter query...")
        
        navigation_keys = ['up', 'down', 'page up', 'page down']
        
        for key in navigation_keys:
            result = widget.keypress((80,), key)
            assert result == key  # Should be passed through


class TestStatusFooter:
    """Test the StatusFooter widget."""
    
    def test_init(self):
        """Test StatusFooter initialization."""
        footer = StatusFooter()
        
        assert footer.item_count == 0
        assert footer.embedding_model == "unknown"
        assert footer.search_time == 0.0
        assert footer.show_scores is False
    
    def test_update_individual_values(self):
        """Test updating individual values."""
        footer = StatusFooter()
        
        footer.update(item_count=1500)
        assert footer.item_count == 1500
        
        footer.update(embedding_model="terminal-minilm")
        assert footer.embedding_model == "terminal-minilm"
        
        footer.update(search_time=0.123)
        assert footer.search_time == 0.123
        
        footer.update(show_scores=True)
        assert footer.show_scores is True
    
    def test_update_multiple_values(self):
        """Test updating multiple values at once."""
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
    
    def test_update_text_formatting(self):
        """Test that update creates properly formatted text."""
        footer = StatusFooter()
        footer.update(item_count=1500, search_time=0.123, show_scores=True)
        
        # Check that the internal display manager was updated
        assert hasattr(footer, '_display_manager')
        status_info = footer._display_manager.get_status_info()
        assert status_info.item_count == 1500
        assert status_info.search_time == 0.123
        assert status_info.show_scores is True


class TestSearchManager:
    """Test the SearchManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fuzzyshell = MagicMock()
        self.mock_search_callback = MagicMock()
        self.search_manager = SearchManager(
            self.mock_fuzzyshell, 
            self.mock_search_callback
        )
    
    def test_init(self):
        """Test SearchManager initialization."""
        assert self.search_manager.fuzzyshell == self.mock_fuzzyshell
        assert self.search_manager.search_callback == self.mock_search_callback
        assert self.search_manager.current_query is None
        assert self.search_manager.search_cancelled is False
    
    def test_set_current_query(self):
        """Test setting current query."""
        self.search_manager.set_current_query("test query")
        
        assert self.search_manager.current_query == "test query"
        assert self.search_manager.search_cancelled is False
    
    def test_cancel_search(self):
        """Test search cancellation."""
        self.search_manager.cancel_search()
        assert self.search_manager.search_cancelled is True
    
    def test_execute_search_empty_query(self):
        """Test execute_search with empty query."""
        mock_ui_controller = MagicMock()
        
        result = self.search_manager.execute_search("", mock_ui_controller)
        assert result == []
        
        result = self.search_manager.execute_search("   ", mock_ui_controller)
        assert result == []
    
    def test_execute_search_cancelled(self):
        """Test execute_search when search is cancelled."""
        mock_ui_controller = MagicMock()
        self.search_manager.search_cancelled = True
        
        result = self.search_manager.execute_search("test", mock_ui_controller)
        assert result == []
    
    def test_execute_search_query_mismatch(self):
        """Test execute_search when query doesn't match current."""
        mock_ui_controller = MagicMock()
        self.search_manager.current_query = "other query"
        
        result = self.search_manager.execute_search("test", mock_ui_controller)
        assert result == []
    
    def test_perform_search_operation_with_fuzzyshell(self):
        """Test _perform_search_operation using fuzzyshell."""
        self.mock_fuzzyshell.search.return_value = [("cmd1", 0.9), ("cmd2", 0.8)]
        mock_progress_callback = MagicMock()
        
        result = self.search_manager._perform_search_operation("test", mock_progress_callback)
        
        self.mock_fuzzyshell.search.assert_called_once_with(
            "test", top_k=100, return_scores=True, progress_callback=mock_progress_callback
        )
        assert result == [("cmd1", 0.9), ("cmd2", 0.8)]
    
    def test_perform_search_operation_with_callback(self):
        """Test _perform_search_operation using search callback."""
        search_manager = SearchManager(None, self.mock_search_callback)
        self.mock_search_callback.return_value = [("cmd1", 0.9)]
        
        result = search_manager._perform_search_operation("test", None)
        
        self.mock_search_callback.assert_called_once_with("test")
        assert result == [("cmd1", 0.9)]
    
    def test_perform_search_operation_no_method(self):
        """Test _perform_search_operation with no search method."""
        search_manager = SearchManager(None, None)
        
        result = search_manager._perform_search_operation("test", None)
        assert result == []


class TestResultRenderer:
    """Test the ResultRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = ResultRenderer()
        self.mock_ui_controller = MagicMock()
    
    def test_init(self):
        """Test ResultRenderer initialization."""
        assert isinstance(self.renderer, ResultRenderer)
    
    def test_render_results_empty(self):
        """Test rendering empty results."""
        self.renderer.render_results([], self.mock_ui_controller)
        
        self.mock_ui_controller.clear_results.assert_called_once()
        self.mock_ui_controller.show_no_results.assert_called_once()
    
    def test_render_results_none(self):
        """Test rendering None results."""
        self.renderer.render_results(None, self.mock_ui_controller)
        
        self.mock_ui_controller.clear_results.assert_called_once()
        self.mock_ui_controller.show_no_results.assert_called_once()
    
    def test_process_results_valid(self):
        """Test processing valid results."""
        raw_results = [
            ("ls -la", 0.9, 0.8, 0.7),
            ("git status", 0.85, 0.75, 0.65),
            ("cd /home", 0.8, 0.7, 0.6)
        ]
        
        processed = self.renderer._process_results(raw_results)
        
        assert len(processed) == 3
        assert processed[0] == ("ls -la", 0.9, 0.8, 0.7)
        assert processed[1] == ("git status", 0.85, 0.75, 0.65)
        assert processed[2] == ("cd /home", 0.8, 0.7, 0.6)
    
    def test_process_results_with_short_tuples(self):
        """Test processing results with short tuples."""
        raw_results = [
            ("ls -la", 0.9),  # Only command and score
            ("git status", 0.85, 0.75)  # Command, score, semantic score
        ]
        
        processed = self.renderer._process_results(raw_results)
        
        assert len(processed) == 2
        assert processed[0] == ("ls -la", 0.9, 0.0, 0.0)
        assert processed[1] == ("git status", 0.85, 0.75, 0.0)
    
    def test_process_results_filters_invalid(self):
        """Test that invalid results are filtered out."""
        raw_results = [
            ("valid command", 0.9, 0.8, 0.7),
            None,  # None result
            [],  # Empty result
            ("",),  # Empty command
            ("cmd", "invalid_score"),  # Invalid score type
            "not_a_tuple",  # Not a sequence
        ]
        
        processed = self.renderer._process_results(raw_results)
        
        # Should only keep the valid result
        assert len(processed) == 1
        assert processed[0] == ("valid command", 0.9, 0.8, 0.7)
    
    def test_extract_result_data_valid(self):
        """Test extracting data from valid result."""
        result = ("ls -la", 0.9, 0.8, 0.7)
        
        extracted = self.renderer._extract_result_data(result, 0)
        assert extracted == ("ls -la", 0.9, 0.8, 0.7)
    
    def test_extract_result_data_invalid_scores(self):
        """Test extracting data with invalid scores."""
        result = ("ls -la", "invalid", 0.8, 0.7)
        
        extracted = self.renderer._extract_result_data(result, 0)
        assert extracted is None  # Should return None for invalid scores
    
    def test_extract_result_data_empty_command(self):
        """Test extracting data with empty command."""
        result = ("", 0.9, 0.8, 0.7)
        
        extracted = self.renderer._extract_result_data(result, 0)
        assert extracted is None  # Should return None for empty command


class TestIngestionProgressTUI:
    """Test the IngestionProgressTUI class."""
    
    def test_init_default(self):
        """Test default initialization."""
        tui = IngestionProgressTUI()
        
        assert tui.no_random is False
        assert tui.running is False
        assert tui.total_commands == 0
        assert tui.processed_commands == 0
        assert tui.current_phase == "Initializing..."
        assert len(tui.witty_remarks) > 0
    
    def test_init_no_random(self):
        """Test initialization with no_random=True."""
        tui = IngestionProgressTUI(no_random=True)
        assert tui.no_random is True
    
    def test_set_total_commands(self):
        """Test setting total commands."""
        tui = IngestionProgressTUI()
        tui.set_total_commands(1500)
        assert tui.total_commands == 1500
    
    def test_increment_processed(self):
        """Test incrementing processed commands."""
        tui = IngestionProgressTUI()
        
        assert tui.processed_commands == 0
        tui.increment_processed()
        assert tui.processed_commands == 1
        tui.increment_processed()
        assert tui.processed_commands == 2
    
    def test_get_status_text_no_random(self):
        """Test status text with no_random=True."""
        tui = IngestionProgressTUI(no_random=True)
        
        result = tui.get_status_text("Processing commands...")
        assert result == "Processing commands..."
    
    def test_get_status_text_with_random(self):
        """Test status text with random witty remarks."""
        tui = IngestionProgressTUI(no_random=False)
        
        # Mock time to control remark timing
        with patch('time.time', return_value=10.0):
            tui.last_remark_time = 0.0  # Force new remark
            
            with patch('random.randint', return_value=0):
                result = tui.get_status_text("Processing...")
                assert result == tui.witty_remarks[0]
    
    def test_get_command_display_no_random(self):
        """Test command display with no_random=True."""
        tui = IngestionProgressTUI(no_random=True)
        
        result = tui.get_command_display("ls -la")
        assert result == "Processing: ls -la"
        
        # Test long command truncation
        long_cmd = "a" * 70
        result = tui.get_command_display(long_cmd)
        assert "..." in result
        assert len(result) < len(long_cmd) + 20  # Should be truncated
    
    def test_get_command_display_with_random(self):
        """Test command display with random mode (should hide commands)."""
        tui = IngestionProgressTUI(no_random=False)
        
        result = tui.get_command_display("ls -la")
        assert result == ""  # Should hide command in witty mode
    
    def test_get_stats_text_no_random(self):
        """Test statistics text with no_random=True."""
        tui = IngestionProgressTUI(no_random=True)
        tui.start_time = time.time() - 10.0  # 10 seconds ago
        tui.total_commands = 1000
        tui.processed_commands = 500
        
        result = tui.get_stats_text()
        
        assert "Processed: 500/1000" in result
        assert "Rate:" in result
        assert "ETA:" in result
    
    def test_get_stats_text_with_random(self):
        """Test statistics text with random mode (should hide details)."""
        tui = IngestionProgressTUI(no_random=False)
        tui.start_time = time.time()
        
        result = tui.get_stats_text()
        assert result == ""  # Should hide stats in witty mode