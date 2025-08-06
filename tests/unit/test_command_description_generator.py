#!/usr/bin/env python3
"""
Unit tests for CommandDescriptionGenerator class.

Tests simplified synchronous command description generation with caching
and proper state management for TUI display widgets.
"""

import pytest
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.command_description_generator import CommandDescriptionGenerator


@pytest.fixture
def mock_description_handler():
    """Create mock DescriptionHandler."""
    handler = Mock()
    handler.generate_description.return_value = "Mock description for test command"
    return handler


@pytest.fixture
def description_generator():
    """Create CommandDescriptionGenerator instance."""
    return CommandDescriptionGenerator()


def test_initialization(description_generator):
    """Test CommandDescriptionGenerator initializes correctly."""
    assert description_generator._description_handler is None
    assert description_generator._current_command is None
    assert description_generator.description == ""
    assert description_generator._cache == {}


def test_get_display_text_initial_state(description_generator):
    """Test initial display text shows placeholder."""
    text = description_generator.get_display_text()
    
    assert isinstance(text, list)
    assert len(text) == 2
    assert text[0] == ('dark gray', "💡 ")
    assert text[1] == ('dark gray', "Select a command to see its description")


def test_get_display_text_no_description(description_generator):
    """Test display text when no description is set."""
    text = description_generator.get_display_text()
    
    assert isinstance(text, list)
    assert len(text) == 2
    assert text[0] == ('dark gray', "💡 ")
    assert text[1] == ('dark gray', "Select a command to see its description")


def test_update_command_new_command(description_generator):
    """Test updating to a new command generates description."""
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("ls -la")
        
        assert description_generator._current_command == "ls -la"
        mock_generate.assert_called_once_with("ls -la")


def test_get_display_text_with_description(description_generator):
    """Test display text with actual description."""
    description_generator.description = "Test command description"
    
    text = description_generator.get_display_text()
    
    assert isinstance(text, list)
    assert len(text) == 2
    assert text[0] == ('light blue', "💡 ")
    assert text[1] == ('white', "Test command description")


def test_clear(description_generator):
    """Test clearing description resets state."""
    # Set some state
    description_generator.description = "Some description"
    description_generator._current_command = "test"
    
    description_generator.clear()
    
    assert description_generator.description == ""
    assert description_generator._current_command is None


def test_update_command_same_command(description_generator):
    """Test updating to same command does nothing."""
    description_generator._current_command = "ls -la"
    description_generator.description = "Original description"
    
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("ls -la")
        
        # Should not call generate since it's the same command
        mock_generate.assert_not_called()
        assert description_generator.description == "Original description"


def test_init_description_handler_success(description_generator):
    """Test successful description handler initialization."""
    mock_handler = Mock()
    # Set handler directly to test the logic
    description_generator._description_handler = mock_handler
    
    handler = description_generator._init_description_handler()
    
    assert handler == mock_handler
    assert description_generator._description_handler == mock_handler


def test_init_description_handler_failure(description_generator):
    """Test description handler initialization failure."""
    # Set to False to simulate previous failure
    description_generator._description_handler = False
    
    handler = description_generator._init_description_handler()
    
    assert handler is None
    assert description_generator._description_handler is False


def test_init_description_handler_already_initialized(description_generator, mock_description_handler):
    """Test handler initialization when already set."""
    description_generator._description_handler = mock_description_handler
    
    handler = description_generator._init_description_handler()
    
    assert handler == mock_description_handler


def test_init_description_handler_failed_before(description_generator):
    """Test handler initialization when previously failed."""
    description_generator._description_handler = False
    
    handler = description_generator._init_description_handler()
    
    assert handler is None


def test_update_command_empty_command(description_generator):
    """Test updating with empty command does nothing."""
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("")
        
        mock_generate.assert_not_called()
        assert description_generator._current_command is None


def test_update_command_uses_cache(description_generator):
    """Test updating with cached command uses cache."""
    # Set up cache
    description_generator._cache["ls -la"] = "Cached description"
    
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("ls -la")
        
        # Should use cache, not generate
        mock_generate.assert_not_called()
        assert description_generator.description == "Cached description"


def test_generate_description_with_handler(description_generator, mock_description_handler):
    """Test description generation with handler."""
    description_generator._description_handler = mock_description_handler
    
    description_generator._generate_description("test cmd")
    
    mock_description_handler.generate_description.assert_called_once_with("test cmd")
    assert description_generator.description == "Mock description for test command"
    assert description_generator._cache["test cmd"] == "Mock description for test command"


def test_generate_description_without_handler(description_generator):
    """Test description generation without handler uses fallback."""
    with patch.object(description_generator, '_init_description_handler', return_value=None):
        description_generator._generate_description("test cmd")
        
        assert description_generator.description == "Command: test cmd"


def test_generate_description_with_error(description_generator, mock_description_handler):
    """Test description generation handles errors gracefully."""
    mock_description_handler.generate_description.side_effect = Exception("Test error")
    description_generator._description_handler = mock_description_handler
    
    description_generator._generate_description("test cmd")
    
    # Should fallback to simple description
    assert description_generator.description == "Command: test cmd"


def test_init_description_handler_first_time(description_generator):
    """Test description handler initialization on first call."""
    mock_handler_instance = Mock()
    
    # Mock the import process
    with patch('builtins.__import__') as mock_import:
        mock_model_handler = Mock()
        mock_model_handler.DescriptionHandler = Mock(return_value=mock_handler_instance)
        mock_import.return_value = mock_model_handler
        
        handler = description_generator._init_description_handler()
        
        assert handler == mock_handler_instance
        assert description_generator._description_handler == mock_handler_instance
        mock_model_handler.DescriptionHandler.assert_called_once()


def test_init_description_handler_initialization_error(description_generator):
    """Test description handler initialization error."""
    # Mock the import to fail
    with patch('builtins.__import__', side_effect=ImportError("Import failed")):
        handler = description_generator._init_description_handler()
        
        assert handler is None
        assert description_generator._description_handler is False


def test_get_cached_description(description_generator):
    """Test getting cached description."""
    description_generator._cache["test cmd"] = "Cached description"
    
    result = description_generator.get_cached_description("test cmd")
    assert result == "Cached description"
    
    result = description_generator.get_cached_description("unknown cmd")
    assert result is None


def test_clear_cache(description_generator):
    """Test clearing the description cache."""
    description_generator._cache["cmd1"] = "desc1"
    description_generator._cache["cmd2"] = "desc2"
    
    description_generator.clear_cache()
    
    assert description_generator._cache == {}


def test_update_command_integration(description_generator, mock_description_handler):
    """Test full update_command integration."""
    with patch.object(description_generator, '_init_description_handler', return_value=mock_description_handler):
        description_generator.update_command("git status")
        
        assert description_generator._current_command == "git status"
        assert description_generator.description == "Mock description for test command"
        assert description_generator._cache["git status"] == "Mock description for test command"


def test_display_text_formatting(description_generator):
    """Test display text formatting for different states."""
    # No description
    text = description_generator.get_display_text()
    assert text[0] == ('dark gray', "💡 ")
    assert text[1] == ('dark gray', "Select a command to see its description")
    
    # With description
    description_generator.description = "Test description"
    text = description_generator.get_display_text()
    assert text[0] == ('light blue', "💡 ")
    assert text[1] == ('white', "Test description")


def test_thread_safety_initialization(description_generator):
    """Test thread-safe initialization of description handler."""
    # Test that _init_lock is used properly
    assert hasattr(description_generator, '_init_lock')
    assert description_generator._init_lock is not None


def test_caching_behavior(description_generator):
    """Test caching behavior with multiple commands."""
    # First call should generate and cache
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("ls -la")
        mock_generate.assert_called_once_with("ls -la")
    
    # Reset current command to simulate new update
    description_generator._current_command = None
    
    # Set up the cache as if description was generated
    description_generator._cache["ls -la"] = "List files"
    
    # Second call should use cache
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("ls -la")
        mock_generate.assert_not_called()
        assert description_generator.description == "List files"


def test_error_handling_robustness(description_generator):
    """Test error handling in various scenarios."""
    # Test with None command
    description_generator.update_command(None)
    assert description_generator._current_command is None
    
    # Test with handler initialization error
    with patch.object(description_generator, '_init_description_handler', return_value=None):
        description_generator._generate_description("test")
        assert description_generator.description == "Command: test"


def test_cache_persistence(description_generator, mock_description_handler):
    """Test that cache persists across multiple operations."""
    description_generator._description_handler = mock_description_handler
    
    # Generate description for first command
    description_generator._generate_description("cmd1")
    assert "cmd1" in description_generator._cache
    
    # Generate description for second command
    description_generator._generate_description("cmd2")
    assert "cmd2" in description_generator._cache
    
    # Both should be in cache
    assert len(description_generator._cache) == 2
    assert description_generator._cache["cmd1"] == "Mock description for test command"
    assert description_generator._cache["cmd2"] == "Mock description for test command"


def test_command_variations_handling(description_generator):
    """Test handling of various command formats."""
    test_commands = [
        "ls -la",
        "git status --porcelain",
        "docker run -it ubuntu:latest",
        "python -m pytest tests/",
        "curl -X POST https://api.example.com"
    ]
    
    for cmd in test_commands:
        with patch.object(description_generator, '_generate_description') as mock_generate:
            description_generator.update_command(cmd)
            mock_generate.assert_called_once_with(cmd)
            assert description_generator._current_command == cmd


def test_state_consistency(description_generator):
    """Test that internal state remains consistent."""
    # Initial state
    assert description_generator._current_command is None
    assert description_generator.description == ""
    assert description_generator._cache == {}
    
    # After update
    description_generator._current_command = "test"
    description_generator.description = "test desc"
    description_generator._cache["test"] = "test desc"
    
    # After clear
    description_generator.clear()
    assert description_generator._current_command is None
    assert description_generator.description == ""
    # Cache should remain (only clear() method clears cache separately)


def test_fallback_description_format(description_generator):
    """Test fallback description format for various commands."""
    test_cases = [
        ("ls -la", "Command: ls -la"),
        ("git status", "Command: git status"),
        ("", "Command: "),
        ("very long command with many arguments", "Command: very long command with many arguments")
    ]
    
    for command, expected in test_cases:
        with patch.object(description_generator, '_init_description_handler', return_value=None):
            description_generator._generate_description(command)
            assert description_generator.description == expected


def test_description_persistence(description_generator):
    """Test that descriptions persist until explicitly cleared or changed."""
    description_generator.description = "Persistent description"
    
    # Should remain after multiple get_display_text calls
    for _ in range(5):
        text = description_generator.get_display_text()
        assert text[1] == ('white', "Persistent description")
    
    # Should remain until explicitly cleared
    description_generator.clear()
    text = description_generator.get_display_text()
    assert text[1] == ('dark gray', "Select a command to see its description")


@pytest.mark.parametrize("command,expected_fallback", [
    ("ls -la", "Command: ls -la"),
    ("git status", "Command: git status"),
    ("", "Command: "),
    ("very long command with many arguments", "Command: very long command with many arguments")
])
def test_fallback_descriptions(description_generator, command, expected_fallback):
    """Test fallback description generation."""
    # Test the fallback logic directly
    with patch.object(description_generator, '_init_description_handler', return_value=None):
        description_generator._generate_description(command)
        
        # Check that fallback description was set
        assert description_generator.description == expected_fallback


def test_description_caching_optimization(description_generator, mock_description_handler):
    """Test that cached commands don't regenerate descriptions."""
    description_generator._cache["test cmd"] = "cached description"
    
    with patch.object(description_generator, '_generate_description') as mock_generate:
        description_generator.update_command("test cmd")
        
        # Should not call generate since it's cached
        mock_generate.assert_not_called()
        assert description_generator.description == "cached description"


def test_immediate_update_behavior(description_generator):
    """Test that updates happen immediately (synchronous behavior)."""
    with patch.object(description_generator, '_generate_description') as mock_generate:
        # Should generate immediately when called
        description_generator.update_command("immediate cmd")
        
        # Should be called synchronously
        mock_generate.assert_called_once_with("immediate cmd")
        assert description_generator._current_command == "immediate cmd"


def test_synchronous_operation(description_generator, mock_description_handler):
    """Test that all operations are synchronous and immediate."""
    description_generator._description_handler = mock_description_handler
    
    # Should complete immediately
    description_generator.update_command("sync test")
    
    # Description should be available immediately
    assert description_generator.description == "Mock description for test command"
    assert description_generator._current_command == "sync test"
    assert "sync test" in description_generator._cache