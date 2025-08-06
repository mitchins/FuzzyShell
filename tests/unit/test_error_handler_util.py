#!/usr/bin/env python3
"""
Unit tests for ErrorHandlerUtil class.

Tests error handling patterns, logging integration, exception chaining,
and fallback mechanisms for standardized error handling throughout FuzzyShell.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.error_handler_util import ErrorHandlerUtil, ErrorContext


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock(spec=logging.Logger)


def test_log_and_raise_basic_functionality(mock_logger):
    """Test basic log_and_raise functionality."""
    with pytest.raises(RuntimeError, match="Test error message"):
        ErrorHandlerUtil.log_and_raise(
            message="Test error message",
            logger_instance=mock_logger
        )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "Test error message")


def test_log_and_raise_custom_exception_class(mock_logger):
    """Test log_and_raise with custom exception class."""
    with pytest.raises(ValueError, match="Custom error"):
        ErrorHandlerUtil.log_and_raise(
            message="Custom error",
            exception_class=ValueError,
            logger_instance=mock_logger
        )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "Custom error")


def test_log_and_raise_with_cause(mock_logger):
    """Test log_and_raise with exception chaining."""
    original_error = ValueError("Original error")
    
    with pytest.raises(RuntimeError, match="Chained error") as exc_info:
        ErrorHandlerUtil.log_and_raise(
            message="Chained error",
            logger_instance=mock_logger,
            cause=original_error
        )
    
    # Verify exception chaining
    assert exc_info.value.__cause__ is original_error
    mock_logger.log.assert_called_once_with(logging.ERROR, "Chained error")


def test_log_and_raise_custom_log_level(mock_logger):
    """Test log_and_raise with custom log level."""
    with pytest.raises(RuntimeError):
        ErrorHandlerUtil.log_and_raise(
            message="Warning level error",
            logger_instance=mock_logger,
            log_level=logging.WARNING
        )
    
    mock_logger.log.assert_called_once_with(logging.WARNING, "Warning level error")


def test_log_and_raise_default_logger():
    """Test log_and_raise uses default logger when none provided."""
    with patch('fuzzyshell.error_handler_util.logger') as mock_default_logger:
        with pytest.raises(RuntimeError):
            ErrorHandlerUtil.log_and_raise("Default logger test")
        
        mock_default_logger.log.assert_called_once_with(logging.ERROR, "Default logger test")


def test_log_and_raise_initialization_error(mock_logger):
    """Test log_and_raise_initialization_error convenience method."""
    with pytest.raises(RuntimeError, match="❌ TEST COMPONENT FAILED TO INITIALIZE"):
        ErrorHandlerUtil.log_and_raise_initialization_error(
            component_name="test component",
            logger_instance=mock_logger
        )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "❌ TEST COMPONENT FAILED TO INITIALIZE")


def test_log_and_raise_initialization_error_with_cause(mock_logger):
    """Test initialization error with exception chaining."""
    original_error = ImportError("Module not found")
    
    with pytest.raises(RuntimeError) as exc_info:
        ErrorHandlerUtil.log_and_raise_initialization_error(
            component_name="model",
            logger_instance=mock_logger,
            cause=original_error
        )
    
    assert exc_info.value.__cause__ is original_error


def test_log_and_raise_operation_error(mock_logger):
    """Test log_and_raise_operation_error convenience method."""
    with pytest.raises(RuntimeError, match="❌ SEARCH FAILED"):
        ErrorHandlerUtil.log_and_raise_operation_error(
            operation_name="search",
            logger_instance=mock_logger
        )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "❌ SEARCH FAILED")


def test_log_and_raise_operation_error_with_details(mock_logger):
    """Test operation error with details."""
    with pytest.raises(RuntimeError, match="❌ DATABASE CONNECTION FAILED: Connection timeout"):
        ErrorHandlerUtil.log_and_raise_operation_error(
            operation_name="database connection",
            details="Connection timeout",
            logger_instance=mock_logger
        )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "❌ DATABASE CONNECTION FAILED: Connection timeout")


def test_handle_with_fallback_success():
    """Test handle_with_fallback when operation succeeds."""
    def successful_operation():
        return "success result"
    
    result = ErrorHandlerUtil.handle_with_fallback(
        operation_callable=successful_operation,
        fallback_value="fallback"
    )
    
    assert result == "success result"


def test_handle_with_fallback_failure(mock_logger):
    """Test handle_with_fallback when operation fails."""
    def failing_operation():
        raise ValueError("Operation failed")
    
    result = ErrorHandlerUtil.handle_with_fallback(
        operation_callable=failing_operation,
        fallback_value="fallback result",
        error_message="Test operation",
        logger_instance=mock_logger
    )
    
    assert result == "fallback result"
    mock_logger.log.assert_called_once_with(logging.WARNING, "Test operation: Operation failed")


def test_handle_with_fallback_custom_log_level(mock_logger):
    """Test handle_with_fallback with custom log level."""
    def failing_operation():
        raise Exception("Test error")
    
    ErrorHandlerUtil.handle_with_fallback(
        operation_callable=failing_operation,
        fallback_value=None,
        logger_instance=mock_logger,
        log_level=logging.DEBUG
    )
    
    mock_logger.log.assert_called_once_with(logging.DEBUG, "Operation failed: Test error")


def test_log_and_continue(mock_logger):
    """Test log_and_continue functionality."""
    error = ValueError("Test error")
    
    # Should not raise an exception
    ErrorHandlerUtil.log_and_continue(
        error=error,
        context="Database operation",
        logger_instance=mock_logger
    )
    
    mock_logger.log.assert_called_once_with(logging.ERROR, "Database operation failed: Test error")


def test_log_and_continue_custom_log_level(mock_logger):
    """Test log_and_continue with custom log level."""
    error = RuntimeError("Test error")
    
    ErrorHandlerUtil.log_and_continue(
        error=error,
        context="Cache operation",
        logger_instance=mock_logger,
        log_level=logging.INFO
    )
    
    mock_logger.log.assert_called_once_with(logging.INFO, "Cache operation failed: Test error")


def test_create_error_context():
    """Test create_error_context factory method."""
    with patch('fuzzyshell.error_handler_util.logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        context = ErrorHandlerUtil.create_error_context("TestComponent")
        
        assert isinstance(context, ErrorContext)
        assert context.component_name == "TestComponent"
        assert context.logger is mock_logger
        mock_get_logger.assert_called_once_with("FuzzyShell.TestComponent")


def test_create_error_context_custom_logger_name():
    """Test create_error_context with custom logger name."""
    with patch('fuzzyshell.error_handler_util.logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        context = ErrorHandlerUtil.create_error_context(
            "TestComponent",
            logger_name="Custom.Logger"
        )
        
        mock_get_logger.assert_called_once_with("Custom.Logger")


class TestErrorContext:
    """Test the ErrorContext class."""
    
    @pytest.fixture
    def error_context(self, mock_logger):
        """Create ErrorContext for testing."""
        return ErrorContext("TestComponent", mock_logger)
    
    def test_error_context_initialization(self, mock_logger):
        """Test ErrorContext initialization."""
        context = ErrorContext("TestComponent", mock_logger)
        assert context.component_name == "TestComponent"
        assert context.logger is mock_logger
    
    def test_context_log_and_raise(self, error_context, mock_logger):
        """Test ErrorContext log_and_raise method."""
        with pytest.raises(ValueError, match="Context error"):
            error_context.log_and_raise(
                message="Context error",
                exception_class=ValueError
            )
        
        mock_logger.log.assert_called_once_with(logging.ERROR, "Context error")
    
    def test_context_log_and_raise_initialization_error(self, error_context, mock_logger):
        """Test ErrorContext initialization error method."""
        with pytest.raises(RuntimeError, match="❌ TESTCOMPONENT FAILED TO INITIALIZE"):
            error_context.log_and_raise_initialization_error()
        
        mock_logger.log.assert_called_once_with(logging.ERROR, "❌ TESTCOMPONENT FAILED TO INITIALIZE")
    
    def test_context_log_and_raise_operation_error(self, error_context, mock_logger):
        """Test ErrorContext operation error method."""
        with pytest.raises(RuntimeError, match="❌ SEARCH FAILED: No results"):
            error_context.log_and_raise_operation_error(
                operation_name="search",
                details="No results"
            )
        
        mock_logger.log.assert_called_once_with(logging.ERROR, "❌ SEARCH FAILED: No results")
    
    def test_context_handle_with_fallback(self, error_context, mock_logger):
        """Test ErrorContext handle_with_fallback method."""
        def failing_operation():
            raise Exception("Test error")
        
        result = error_context.handle_with_fallback(
            operation_callable=failing_operation,
            fallback_value="fallback",
            error_message="Context operation"
        )
        
        assert result == "fallback"
        mock_logger.log.assert_called_once_with(logging.WARNING, "Context operation: Test error")
    
    def test_context_log_and_continue(self, error_context, mock_logger):
        """Test ErrorContext log_and_continue method."""
        error = RuntimeError("Test error")
        
        error_context.log_and_continue(
            error=error,
            context="Context operation"
        )
        
        mock_logger.log.assert_called_once_with(logging.ERROR, "Context operation failed: Test error")


def test_error_patterns_integration():
    """Integration test showing how different error patterns work together."""
    # Create context for a component
    context = ErrorHandlerUtil.create_error_context("SearchEngine")
    
    # Test that all patterns work with the context
    with patch.object(context.logger, 'log') as mock_log:
        # Test fallback pattern
        def failing_op():
            raise ValueError("Network error")
        
        result = context.handle_with_fallback(
            operation_callable=failing_op,
            fallback_value="offline_mode",
            error_message="Network operation"
        )
        
        assert result == "offline_mode"
        mock_log.assert_called_with(logging.WARNING, "Network operation: Network error")


@pytest.mark.parametrize("component_name,expected_logger_name", [
    ("ModelHandler", "FuzzyShell.ModelHandler"),
    ("SearchEngine", "FuzzyShell.SearchEngine"), 
    ("DatabaseProvider", "FuzzyShell.DatabaseProvider"),
])
def test_create_error_context_various_components(component_name, expected_logger_name):
    """Test error context creation for various component names."""
    with patch('fuzzyshell.error_handler_util.logging.getLogger') as mock_get_logger:
        ErrorHandlerUtil.create_error_context(component_name)
        mock_get_logger.assert_called_once_with(expected_logger_name)


def test_error_message_formatting():
    """Test that error messages are formatted consistently."""
    # Test initialization error formatting
    with pytest.raises(RuntimeError) as exc_info:
        ErrorHandlerUtil.log_and_raise_initialization_error("model handler")
    
    assert "❌ MODEL HANDLER FAILED TO INITIALIZE" in str(exc_info.value)
    
    # Test operation error formatting
    with pytest.raises(RuntimeError) as exc_info:
        ErrorHandlerUtil.log_and_raise_operation_error("database query", "timeout occurred")
    
    assert "❌ DATABASE QUERY FAILED: timeout occurred" in str(exc_info.value)


def test_exception_chaining_preservation():
    """Test that exception chaining is properly preserved."""
    original = ValueError("Original issue")
    
    with pytest.raises(RuntimeError) as exc_info:
        ErrorHandlerUtil.log_and_raise(
            "Wrapper error",
            cause=original
        )
    
    # Verify the chain is preserved
    assert exc_info.value.__cause__ is original
    assert str(exc_info.value.__cause__) == "Original issue"