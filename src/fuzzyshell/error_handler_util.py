"""
Error handling utilities for FuzzyShell.

Provides standardized error handling patterns to reduce boilerplate
code throughout the codebase. Centralizes logging, error formatting,
and exception raising patterns.
"""

import logging
from typing import Optional, Any, Type, Union

logger = logging.getLogger('FuzzyShell.ErrorHandler')


class ErrorHandlerUtil:
    """
    Utility class for standardized error handling patterns.
    
    Reduces boilerplate code by providing consistent methods for
    logging errors and raising exceptions with formatted messages.
    """
    
    @staticmethod
    def log_and_raise(
        message: str,
        exception_class: Type[Exception] = RuntimeError,
        logger_instance: Optional[logging.Logger] = None,
        cause: Optional[Exception] = None,
        log_level: int = logging.ERROR
    ) -> None:
        """
        Log an error message and raise an exception.
        
        Args:
            message: Error message to log and include in exception
            exception_class: Type of exception to raise (default: RuntimeError)
            logger_instance: Logger to use (default: module logger)
            cause: Original exception to chain from (using 'from cause')
            log_level: Logging level to use (default: ERROR)
            
        Raises:
            The specified exception_class with the provided message
        """
        log_instance = logger_instance or logger
        log_instance.log(log_level, message)
        
        if cause:
            raise exception_class(message) from cause
        else:
            raise exception_class(message)
    
    @staticmethod
    def log_and_raise_initialization_error(
        component_name: str,
        logger_instance: Optional[logging.Logger] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Standard pattern for component initialization failures.
        
        Args:
            component_name: Name of the component that failed to initialize
            logger_instance: Logger to use (default: module logger)
            cause: Original exception to chain from
        """
        message = f"❌ {component_name.upper()} FAILED TO INITIALIZE"
        ErrorHandlerUtil.log_and_raise(
            message=message,
            exception_class=RuntimeError,
            logger_instance=logger_instance,
            cause=cause
        )
    
    @staticmethod
    def log_and_raise_operation_error(
        operation_name: str,
        details: str = "",
        logger_instance: Optional[logging.Logger] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Standard pattern for operation failures.
        
        Args:
            operation_name: Name of the operation that failed
            details: Additional details about the failure
            logger_instance: Logger to use (default: module logger)
            cause: Original exception to chain from
        """
        message = f"❌ {operation_name.upper()} FAILED"
        if details:
            message += f": {details}"
        
        ErrorHandlerUtil.log_and_raise(
            message=message,
            exception_class=RuntimeError,
            logger_instance=logger_instance,
            cause=cause
        )
    
    @staticmethod
    def handle_with_fallback(
        operation_callable,
        fallback_value: Any = None,
        error_message: str = "Operation failed",
        logger_instance: Optional[logging.Logger] = None,
        log_level: int = logging.WARNING
    ) -> Any:
        """
        Execute an operation with automatic error handling and fallback.
        
        Args:
            operation_callable: Function/callable to execute
            fallback_value: Value to return if operation fails
            error_message: Message to log on failure
            logger_instance: Logger to use (default: module logger)
            log_level: Logging level for errors (default: WARNING)
            
        Returns:
            Result of operation_callable or fallback_value on failure
        """
        try:
            return operation_callable()
        except Exception as e:
            log_instance = logger_instance or logger
            log_instance.log(log_level, f"{error_message}: {str(e)}")
            return fallback_value
    
    @staticmethod
    def log_and_continue(
        error: Exception,
        context: str = "Operation",
        logger_instance: Optional[logging.Logger] = None,
        log_level: int = logging.ERROR
    ) -> None:
        """
        Log an error and continue execution (no exception raised).
        
        Args:
            error: Exception that occurred
            context: Context description for the error
            logger_instance: Logger to use (default: module logger)
            log_level: Logging level to use (default: ERROR)
        """
        log_instance = logger_instance or logger
        log_instance.log(log_level, f"{context} failed: {str(error)}")
    
    @staticmethod
    def create_error_context(
        component_name: str,
        logger_name: Optional[str] = None
    ) -> 'ErrorContext':
        """
        Create an error context for a specific component.
        
        Args:
            component_name: Name of the component
            logger_name: Logger name to use (default: FuzzyShell.{component_name})
            
        Returns:
            ErrorContext instance for the component
        """
        if logger_name is None:
            logger_name = f'FuzzyShell.{component_name}'
        
        component_logger = logging.getLogger(logger_name)
        return ErrorContext(component_name, component_logger)


class ErrorContext:
    """
    Context object for component-specific error handling.
    
    Provides convenience methods for error handling within a specific
    component context, with pre-configured logger and component name.
    """
    
    def __init__(self, component_name: str, logger_instance: logging.Logger):
        self.component_name = component_name
        self.logger = logger_instance
    
    def log_and_raise(
        self,
        message: str,
        exception_class: Type[Exception] = RuntimeError,
        cause: Optional[Exception] = None
    ) -> None:
        """Log error and raise exception with component context."""
        ErrorHandlerUtil.log_and_raise(
            message=message,
            exception_class=exception_class,
            logger_instance=self.logger,
            cause=cause
        )
    
    def log_and_raise_initialization_error(self, cause: Optional[Exception] = None) -> None:
        """Log initialization error for this component."""
        ErrorHandlerUtil.log_and_raise_initialization_error(
            component_name=self.component_name,
            logger_instance=self.logger,
            cause=cause
        )
    
    def log_and_raise_operation_error(
        self,
        operation_name: str,
        details: str = "",
        cause: Optional[Exception] = None
    ) -> None:
        """Log operation error for this component."""
        ErrorHandlerUtil.log_and_raise_operation_error(
            operation_name=operation_name,
            details=details,
            logger_instance=self.logger,
            cause=cause
        )
    
    def handle_with_fallback(
        self,
        operation_callable,
        fallback_value: Any = None,
        error_message: str = "Operation failed"
    ) -> Any:
        """Execute operation with fallback using component logger."""
        return ErrorHandlerUtil.handle_with_fallback(
            operation_callable=operation_callable,
            fallback_value=fallback_value,
            error_message=error_message,
            logger_instance=self.logger
        )
    
    def log_and_continue(
        self,
        error: Exception,
        context: str = "Operation"
    ) -> None:
        """Log error and continue using component logger."""
        ErrorHandlerUtil.log_and_continue(
            error=error,
            context=context,
            logger_instance=self.logger
        )