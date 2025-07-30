import time
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, Footer, Static, ProgressBar
from textual.reactive import reactive
from textual.binding import Binding
from textual import events
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.panel import Panel
import asyncio
import threading
import logging

logger = logging.getLogger('FuzzyShell.TUI')
# Import version to avoid circular import
try:
    from . import __version__
except ImportError:
    __version__ = "0.1.0"

class SearchModeIndicator(Static):
    """Widget to show current search mode with icon."""
    
    search_mode = reactive("hybrid")
    
    def __init__(self):
        super().__init__()
    
    def render(self):
        """Render the search mode indicator."""
        mode_icons = {
            "semantic": "🧠",
            "keyword": "🔍", 
            "hybrid": "⚡"
        }
        mode_colors = {
            "semantic": "blue",
            "keyword": "green",
            "hybrid": "magenta"
        }
        
        icon = mode_icons.get(self.search_mode, "⚡")
        color = mode_colors.get(self.search_mode, "magenta")
        
        text = Text()
        text.append(f"{icon} ", style=color)
        text.append(self.search_mode.upper(), style=f"bold {color}")
        
        return text

class StatusBar(Static):
    """Bottom status bar showing app version, item count, and stats."""
    
    version = reactive(__version__)
    item_count = reactive(0)
    database_size = reactive("0B")
    search_time = reactive(0.0)
    
    def render(self):
        """Render the status bar."""
        left_text = Text()
        left_text.append(f"FuzzyShell v{self.version}", style="bold cyan")
        
        right_text = Text()
        if self.item_count > 0:
            right_text.append(f"{self.item_count:,} items ", style="dim")
            right_text.append(f"({self.database_size}) ", style="dim")
        
        if self.search_time > 0:
            right_text.append(f"• {self.search_time*1000:.0f}ms", style="dim green")
        
        # Create full width text with left and right alignment
        full_text = Text()
        full_text.append(left_text)
        
        # Calculate padding to right-align
        terminal_width = self.size.width if self.size else 80
        used_width = len(left_text) + len(right_text)
        padding = max(0, terminal_width - used_width)
        
        full_text.append(" " * padding)
        full_text.append(right_text)
        
        return full_text

class StatusFooter(Static):
    """Combined footer showing status info and keybindings."""
    
    version = reactive(__version__)
    item_count = reactive(0)
    embedding_model = reactive("unknown")
    search_time = reactive(0.0)
    
    def render(self):
        """Render the combined status footer."""
        # Left side: Version and stats
        left_text = Text()
        left_text.append(f"FuzzyShell v{self.version}", style="bold cyan")
        
        if self.item_count > 0:
            left_text.append(f" • {self.item_count:,} items", style="dim")
            if self.embedding_model != "unknown":
                # Shorten model name for display
                model_short = self.embedding_model.replace("all-MiniLM-L6-v2", "MiniLM-L6")
                left_text.append(f" ({model_short})", style="dim blue")
        
        if self.search_time > 0:
            left_text.append(f" • {self.search_time*1000:.0f}ms", style="dim green")
        
        # Right side: Key bindings (simplified)
        right_text = Text()
        right_text.append("ESC ", style="bold")
        right_text.append("quit ", style="dim")
        right_text.append("↑↓ ", style="bold") 
        right_text.append("navigate ", style="dim")
        right_text.append("ENTER ", style="bold")
        right_text.append("select ", style="dim")
        right_text.append("TAB ", style="bold")
        right_text.append("mode", style="dim")
        
        # Create full width text with padding
        full_text = Text()
        full_text.append(left_text)
        
        # Calculate padding to right-align
        terminal_width = self.size.width if self.size else 80
        used_width = len(left_text) + len(right_text)
        padding = max(0, terminal_width - used_width)
        
        full_text.append(" " * padding)
        full_text.append(right_text)
        
        return full_text

class LoadingIndicator(Static):
    """Shows loading state with spinner."""
    
    is_loading = reactive(False)
    message = reactive("Loading...")
    
    def render(self):
        """Render loading indicator."""
        if not self.is_loading:
            return Text("")
        
        # Simple spinner animation
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spinner_index = int(time.time() * 8) % len(spinner_chars)
        spinner = spinner_chars[spinner_index]
        
        text = Text()
        text.append(f"{spinner} ", style="cyan")
        text.append(self.message, style="dim")
        
        return text

class CommandDescriptionPane(Static):
    """Widget to display command descriptions using T5-small model."""
    
    description = reactive("")
    is_loading = reactive(False)
    
    def __init__(self):
        super().__init__()
        self._description_handler = None
        self._current_command = None
        self._init_lock = threading.Lock()
        self.border_title = "Description"
        
    def render(self):
        """Render the description pane."""
        if self.is_loading:
            text = Text()
            spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spinner_index = int(time.time() * 10) % len(spinner_chars)
            spinner = spinner_chars[spinner_index]
            text.append(f"{spinner} ", style="cyan bold")
            text.append("Generating description...", style="dim italic")
            return text
        elif self.description:
            text = Text()
            text.append("📝 ", style="blue")
            text.append(self.description, style="white")
            return text
        else:
            text = Text()
            text.append("💡 ", style="dim")
            text.append("Select a command to see its description", style="dim")
            return text
    
    def _init_description_handler(self):
        """Initialize the description handler lazily with thread safety."""
        with self._init_lock:
            if self._description_handler is None:
                try:
                    from .model_handler import DescriptionHandler
                    self._description_handler = DescriptionHandler()
                except Exception as e:
                    logger.error("Failed to initialize DescriptionHandler: %s", str(e))
                    self._description_handler = False  # Mark as failed
        return self._description_handler if self._description_handler is not False else None
    
    def generate_description_async(self, command: str):
        """Generate description for a command asynchronously."""
        if not command or command == self._current_command:
            return
            
        self._current_command = command
        self.is_loading = True
        self.description = ""
        
        def generate_in_thread():
            try:
                handler = self._init_description_handler()
                if handler is None:
                    # Fallback description if handler failed to initialize
                    desc = f"Command: {command}"
                else:
                    desc = handler.generate_description(command)
                
                # Update UI on main thread
                self.app.call_from_thread(self._update_description, desc)
            except Exception as e:
                logger.error("Error in description generation thread: %s", str(e))
                # Fallback on error
                self.app.call_from_thread(self._update_description, f"Command: {command}")
        
        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()
    
    def _update_description(self, desc: str):
        """Update description on main thread."""
        self.is_loading = False
        self.description = desc

class SearchResult(Static):
    """A widget to display a single search result with enhanced styling."""
    
    is_selected = reactive(False)
    
    def __init__(self, command: str, score: float = 0.0, search_mode: str = "hybrid"):
        super().__init__()
        self.command = command
        self.score = score
        self.search_mode = search_mode
        
    def render(self):
        """Render the search result with styling."""
        text = Text()
        
        # Add selection indicator
        if self.is_selected:
            text.append("▶ ", style="bold magenta")
        else:
            text.append("  ", style="dim")
        
        # Add command text
        text.append(self.command, style="white")
        
        return text

class FuzzyShellApp(App):
    """The main Fuzzy Shell application with modern UI."""
    
    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("up,k", "cursor_up", "Previous", show=True),
        Binding("down,j", "cursor_down", "Next", show=True),
        Binding("enter", "select_command", "Select", show=True),
        Binding("tab", "cycle_search_mode", "Mode", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
    ]
    
    CSS = """
    #search-input {
        dock: top;
        margin: 1;
    }
    
    #mode-indicator {
        dock: top;
        height: 1;
        text-align: center;
    }
    
    #loading {
        dock: top;
        height: 1;
        text-align: center;
    }
    
    #results-container {
        height: 1fr;
        overflow-y: auto;
        margin: 1;
    }
    
    #description-pane {
        dock: bottom;
        height: 3;
        margin: 1;
        border: solid $accent;
    }
    
    #status-footer {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text;
    }
    
    SearchResult {
        height: 1;
        padding: 0 1;
    }
    
    SearchResult.selected {
        background: $accent;
        color: $text;
    }
    """
    
    def __init__(self, search_callback, fuzzyshell_instance=None):
        super().__init__()
        self.search_callback = search_callback
        self.fuzzyshell = fuzzyshell_instance
        self.selected_index = 0
        self._last_search_time = 0
        self._search_delay = 0.15  # 150ms debounce delay
        self.search_mode = "hybrid"
        self.current_results = []
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""        
        # Search input
        yield Input(
            placeholder="Type to search commands... (Tab to cycle modes)",
            id="search-input"
        )
        
        # Mode indicator
        mode_indicator = SearchModeIndicator()
        mode_indicator.id = "mode-indicator"
        yield mode_indicator
        
        # Loading indicator
        loading = LoadingIndicator()
        loading.id = "loading"
        yield loading
        
        # Results container
        yield Container(id="results-container")
        
        # Description pane
        description_pane = CommandDescriptionPane()
        description_pane.id = "description-pane"
        yield description_pane
        
        # Combined footer with status and keybindings
        footer = StatusFooter()
        footer.id = "status-footer"
        yield footer

    def on_mount(self) -> None:
        """When app is mounted, setup and focus."""
        self.query_one("#search-input").focus()
        
        # Update status bar with initial info
        if self.fuzzyshell:
            try:
                # Get comprehensive database info
                db_info = self.fuzzyshell.get_database_info()
                
                footer = self.query_one("#status-footer", StatusFooter)
                footer.item_count = db_info['item_count']
                footer.embedding_model = db_info['embedding_model']
                
            except Exception as e:
                logger.debug("Error updating status bar: %s", str(e))
                
        # Update mode indicator
        mode_indicator = self.query_one("#mode-indicator", SearchModeIndicator)
        mode_indicator.search_mode = self.search_mode
        
    def action_cycle_search_mode(self) -> None:
        """Cycle through search modes using if/elif for Python 3.9 compatibility."""
        if self.search_mode == "hybrid":
            self.search_mode = "semantic"
        elif self.search_mode == "semantic":
            self.search_mode = "keyword" 
        elif self.search_mode == "keyword":
            self.search_mode = "hybrid"
        else:
            self.search_mode = "hybrid"  # fallback
        
        # Update mode indicator
        mode_indicator = self.query_one("#mode-indicator", SearchModeIndicator)
        mode_indicator.search_mode = self.search_mode
        
        # Re-run search if there's a query
        search_input = self.query_one("#search-input", Input)
        if search_input.value.strip():
            self._perform_search(search_input.value.strip())
    
    def action_refresh(self) -> None:
        """Refresh the search results."""
        search_input = self.query_one("#search-input", Input)
        if search_input.value.strip():
            self._perform_search(search_input.value.strip())
    
    def _show_loading(self, message: str = "Searching...") -> None:
        """Show loading indicator."""
        loading = self.query_one("#loading", LoadingIndicator)
        loading.is_loading = True
        loading.message = message
        
    def _hide_loading(self) -> None:
        """Hide loading indicator."""
        loading = self.query_one("#loading", LoadingIndicator)
        loading.is_loading = False
        
    def _perform_search(self, query: str) -> None:
        """Perform search with loading indicator."""
        if not query.strip():
            self._clear_results()
            return
            
        self._show_loading("Searching...")
        
        start_time = time.time()
        
        try:
            # Perform the search synchronously to avoid SQLite thread issues
            results = self.search_callback(query)
            
            search_time = time.time() - start_time
            
            # Update status bar with search time
            footer = self.query_one("#status-footer", StatusFooter)
            footer.search_time = search_time
            
            self.current_results = results
            self._display_results(results)
            
        except Exception as e:
            # Show error
            self._display_error(f"Search error: {str(e)}")
        finally:
            self._hide_loading()
    
    def _display_results(self, results) -> None:
        """Display search results."""
        results_container = self.query_one("#results-container")
        results_container.remove_children()
        
        if not results:
            results_container.mount(
                Static("No matches found", classes="dim", id="no-results")
            )
            return
        
        for i, (cmd, score) in enumerate(results):
            result_widget = SearchResult(cmd, score, self.search_mode)
            if i == 0:  # Select first result
                result_widget.is_selected = True
                result_widget.add_class("selected")
                # Generate description for first result
                description_pane = self.query_one("#description-pane", CommandDescriptionPane)
                description_pane.generate_description_async(cmd)
            results_container.mount(result_widget)
            
        self.selected_index = 0
    
    def _display_error(self, error_message: str) -> None:
        """Display error message."""
        results_container = self.query_one("#results-container")
        results_container.remove_children()
        results_container.mount(
            Static(error_message, classes="error")
        )
    
    def _clear_results(self) -> None:
        """Clear all results."""
        results_container = self.query_one("#results-container")
        results_container.remove_children()
        self.current_results = []
        self.selected_index = 0
        
        # Clear description
        description_pane = self.query_one("#description-pane", CommandDescriptionPane)
        description_pane.description = ""
        description_pane.is_loading = False
        description_pane._current_command = None
        
        # Clear search time
        footer = self.query_one("#status-footer", StatusFooter)
        footer.search_time = 0.0
        
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes with debouncing."""
        query = event.value.strip()
        current_time = time.time()
        
        # Clear results immediately if query is empty
        if not query:
            self._clear_results()
            return
            
        # Debounce the search
        if current_time - self._last_search_time < self._search_delay:
            return
            
        self._last_search_time = current_time
        self._perform_search(query)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key pressed in input field - select current command."""
        self.action_select_command()
    
    def _select_result(self, index: int) -> None:
        """Select a result by index."""
        results = self.query_one("#results-container").children
        if not results or not isinstance(results[0], SearchResult):
            return
            
        # Ensure index is within bounds
        index = max(0, min(index, len(results) - 1))
        
        # Update selection
        for i, result in enumerate(results):
            if isinstance(result, SearchResult):
                result.is_selected = (i == index)
                if i == index:
                    result.add_class("selected")
                    # Trigger description generation for selected command
                    description_pane = self.query_one("#description-pane", CommandDescriptionPane)
                    description_pane.generate_description_async(result.command)
                else:
                    result.remove_class("selected")
                    
        self.selected_index = index
        
    def action_cursor_up(self) -> None:
        """Move selection up."""
        self._select_result(self.selected_index - 1)
        
    def action_cursor_down(self) -> None:
        """Move selection down."""
        self._select_result(self.selected_index + 1)
        
    def action_select_command(self) -> None:
        """Select the current command and exit the app."""
        if not self.current_results:
            # No results available, just exit without selection
            self.exit(None)
            return
            
        if 0 <= self.selected_index < len(self.current_results):
            selected_command = self.current_results[self.selected_index][0]
            # Exit the app and return the selected command
            self.exit(selected_command)
        else:
            # Invalid selection, exit without command
            self.exit(None)
            
    def on_key(self, event: events.Key) -> None:
        """Handle key events."""
        if event.key in ("escape", "ctrl+c", "ctrl+d"):
            self.exit()