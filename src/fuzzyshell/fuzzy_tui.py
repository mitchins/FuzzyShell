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
    
    #status-bar {
        dock: bottom;
        height: 1;
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
        
        # Status bar
        status_bar = StatusBar()
        status_bar.id = "status-bar"
        yield status_bar
        
        # Footer with keybindings
        yield Footer()

    def on_mount(self) -> None:
        """When app is mounted, setup and focus."""
        self.query_one("#search-input").focus()
        
        # Update status bar with initial info
        if self.fuzzyshell:
            try:
                # Get database stats
                cursor = self.fuzzyshell.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM commands")
                count = cursor.fetchone()[0]
                
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.item_count = count
                
                # Estimate database size (rough)
                status_bar.database_size = f"{count * 50 // 1024}KB"  # Rough estimate
                
            except Exception:
                pass  # Ignore errors during startup
                
        # Update mode indicator
        mode_indicator = self.query_one("#mode-indicator", SearchModeIndicator)
        mode_indicator.search_mode = self.search_mode
        
    def action_cycle_search_mode(self) -> None:
        """Cycle through search modes using modern Python match/case."""
        match self.search_mode:
            case "hybrid":
                self.search_mode = "semantic"
            case "semantic":
                self.search_mode = "keyword" 
            case "keyword":
                self.search_mode = "hybrid"
            case _:
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
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.search_time = search_time
            
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
        
        # Clear search time
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.search_time = 0.0
        
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