import time
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.scroll_view import ScrollView
from textual.widgets import Input, Footer, Static, ProgressBar
from textual.reactive import reactive
from textual.binding import Binding
from textual import events
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.panel import Panel
from textual.geometry import Size
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
    show_scores = reactive(False)
    
    def render(self):
        """Render the combined status footer."""
        # Left side: Version and stats
        left_text = Text()
        left_text.append(f"FuzzyShell v{self.version}", style="bold cyan")
        
        if self.item_count > 0:
            left_text.append(f" • {self.item_count:,} items", style="dim")
            if self.embedding_model != "unknown":
                # Shorten model name for display
                model_short = self.embedding_model.replace("minilm-l6-v2-terminal-describer", "Terminal-MiniLM")
                model_short = model_short.replace("all-MiniLM-L6-v2", "MiniLM-L6")  # Legacy compatibility
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
        right_text.append("mode ", style="dim")
        right_text.append("^S ", style="bold")
        if self.show_scores:
            right_text.append("scores(ON)", style="dim bright_green")
        else:
            right_text.append("scores(OFF)", style="dim")
        
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
    
    def __init__(self, command: str, score: float = 0.0, search_mode: str = "hybrid", 
                 semantic_score: float = 0.0, bm25_score: float = 0.0):
        super().__init__()
        self.command = command
        self.score = score
        self.search_mode = search_mode
        self.semantic_score = semantic_score
        self.bm25_score = bm25_score
        
    def render(self):
        """Render the search result with styling and optional score."""
        text = Text()
        
        # Add selection indicator
        if self.is_selected:
            text.append("▶ ", style="bold magenta")
        else:
            text.append("  ", style="dim")
        
        # Add command text
        text.append(self.command, style="white")
        
        # Add match score and type if show_scores is enabled (get from parent app)
        app = self.app
        if hasattr(app, 'show_scores') and app.show_scores and self.score > 0:
            # Format score as percentage
            score_pct = int(self.score * 100)
            
            # Determine match type based on which score is higher
            semantic_pct = int(self.semantic_score * 100)
            bm25_pct = int(self.bm25_score * 100)
            
            # Determine primary match type
            if bm25_pct > semantic_pct * 1.5:  # BM25 significantly higher
                match_type = "keyword"
                match_icon = "🔍"
                type_color = "green"
            elif semantic_pct > bm25_pct * 1.5:  # Semantic significantly higher
                match_type = "semantic"
                match_icon = "🧠"
                type_color = "blue"
            else:  # Both contribute significantly
                match_type = "hybrid"
                match_icon = "⚡"
                type_color = "magenta"
            
            # Color code the score based on relevance
            if score_pct >= 80:
                score_color = "bright_green"
            elif score_pct >= 60:
                score_color = "yellow"
            elif score_pct >= 40:
                score_color = "orange1"
            else:
                score_color = "red"
            
            # Add score and match type with appropriate styling
            text.append(f" ({score_pct}% ", style=f"dim {score_color}")
            text.append(f"{match_icon}{match_type}", style=f"dim {type_color}")
            text.append(")", style=f"dim {score_color}")
        
        return text

class FuzzyShellApp(App):
    """The main Fuzzy Shell application with modern UI."""
    
    # Reactive properties
    show_scores = reactive(False)
    
    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("up,k", "cursor_up", "Previous", show=True),
        Binding("down,j", "cursor_down", "Next", show=True),
        Binding("pageup,ctrl+u", "page_up", "Page Up", show=False),
        Binding("pagedown,ctrl+d", "page_down", "Page Down", show=False),
        Binding("home,ctrl+a", "go_to_top", "Top", show=False),
        Binding("end,ctrl+e", "go_to_bottom", "Bottom", show=False),
        Binding("enter", "select_command", "Select", show=True),
        Binding("tab", "cycle_search_mode", "Mode", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
        Binding("ctrl+s", "toggle_scores", "Scores", show=True),
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
        margin: 1;
    }
    
    #results-inner {
        width: 100%;
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
        self._search_delay = 0.05  # 50ms debounce delay (reduced for responsiveness)
        self._pending_search = None  # Track pending searches
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
        
        # Results container with scrolling (no inner container)
        yield ScrollView(id="results-container")
        
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
    
    def action_toggle_scores(self) -> None:
        """Toggle display of match scores/percentages."""
        self.show_scores = not self.show_scores
        
        # Update status footer to reflect the change
        footer = self.query_one("#status-footer", StatusFooter)
        footer.show_scores = self.show_scores
        
        # Refresh results display to show/hide scores
        results_container = self.query_one("#results-container", ScrollView)
        results_container.remove_children()
        
        # Re-display current results with updated score visibility
        if self.current_results:
            self._display_results(self.current_results)
    
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
        results_container = self.query_one("#results-container", ScrollView)
        results_container.remove_children()
        
        if not results:
            results_container.mount(
                Static("No matches found", classes="dim", id="no-results")
            )
            return
        
        for i, result_tuple in enumerate(results):
            # Handle both 2-tuple (legacy) and 4-tuple (detailed scores) formats
            if len(result_tuple) == 4:
                cmd, score, semantic_score, bm25_score = result_tuple
                result_widget = SearchResult(cmd, score, self.search_mode, semantic_score, bm25_score)
            else:
                # Legacy format compatibility
                cmd, score = result_tuple
                result_widget = SearchResult(cmd, score, self.search_mode)
                
            if i == 0:  # Select first result
                result_widget.is_selected = True
                result_widget.add_class("selected")
                # Generate description for first result
                description_pane = self.query_one("#description-pane", CommandDescriptionPane)
                description_pane.generate_description_async(cmd)
            results_container.mount(result_widget)

        # Update ScrollView virtual size based on results
        scroll_view = self.query_one("#results-container", ScrollView)
        total_lines = len(results)  # 1 line per SearchResult widget
        visible_height = scroll_view.size.height
        # Add extra lines equal to viewport height minus one to eliminate bottom dead zone
        vs_height = total_lines + visible_height - 1
        scroll_view.virtual_size = Size(scroll_view.size.width, vs_height)
        scroll_view._scroll_update(scroll_view.virtual_size)

        # Adjust virtual size post-layout to account for actual content region
        def adjust_virtual_size():
            content_height = scroll_view.scrollable_content_region.height
            visible_height = scroll_view.size.height
            vs = Size(scroll_view.size.width, content_height + visible_height - 1)
            scroll_view.virtual_size = vs
            scroll_view._scroll_update(vs)
        self.call_after_refresh(adjust_virtual_size)
            
        self.selected_index = 0
    
    def _display_error(self, error_message: str) -> None:
        """Display error message."""
        results_container = self.query_one("#results-container", ScrollView)
        results_container.remove_children()
        results_container.mount(
            Static(error_message, classes="error")
        )
    
    def _clear_results(self) -> None:
        """Clear all results."""
        results_container = self.query_one("#results-container", ScrollView)
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
        """Handle input changes with improved debouncing."""
        query = event.value.strip()
        
        # Clear results immediately if query is empty
        if not query:
            self._clear_results()
            if self._pending_search:
                self._pending_search.stop()
                self._pending_search = None
            return
        
        # Stop any pending search
        if self._pending_search:
            self._pending_search.stop()
        
        # Schedule new search with debouncing
        self._pending_search = self.set_timer(
            self._search_delay, 
            lambda: self._perform_search_scheduled(query)
        )
    
    def _perform_search_scheduled(self, query: str) -> None:
        """Perform scheduled search and clear pending search."""
        self._pending_search = None
        self._perform_search(query)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key pressed in input field - select current command."""
        self.action_select_command()
    
    def _select_result(self, index: int) -> None:
        """Select a result by index with automatic scrolling."""
        scroll_view = self.query_one("#results-container", ScrollView)
        results = scroll_view.children
        if not results or not isinstance(results[0], SearchResult):
            return
            
        # Ensure index is within bounds
        index = max(0, min(index, len(results) - 1))
        
        # Update selection
        selected_widget = None
        for i, result in enumerate(results):
            if isinstance(result, SearchResult):
                result.is_selected = (i == index)
                if i == index:
                    result.add_class("selected")
                    selected_widget = result
                    # Trigger description generation for selected command
                    description_pane = self.query_one("#description-pane", CommandDescriptionPane)
                    description_pane.generate_description_async(result.command)
                else:
                    result.remove_class("selected")
        
        # Auto-scroll to keep selected item visible
        if selected_widget:
            # Use call_after_refresh to ensure widget is properly laid out
            self.call_after_refresh(lambda: self._scroll_to_selection(scroll_view, selected_widget, index))
                    
        self.selected_index = index
    
    def _scroll_to_selection(self, scroll_view: ScrollView, selected_widget: Static, index: int) -> None:
        """Scrolls the view to ensure the selected widget is visible using scroll_to_region."""
        from textual.geometry import Region

        # Get absolute widget region
        widget_region = selected_widget.region

        # Get scrollable content region (position of content start)
        content_region = scroll_view.scrollable_content_region

        # Compute widget region relative to content origin
        rel_x = widget_region.x - content_region.x
        rel_y = widget_region.y - content_region.y

        # Build a Region for the widget
        region = Region(rel_x, rel_y, widget_region.width, widget_region.height)

        # Scroll so that the widget region is visible (explicitly pass top=False)
        scroll_view.scroll_to_region(
            region,
            force=True,
            animate=False,
            immediate=True,
            top=False,
        )
        
    def action_cursor_up(self) -> None:
        """Move selection up."""
        self._select_result(self.selected_index - 1)
        
    def action_cursor_down(self) -> None:
        """Move selection down."""
        self._select_result(self.selected_index + 1)
    
    def action_page_up(self) -> None:
        """Move selection up by one page."""
        # Move up by ~10 items (approximate page size)
        self._select_result(self.selected_index - 10)
    
    def action_page_down(self) -> None:
        """Move selection down by one page."""
        # Move down by ~10 items (approximate page size)
        self._select_result(self.selected_index + 10)
    
    def action_go_to_top(self) -> None:
        """Go to the first result."""
        self._select_result(0)
    
    def action_go_to_bottom(self) -> None:
        """Go to the last result."""
        if self.current_results:
            self._select_result(len(self.current_results) - 1)
        
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