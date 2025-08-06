import time
import urwid
import threading
import logging
import asyncio
import random

from .tui.widgets import (
    HelpDialog, WatermarkEdit, SearchModeIndicator, LoadingIndicator,
    StatusFooter, CommandDescriptionPane, SearchResult
)
from .tui.screens.expert_screen import ExpertScreen, EXPERT_SCREEN_PALETTE
from .tui.key_bindings import *

logger = logging.getLogger('FuzzyShell.TUI')

# Import version at module level to avoid circular imports
try:
    from fuzzyshell import __version__
except ImportError:
    # Fallback for development/circular import issues
    __version__ = "1.0.0"


class WatermarkEdit(urwid.Edit):
    """Custom Edit widget with watermark text when empty."""
    
    def __init__(self, caption="", edit_text="", watermark_text="", **kwargs):
        super().__init__(caption, edit_text, **kwargs)
        self.watermark_text = watermark_text
        self._has_been_edited = False
    
    def render(self, size, focus=False):
        """Override render to show watermark when empty."""
        try:
            if not self.edit_text and not self._has_been_edited and self.watermark_text:
                # Show watermark until user starts typing
                full_watermark = f"{self.caption}{self.watermark_text}"
                watermark_widget = urwid.Text([('placeholder_text', full_watermark)], align='left')
                return watermark_widget.render(size, focus)
            else:
                # Render normal edit widget
                return super().render(size, focus)
        except Exception as e:
            logger.debug(f"Error in WatermarkEdit.render: {e}")
            # Fallback to basic text widget
            fallback = urwid.Text(f"Search: {self.edit_text}")
            return fallback.render(size, focus)
    
    def keypress(self, size, key):
        """Track when user starts editing to hide watermark and handle global commands."""
        
        # Handle global commands that should work even when input is focused
        if key in (KEY_TOGGLE_SCORES, '\x13', 'ctrl S', '\x1f'):  # Multiple representations of Ctrl+S
            # Let the parent handle this global command
            logger.debug(f"WatermarkEdit: Passing through ctrl s: '{key}'")
            return key
        elif key == KEY_REFRESH_SEARCH:
            # Let the parent handle this global command  
            return key
        elif key in (KEY_QUIT, 'escape'):
            # Let the parent handle escape
            return key
        elif key in NAVIGATION_KEYS:
            # Let urwid handle navigation naturally - just pass through to unhandled_input
            logger.info(f"🔑 WatermarkEdit: Passing navigation key '{key}' to parent")
            return key
        
        # Mark as edited when user starts typing
        if key and len(key) == 1 and key.isprintable():
            self._has_been_edited = True
        elif key == 'backspace' and self.edit_text:
            self._has_been_edited = True
        
        return super().keypress(size, key)

try:
    from . import __version__
except ImportError:
    __version__ = "1.0.0"

# SearchModeIndicator is now imported from tui.widgets

# StatusFooter is now imported from tui.widgets

# CommandDescriptionPane is now imported from tui.widgets

# SearchResult is now imported from tui.widgets

# LoadingIndicator is now imported from tui.widgets

class SearchManager:
    """Handles search operations, progress tracking, and cancellation."""
    
    def __init__(self, fuzzyshell_instance, search_callback=None):
        self.fuzzyshell = fuzzyshell_instance
        self.search_callback = search_callback
        self.current_query = None
        self.search_cancelled = False
        
    def execute_search(self, query, ui_controller):
        """Execute a search with UI updates and progress tracking."""
        if not query.strip():
            return []
        
        # Validation
        if self.search_cancelled or query != self.current_query:
            return []
        
        start_time = time.time()
        self._update_ui_start(query, ui_controller)
        
        try:
            # Create progress callback
            progress_callback = self._create_progress_callback(query, start_time, ui_controller)
            
            # Execute search
            results = self._perform_search_operation(query, progress_callback)
            
            # Process results
            return self._process_search_results(query, results, start_time, ui_controller)
            
        except Exception as e:
            return self._handle_search_error(e, start_time, ui_controller)
        finally:
            self._cleanup_ui(ui_controller)
    
    def cancel_search(self):
        """Cancel the current search operation."""
        self.search_cancelled = True
        
    def set_current_query(self, query):
        """Set the current query and reset cancellation.""" 
        self.current_query = query
        self.search_cancelled = False
    
    def _update_ui_start(self, query, ui_controller):
        """Update UI elements at search start."""
        if not ui_controller.has_loop():
            return
            
        # Update loading indicator
        if hasattr(self.fuzzyshell, '_model') and self.fuzzyshell._model is not None:
            message = f"🔍 Initializing semantic search for '{query}'..."
        else:
            message = f"⚠️ Model not ready for '{query}'..."
        
        ui_controller.start_loading(message)
        ui_controller.set_search_status("Searching...")
    
    def _create_progress_callback(self, query, start_time, ui_controller):
        """Create a progress callback function."""
        def progress_callback(current, total, stage, partial_results):
            if self.search_cancelled or query != self.current_query:
                return
                
            if not ui_controller.has_loop():
                return
                
            # Calculate progress
            percent = int((current / total) * 100) if total > 0 else 0
            elapsed = time.time() - start_time
            
            # Update status
            status_text = f"Searching ({elapsed:.0f}s)" if elapsed > 0.5 else "Searching..."
            ui_controller.set_search_status(status_text)
            
            # Update progress message
            progress_msg = f"🔍 {stage} ({current:,}/{total:,} records, {percent}%)"
            ui_controller.update_loading(progress_msg)
            
            # Show partial results if complete
            if (partial_results and len(partial_results) > 0 and 
                current >= total and stage == "Complete"):
                if not self.search_cancelled and query == self.current_query:
                    ui_controller.display_results(partial_results)
                    logger.debug(f"Progress callback updated results for '{query}' ({len(partial_results)} results)")
            
            ui_controller.redraw()
        
        return progress_callback
    
    def _perform_search_operation(self, query, progress_callback):
        """Execute the actual search operation."""
        if hasattr(self.fuzzyshell, 'search'):
            return self.fuzzyshell.search(query, top_k=100, return_scores=True, progress_callback=progress_callback)
        else:
            return self.search_callback(query) if self.search_callback else []
    
    def _process_search_results(self, query, results, start_time, ui_controller):
        """Process and validate search results."""
        search_time = time.time() - start_time
        
        # Validate results
        if results is None:
            logger.warning(f"Search callback returned None for query: '{query}'")
            results = []
        elif not isinstance(results, (list, tuple)):
            logger.error(f"Search callback returned invalid type: {type(results)}, expected list/tuple")
            results = []
        elif len(results) == 0:
            logger.debug(f"Search callback returned empty results for query: '{query}'")
        
        logger.debug(f"Search for '{query}' returned {len(results)} results in {search_time:.3f}s")
        
        # Update UI if search wasn't cancelled
        if not self.search_cancelled and query == self.current_query:
            ui_controller.update_footer(search_time=search_time)
            ui_controller.display_results(results)
            
            # Update status
            if len(results) > 0:
                ui_controller.set_search_status(f"Found {len(results)}")
            else:
                ui_controller.set_search_status("No results")
        else:
            logger.debug(f"Search for '{query}' was cancelled, not updating results")
            
        return results
    
    def _handle_search_error(self, error, start_time, ui_controller):
        """Handle search errors and update UI appropriately."""
        search_time = time.time() - start_time
        logger.error(f"Search error after {search_time:.3f}s: {error}", exc_info=True)
        
        ui_controller.update_footer(search_time=search_time)
        
        # Create appropriate error message
        error_msg = f"Search error: {str(error)}"
        if "timeout" in str(error).lower() or search_time > 25.0:
            error_msg = f"Model initialization timed out ({search_time:.1f}s). Try again in a moment."
        elif search_time > 10.0:
            error_msg = f"Search took {search_time:.1f}s - model may still be loading"
        
        ui_controller.show_error(error_msg)
        return []
    
    def _cleanup_ui(self, ui_controller):
        """Clean up UI elements after search completion."""
        if ui_controller.has_loop():
            ui_controller.stop_loading()
            ui_controller.redraw()


class ResultRenderer:
    """Handles result processing, validation, and display."""
    
    def __init__(self):
        pass
    
    def render_results(self, results, ui_controller):
        """Process and render search results."""
        ui_controller.clear_results()
        
        if not results:
            ui_controller.show_no_results()
            return
        
        # Process and validate results
        valid_results = self._process_results(results)
        
        if not valid_results:
            ui_controller.show_no_valid_results()
            return
        
        # Create and display result widgets
        self._create_result_widgets(valid_results, ui_controller)
        
        # Set initial focus and description
        self._set_initial_focus(ui_controller)
    
    def _process_results(self, results):
        """Process and validate raw search results."""
        valid_results = []
        logger.debug(f"Processing {len(results)} raw results")
        
        for i, r in enumerate(results):
            try:
                logger.debug(f"Processing result {i}: {type(r)} = {r}")
                
                # Skip None or empty results
                if r is None:
                    logger.warning(f"Skipping None result at index {i}")
                    continue
                    
                if not r:
                    logger.warning(f"Skipping empty result at index {i}: {r}")
                    continue
                    
                # Ensure r is a sequence
                if not hasattr(r, '__len__') or not hasattr(r, '__getitem__'):
                    logger.warning(f"Skipping non-sequence result at index {i}: {type(r)} = {r}")
                    continue
                
                # Extract command and scores
                processed_result = self._extract_result_data(r, i)
                if processed_result:
                    valid_results.append(processed_result)
                    logger.debug(f"Added valid result: {processed_result[0]} (score: {processed_result[1]})")
                    
            except Exception as e:
                logger.error(f"Unexpected error processing result at index {i}: {e}, result: {r}")
                continue
        
        return valid_results
    
    def _extract_result_data(self, result, index):
        """Extract command and score data from a result."""
        try:
            # Handle different result formats
            if len(result) >= 2:
                cmd = str(result[0]).strip()
                if not cmd:
                    logger.warning(f"Empty command at index {index}")
                    return None
                
                # Extract scores - handle various formats
                score = result[1] if len(result) > 1 else 0.0
                sem_score = result[2] if len(result) > 2 else 0.0
                bm25_score = result[3] if len(result) > 3 else 0.0
                
                # Convert scores to float
                try:
                    score = float(score)
                    sem_score = float(sem_score) if isinstance(sem_score, (int, float)) else 0.0
                    bm25_score = float(bm25_score) if isinstance(bm25_score, (int, float)) else 0.0
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting scores to float at index {index}: {e}")
                    return None
                
                return (cmd, score, sem_score, bm25_score)
            else:
                logger.warning(f"Result at index {index} has insufficient data: {len(result)} elements")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting result data at index {index}: {e}")
            return None
    
    def _create_result_widgets(self, valid_results, ui_controller):
        """Create SearchResult widgets for valid results."""
        for cmd, score, sem_score, bm25_score in valid_results:
            widget = SearchResult(
                cmd, score, ui_controller.get_search_mode(), 
                sem_score, bm25_score, ui_controller.get_show_scores()
            )
            ui_controller.add_result_widget(widget)
    
    def _set_initial_focus(self, ui_controller):
        """Set initial focus and trigger description generation."""
        if not ui_controller.has_results():
            return
            
        try:
            # Set focus to first item
            ui_controller.set_focus(0)
            
            # Force visual selection on first item
            first_widget = ui_controller.get_first_result()
            if first_widget and hasattr(first_widget, '_create_widget'):
                first_widget._last_focus_state = None  # Invalidate cached state
                first_widget._create_widget(focused=True)
                first_widget._set_w(first_widget._widget)  # Update WidgetWrap
                logger.debug(f"🔄 Applied initial focus styling to: {getattr(first_widget, 'command', 'unknown')}")
            
            # Generate description for first item
            if first_widget and hasattr(first_widget, 'command'):
                ui_controller.generate_description(first_widget.command)
                logger.info(f"🎯 INITIAL DESCRIPTION generated for: '{first_widget.command}'")
            
            logger.info(f"🎯 INITIAL FOCUS set to index 0 with description generation")
        except Exception as e:
            logger.error(f"Error setting initial focus: {e}")


class IngestionProgressTUI:
    """TUI for showing ingestion progress with clean, professional interface"""
    
    def __init__(self, no_random=False):
        self.loop = None
        self.progress_bar = None
        self.status_text = None
        self.stats_text = None
        self.current_command_text = None
        self.phase_text = None
        self.running = False
        
        # Progress tracking
        self.total_commands = 0
        self.processed_commands = 0
        self.current_phase = "Initializing..."
        self.current_command = ""
        self.start_time = None
        self.no_random = no_random
        
        # Witty remarks for loading screen
        self.witty_remarks = [
            "Solving the universe, one command at a time...",
            "Chasing my own tail... but productively.",
            "Reticulating splines... for maximum fuzziness.",
            "Herding cats... also known as tokenizing commands.",
            "Polishing the bits... they were looking a bit dull.",
            "Teaching the model to fish for commands.",
            "Warming up the neural nets... they get cranky when cold.",
            "Finding the meaning of life in your shell history.",
            "Don't worry, I'm a professional... at being fuzzy.",
            "Making the command line feel... well, fuzzier.",
            "Generating witty remarks... this is the best one.",
            "Almost there... maybe.",
            "Just a few more moments of suspense...",
            "Are we there yet? No, but the journey is beautiful.",
            "I'm not sleeping, I'm just processing... at the speed of thought.",
        ]
        self.last_remark_index = -1
        self.last_remark_time = 0.0 # Initialize last_remark_time
        
    def setup_ui(self):
        """Create the TUI layout"""
        # Header
        header = urwid.Text([
            ('bold', 'FuzzyShell'), ' - Ingesting Command History\n'
        ], align='center')
        
        # Phase indicator
        self.phase_text = urwid.Text("Phase: Initializing...", align='center')
        
        # Progress bar
        self.progress_bar = urwid.ProgressBar('pg normal', 'pg complete', current=0, done=100)
        
        # Status text
        self.status_text = urwid.Text("Preparing to ingest commands...", align='center')
        
        # Current command being processed
        self.current_command_text = urwid.Text("", align='center')
        
        # Statistics
        self.stats_text = urwid.Text("", align='center')
        
        # Instructions
        footer = urwid.Text("Press ESC to cancel (will exit gracefully)", align='center')
        
        # Layout
        pile = urwid.Pile([
            ('pack', urwid.Divider()),
            ('pack', header),
            ('pack', urwid.Divider()),
            ('pack', self.phase_text),
            ('pack', urwid.Divider()),
            ('pack', urwid.Text("Progress:", align='center')),
            ('pack', self.progress_bar),
            ('pack', urwid.Divider()),
            ('pack', self.status_text),
            ('pack', urwid.Divider()),
            ('pack', self.current_command_text),
            ('pack', urwid.Divider()),
            ('pack', self.stats_text),
            ('pack', urwid.Divider()),
            ('pack', urwid.Divider()),
            ('pack', footer),
        ])
        
        # Add padding
        padded = urwid.Padding(pile, align='center', width=('relative', 80))
        filled = urwid.Filler(padded, valign='middle')
        
        return filled
        
    def start(self):
        """Start the TUI"""
        self.running = True
        self.start_time = time.time()
        
        # Setup UI
        main_widget = self.setup_ui()
        
        # Color palette
        palette = [
            ('pg normal', 'white', 'dark blue'),
            ('pg complete', 'white', 'dark green'),
            ('bold', 'bold', ''),
        ]
        
        # Create main loop
        self.loop = urwid.MainLoop(
            main_widget,
            palette=palette,
            handle_mouse=False,
            unhandled_input=self._handle_input
        )
        
        return self.loop
        
    def _handle_input(self, key):
        """Handle keyboard input"""
        if key == 'esc':
            self.running = False
            raise urwid.ExitMainLoop()
    
    def get_status_text(self, message):
        """Get the appropriate status text based on no_random setting."""
        if self.no_random:
            return message
        else:
            # Show witty remarks, ensuring each stays for at least 5 seconds
            current_time = time.time()
            if current_time - self.last_remark_time >= 5.0:
                # Pick a new, random remark that wasn't the last one
                remark_index = self.last_remark_index
                while remark_index == self.last_remark_index:
                    remark_index = random.randint(0, len(self.witty_remarks) - 1)
                self.last_remark_index = remark_index
                self.last_remark_time = current_time
                
                return self.witty_remarks[remark_index]
            # Return None to keep current remark until 5 seconds have passed
            return None
    
    def get_command_display(self, current_command):
        """Get the appropriate command display text based on no_random setting."""
        if not current_command:
            return ""
            
        if self.no_random:
            # Show command being processed when --no-random is used
            # Truncate long commands
            display_cmd = current_command
            if len(display_cmd) > 60:
                display_cmd = display_cmd[:57] + "..."
            return f"Processing: {display_cmd}"
        else:
            # Hide current command in witty mode - keep it clean
            return ""
    
    def get_stats_text(self):
        """Get the appropriate statistics text based on no_random setting."""
        if not self.start_time:
            return ""
            
        if self.no_random:
            # Show detailed progress when --no-random is used
            elapsed = time.time() - self.start_time
            if self.processed_commands > 0:
                rate = self.processed_commands / elapsed
                eta = (self.total_commands - self.processed_commands) / rate if rate > 0 else 0
                return (f"Processed: {self.processed_commands}/{self.total_commands} commands  "
                       f"Rate: {rate:.1f}/s  ETA: {eta:.1f}s")
            else:
                return f"Elapsed: {elapsed:.1f}s"
        else:
            # Hide detailed stats in witty mode - keep it clean
            return ""
            
    def update_progress(self, progress, message, phase=None, current_command=None):
        """Update the progress display"""
        if not self.loop:
            return
            
        try:
            # Update progress bar
            if self.progress_bar:
                self.progress_bar.set_completion(min(100, max(0, progress)))
            
            # Update status message
            if self.status_text:
                status_text = self.get_status_text(message)
                if status_text is not None:
                    self.status_text.set_text(status_text)
            
            # Update phase if provided
            if phase and self.phase_text:
                self.current_phase = phase
                self.phase_text.set_text(f"Phase: {phase}")
            
            # Update current command if provided
            if current_command and self.current_command_text:
                self.current_command = current_command
                command_text = self.get_command_display(current_command)
                self.current_command_text.set_text(command_text)
            
            # Update statistics
            if self.stats_text:
                stats_text = self.get_stats_text()
                self.stats_text.set_text(stats_text)
            
            # Refresh the display
            if hasattr(self.loop, 'draw_screen'):
                self.loop.draw_screen()
                
        except Exception as e:
            logger.error(f"Error updating progress display: {e}")
    
    def set_total_commands(self, total):
        """Set the total number of commands to process"""
        self.total_commands = total
        
    def increment_processed(self):
        """Increment the count of processed commands"""
        self.processed_commands += 1
        
    def finish(self, message="Ingestion complete!"):
        """Finish the progress display"""
        if self.loop:
            self.update_progress(100, message, "Complete")
            # Give user a moment to see completion
            time.sleep(1.5)
            self.running = False
            raise urwid.ExitMainLoop()


class FuzzyShellApp:
    def __init__(self, search_callback, fuzzyshell_instance=None):
        self.search_callback = search_callback
        self.fuzzyshell = fuzzyshell_instance
        self.show_scores = True  # Show scores by default
        self.search_mode = "hybrid"
        self.current_results = []
        self._last_search_time = 0
        self._pending_search = None
        self._current_search_cancelled = False
        self._current_query = None
        self._selected_command = None
        
        # Initialize managers
        self.search_manager = SearchManager(fuzzyshell_instance, search_callback)
        self.result_renderer = ResultRenderer()

        # Create widgets with better styling
        self.input_box = WatermarkEdit(
            caption="Search: ", 
            edit_text="", 
            watermark_text="Type to search commands... (^S for scores)"
        )
        self.loading_indicator = LoadingIndicator()
        self.results_list = urwid.SimpleFocusListWalker([])
        self.results_box = urwid.ListBox(self.results_list)
        
        # Connect to focus change events using urwid's proper observer pattern
        urwid.connect_signal(self.results_list, 'modified', self._on_focus_changed)
        
        self.description_pane = CommandDescriptionPane()
        self.footer = StatusFooter()
        
        # Create search status indicator
        self.search_status = urwid.Text("Idle", align='right')
        
        # Create input area with status indicator on the right
        input_area = urwid.Columns([
            ('weight', 1, urwid.AttrMap(self.input_box, 'input')),
            ('pack', urwid.AttrMap(self.search_status, 'search_status'))
        ])
        
        # Create main body with compact spacing
        body_content = urwid.Pile([
            ('pack', input_area),
            ('pack', urwid.AttrMap(self.loading_indicator, 'loading')),
            ('weight', 3, urwid.AttrMap(self.results_box, 'results')),
            ('pack', urwid.AttrMap(urwid.Divider('─'), 'description_divider')),
            ('pack', urwid.AttrMap(self.description_pane, 'description_box')),
        ])
        
        self.main_layout = urwid.Frame(
            body=urwid.AttrMap(body_content, 'body'),
            footer=urwid.AttrMap(self.footer, 'footer')
        )

        urwid.connect_signal(self.input_box, 'change', self.on_input_changed)
        
        # Initialize database info and try to pre-load model
        if self.fuzzyshell:
            try:
                db_info = self.fuzzyshell.get_database_info()
                self.footer.update(item_count=db_info['item_count'], embedding_model=db_info['embedding_model'])
                
                # Try to initialize model synchronously with short timeout
                logger.debug("Attempting to pre-initialize model...")
                model_ready = False
                try:
                    # Try a quick model initialization
                    model_ready = self.fuzzyshell.wait_for_model(timeout=5.0)
                    if model_ready:
                        logger.info("Model pre-initialized successfully")
                    else:
                        logger.error("❌ SEMANTIC MODEL FAILED TO INITIALIZE - this needs to be fixed!")
                except Exception as e:
                    logger.error(f"❌ SEMANTIC MODEL EXCEPTION: {e}")
                    
            except Exception as e:
                logger.error(f"❌ DATABASE ERROR: {e}")
        
        # Set initial focus to input box
        self._focus_input()

    def on_input_changed(self, widget, new_text):
        # Debug logging for query changes
        logger.info("🔍 Input changed: '%s' -> '%s'", getattr(self, '_current_query', ''), new_text.strip())
        
        # Cancel any pending search
        if self._pending_search:
            logger.debug("Cancelling pending search")
            self.loop.remove_alarm(self._pending_search)
        
        # Cancel any currently running search
        self._current_search_cancelled = True
        
        if not new_text.strip():
            self._clear_results()
            return
        
        # Set up new search
        self._current_search_cancelled = False
        self._current_query = new_text.strip()
        logger.info("📅 Scheduling search for: '%s' (150ms delay)", self._current_query)
        self._pending_search = self.loop.set_alarm_in(0.15, self._perform_search_callback, new_text.strip())  # Increased to 150ms

    def _perform_search_callback(self, loop, query):
        """Callback for the alarm - proper urwid callback signature."""
        logger.info("🚀 Executing debounced search for: '%s'", query)
        self._pending_search = None
        self._perform_search(query)
    
    def _perform_search(self, query):
        """Perform the actual search operation using SearchManager."""
        # Set current query and perform search
        self.search_manager.set_current_query(query)
        self._current_query = query
        self._current_search_cancelled = False
        
        # Execute search using SearchManager
        results = self.search_manager.execute_search(query, self)
        self.current_results = results
    
    # UI Controller Interface methods for SearchManager
    
    def has_loop(self):
        """Check if UI loop is available."""
        return hasattr(self, 'loop') and self.loop is not None
    
    def start_loading(self, message):
        """Start loading indicator with message."""
        if self.has_loop():
            self.loading_indicator.start_loading(self.loop, message)
            self.loop.draw_screen()
    
    def stop_loading(self):
        """Stop loading indicator."""
        if self.has_loop():
            self.loading_indicator.stop_loading(self.loop)
    
    def update_loading(self, message):
        """Update loading indicator message."""
        if self.has_loop():
            self.loading_indicator.stop_loading(self.loop)
            self.loading_indicator.start_loading(self.loop, message)
    
    def set_search_status(self, status):
        """Update search status text."""
        self.search_status.set_text(status)
    
    def update_footer(self, search_time=None):
        """Update footer with search time."""
        if search_time is not None:
            self.footer.update(search_time=search_time)
    
    def display_results(self, results):
        """Display search results using ResultRenderer."""
        self.result_renderer.render_results(results, self)
    
    def show_error(self, error_msg):
        """Show error message in results area."""
        self.results_list.clear()
        self.results_list.append(urwid.Text(error_msg, align='center'))
        self.current_results = []
    
    def redraw(self):
        """Force screen redraw."""
        if self.has_loop():
            self.loop.draw_screen()
    
    # Additional UI Controller methods for ResultRenderer
    
    def clear_results(self):
        """Clear the results list."""
        self.results_list.clear()
    
    def show_no_results(self):
        """Show 'no matches found' message."""
        self.results_list.append(urwid.Text("No matches found", align='center'))
        self.description_pane.description = ""
        self.description_pane.update()
    
    def show_no_valid_results(self):
        """Show 'no valid matches found' message."""
        self.results_list.append(urwid.Text("No valid matches found", align='center'))
        self.description_pane.description = ""
        self.description_pane.update()
    
    def get_search_mode(self):
        """Get current search mode."""
        return self.search_mode
    
    def get_show_scores(self):
        """Get show scores setting."""
        return self.show_scores
    
    def add_result_widget(self, widget):
        """Add a result widget to the list."""
        self.results_list.append(widget)
    
    def has_results(self):
        """Check if results list has items."""
        return self.results_list and len(self.results_list) > 0
    
    def set_focus(self, index):
        """Set focus to specific result index."""
        if self.has_results() and 0 <= index < len(self.results_list):
            self.results_box.set_focus(index)
    
    def get_first_result(self):
        """Get the first result widget."""
        if self.has_results():
            return self.results_list[0]
        return None
    
    def generate_description(self, command):
        """Generate description for a command."""
        loop = getattr(self, 'loop', None) if hasattr(self, 'loop') else None
        self.description_pane.generate_description_async(command, loop)

    def _display_results(self, results):
        self.results_list.clear()
        if not results:
            self.results_list.append(urwid.Text("No matches found", align='center'))
            self.description_pane.description = ""
            self.description_pane.update()
            return

        # Process results with robust error handling
        valid_results = []
        logger.debug(f"Processing {len(results)} raw results")
        
        for i, r in enumerate(results):
            try:
                # Log the raw result for debugging
                logger.debug(f"Processing result {i}: {type(r)} = {r}")
                
                # Handle None or empty results
                if r is None:
                    logger.warning(f"Skipping None result at index {i}")
                    continue
                    
                if not r:
                    logger.warning(f"Skipping empty result at index {i}: {r}")
                    continue
                    
                # Ensure r is a sequence (tuple/list)
                if not hasattr(r, '__len__') or not hasattr(r, '__getitem__'):
                    logger.warning(f"Skipping non-sequence result at index {i}: {type(r)} = {r}")
                    continue
                    
                if len(r) < 2:
                    logger.warning(f"Skipping malformed result at index {i} (length {len(r)}): {r}")
                    continue
                    
                # Extract values based on result length
                try:
                    if len(r) >= 4:
                        cmd, score, sem_score, bm25_score = r[0], r[1], r[2], r[3]
                    elif len(r) >= 2:
                        cmd, score = r[0], r[1]
                        sem_score, bm25_score = 0.0, 0.0
                    else:
                        continue  # Skip invalid results
                except (IndexError, ValueError) as e:
                    logger.warning(f"Error unpacking result at index {i}: {e}, result: {r}")
                    continue
                    
                # Validate that cmd is a string and score is numeric
                if not isinstance(cmd, str):
                    logger.warning(f"Skipping result at index {i} with non-string command: cmd={type(cmd)} = {cmd}")
                    continue
                    
                if not isinstance(score, (int, float)):
                    logger.warning(f"Skipping result at index {i} with non-numeric score: score={type(score)} = {score}")
                    continue
                    
                # Convert to floats to ensure consistency
                try:
                    score = float(score)
                    sem_score = float(sem_score) if isinstance(sem_score, (int, float)) else 0.0
                    bm25_score = float(bm25_score) if isinstance(bm25_score, (int, float)) else 0.0
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting scores to float at index {i}: {e}")
                    continue
                    
                valid_results.append((cmd, score, sem_score, bm25_score))
                logger.debug(f"Added valid result: {cmd} (score: {score})")
                
            except Exception as e:
                logger.error(f"Unexpected error processing result at index {i}: {e}, result: {r}")
                continue
        
        # Display valid results or show "no matches" if none are valid
        if not valid_results:
            self.results_list.append(urwid.Text("No valid matches found", align='center'))
            self.description_pane.description = ""
            self.description_pane.update()
            return
            
        for cmd, score, sem_score, bm25_score in valid_results:
            self.results_list.append(SearchResult(cmd, score, self.search_mode, sem_score, bm25_score, self.show_scores))
        
        if self.results_list and len(self.results_list) > 0:
            try:
                # Set focus to first item AND generate description + show visual selection
                self.results_box.set_focus(0)
                
                # Force visual selection indicator on first item
                first_widget = self.results_list[0]
                if hasattr(first_widget, '_create_widget'):
                    first_widget._last_focus_state = None  # Invalidate cached state
                    first_widget._create_widget(focused=True)
                    first_widget._set_w(first_widget._widget)  # Update WidgetWrap
                    logger.debug(f"🔄 Applied initial focus styling to: {getattr(first_widget, 'command', 'unknown')}")
                
                # Generate description for first item
                first_widget = self.results_list[0]
                if hasattr(first_widget, 'command'):
                    loop = getattr(self, 'loop', None) if hasattr(self, 'loop') else None
                    self.description_pane.generate_description_async(first_widget.command, loop)
                    logger.info(f"🎯 INITIAL DESCRIPTION generated for: '{first_widget.command}'")
                
                logger.info(f"🎯 INITIAL FOCUS set to index 0 with description generation")
            except Exception as e:
                logger.error(f"Error setting initial focus: {e}")

    def _clear_results(self):
        self.results_list.clear()
        self.current_results = []
        self.description_pane.description = ""
        self.description_pane.update()
        self.footer.update(search_time=0.0)
        # Reset search status to idle
        self.search_status.set_text("Idle")
        # Also stop any loading indicator
        if hasattr(self, 'loading_indicator') and hasattr(self, 'loop') and self.loop:
            self.loading_indicator.stop_loading(self.loop)
        if hasattr(self, 'loop') and self.loop:
            self.loop.draw_screen()


    def unhandled_input(self, key):
        # DEBUG: Track ALL keys that reach unhandled_input
        logger.info(f"🔑 UNHANDLED_INPUT received key: '{key}' (type: {type(key)})")
        
        # Handle help dialog close first
        if hasattr(self, 'help_overlay') and self.loop.widget == self.help_overlay:
            if key == KEY_QUIT or key == 'close_help':
                self._close_help_dialog()
                return
            elif key == KEY_HELP:
                self._close_help_dialog()
                return
            # Let help dialog handle its own keys
            return
        
        # Handle expert screen close
        if hasattr(self, 'expert_screen_overlay') and self.loop.widget == self.expert_screen_overlay:
            if key == KEY_QUIT or key == 'close_expert_screen':
                self._close_expert_screen()
                return
            elif key == KEY_EXPERT_SCREEN:
                self._close_expert_screen()
                return
            # Let expert screen handle its own keys
            return
        
        # Special debug mode - press F1 to enable key logging
        if key == 'f1':
            logger.info("Debug mode: Press any key to see its representation, ESC to continue")
            return
        
        # Global shortcuts that work regardless of focus
        if key in ('q', 'Q', KEY_QUIT, 'escape'):
            raise urwid.ExitMainLoop()
        
        elif key == KEY_SELECT:
            # Execute selected command if in results, otherwise do nothing
            if self.results_list and len(self.results_list) > 0 and self.main_layout.focus_position == 'body':
                try:
                    selection = self.results_box.focus
                    if selection:
                        # Handle both WidgetWrap (SearchResult) and direct widgets
                        command = None
                        if hasattr(selection, 'command'):
                            command = selection.command
                        elif hasattr(selection, 'original_widget') and hasattr(selection.original_widget, 'command'):
                            command = selection.original_widget.command
                        
                        if command:
                            # Store the result for retrieval
                            self._selected_command = command
                            raise urwid.ExitMainLoop()
                        else:
                            logger.debug(f"Selection has no command attribute: {type(selection)}")
                except (AttributeError, IndexError) as e:
                    logger.debug(f"Error handling enter key: {e}")
            return

        # Tab key functionality removed - search mode is always hybrid

        elif key == KEY_HELP:
            # Show help dialog
            self._show_help_dialog()
            return
        
        elif key == KEY_EXPERT_SCREEN:
            # Show expert screen / internals  
            self._show_expert_screen()
            return
            
        elif key in (KEY_TOGGLE_SCORES, '\x13', 'ctrl S', '\x1f', KEY_TOGGLE_SCORES_ALT, 'S'):  # Multiple ways to toggle scores
            # Toggle score display
            logger.debug(f"unhandled_input: Got score toggle key '{key}', toggling scores from {self.show_scores}")
            self.show_scores = not self.show_scores
            self.footer.update(show_scores=self.show_scores)
            self._display_results(self.current_results)
            logger.debug(f"unhandled_input: Scores now {self.show_scores}")
            return
        
        elif key == KEY_REFRESH_SEARCH:
            # Refresh search
            if self.input_box.edit_text.strip():
                self._perform_search(self.input_box.edit_text.strip())
            return
        
        # Handle navigation keys for results - let urwid handle them naturally
        elif key in NAVIGATION_KEYS:
            if self.results_list and len(self.results_list) > 0:
                # Navigate to results if not already focused
                self._focus_results()
                # Let urwid handle the navigation naturally - focus change observer will handle description
                logger.debug(f"🔑 Navigation key '{key}' - letting urwid handle naturally")
            return
        
        # For all other keys, let the focused widget handle them
        # Use reasonable terminal size estimate
        self.main_layout.keypress((80, 24), key)
    
    
    def _focus_results(self):
        """Ensure results box has focus for navigation."""
        # Simplified - just set main focus to body and let urwid handle the rest
        try:
            if self.results_list and len(self.results_list) > 0:
                self.main_layout.focus_position = 'body'
        except Exception as e:
            logger.debug(f"Error focusing results: {e}")
    
    def _focus_input(self):
        """Focus the input box for typing."""
        # Simplified - just set main focus to body where input is first
        try:
            self.main_layout.focus_position = 'body'
        except Exception as e:
            logger.debug(f"Error focusing input: {e}")
    
    def _show_help_dialog(self):
        """Show the help dialog as an overlay."""
        help_dialog = HelpDialog()
        overlay = urwid.Overlay(
            help_dialog,
            self.main_layout,
            align='center',
            width=('relative', 80),
            valign='middle',
            height=('relative', 90)
        )
        self.help_overlay = overlay
        self.loop.widget = overlay
    
    def _close_help_dialog(self):
        """Close the help dialog and return to main view."""
        self.loop.widget = self.main_layout
    
    def _show_expert_screen(self):
        """Show the expert screen as an overlay."""
        expert_screen = ExpertScreen(self.fuzzyshell)
        overlay = urwid.Overlay(
            expert_screen,
            self.main_layout,
            align='center',
            width=('relative', 95),
            valign='middle',
            height=('relative', 95)
        )
        self.expert_screen_overlay = overlay
        self.loop.widget = overlay
    
    def _close_expert_screen(self):
        """Close the expert screen and return to main view."""
        self.loop.widget = self.main_layout
    
    def _on_focus_changed(self):
        """Handle focus changes in the results list using urwid's observer pattern."""
        try:
            if not self.results_list or len(self.results_list) == 0:
                return
                
            # Get the currently focused widget
            focus_position = self.results_list.focus
            focused_widget = self.results_list[focus_position] if focus_position < len(self.results_list) else None
            
            if focused_widget and hasattr(focused_widget, 'command'):
                # Avoid redundant description generation for the same command
                current_command = getattr(self.description_pane, '_current_command', None)
                if current_command != focused_widget.command:
                    logger.info(f"🎯 FOCUS CHANGED to: '{focused_widget.command}' (position {focus_position})")
                    
                    # Generate description for the newly focused item
                    loop = getattr(self, 'loop', None) if hasattr(self, 'loop') else None
                    self.description_pane.generate_description_async(focused_widget.command, loop)
                else:
                    logger.debug(f"Focus changed but command is same: '{focused_widget.command}'")
            else:
                logger.debug(f"Focus changed but widget has no command: {type(focused_widget)}")
                
        except Exception as e:
            logger.error(f"Error in focus change handler: {e}")
            import traceback
            logger.error(f"Focus change error traceback: {traceback.format_exc()}")


    def run(self):
        # Disable terminal flow control to allow Ctrl+S
        import subprocess
        try:
            subprocess.run(['stty', '-ixon'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not disable terminal flow control - Ctrl+S may not work")
        
        self.palette = [
            # Basic UI elements - use terminal defaults
            ('body', 'default', 'default'),
            ('header', 'light cyan', 'default'),
            ('footer', 'dark gray', 'default'),
            ('reversed', 'default,standout', 'default'),
            ('expert_screen', 'default', 'default'),
            ('title', 'light cyan,bold', 'default'),
            ('section_header', 'light green,bold', 'default'),
            
            
            # Input area - minimal styling
            ('input', 'white', 'default'),
            ('prompt', 'dark gray', 'default'),
            ('placeholder_text', 'dark gray', 'default'),
            ('search_status', 'dark gray', 'default'),
            
            # Results area - clean default background  
            ('results', 'default', 'default'),
            
            # Loading indicator
            ('loading', 'default', 'default'),
            
            # Description box - subtle styling only
            ('description_box', 'default', 'default'),
            ('description_divider', 'dark gray', 'default'),
            
            # Text colors - more subtle
            ('light blue', 'light blue', 'default'),
            ('light green', 'light green', 'default'),
            ('light magenta', 'light magenta', 'default'),
            ('light cyan', 'light cyan', 'default'),
            ('dark blue', 'dark blue', 'default'),
            ('dark green', 'dark green', 'default'),
            ('dark gray', 'dark gray', 'default'),
            ('dark cyan', 'dark cyan', 'default'),
            ('white', 'white', 'default'),
            ('bold', 'white,bold', 'default'),
            ('yellow', 'yellow', 'default'),
            ('orange', 'light red', 'default'),
            ('light red', 'light red', 'default'),
        ]

        # Add exception handling wrapper
        def exception_wrapper(loop, context):
            import traceback
            logger.error(f"Unhandled exception in urwid: {context}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        self.loop = urwid.MainLoop(
            self.main_layout, 
            self.palette, 
            unhandled_input=self.unhandled_input,
            handle_mouse=False  # Disable mouse handling to prevent crashes
        )

        try:
            self.loop.run()
            # Return the selected command if any
            return self._selected_command
        except Exception as e:
            logger.error(f"TUI Error: {e}")
            # Log the full stack trace to help debug
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

def run_ui(search_callback, fuzzyshell_instance):
    return FuzzyShellApp(search_callback, fuzzyshell_instance).run()
