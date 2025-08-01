import time
import urwid
import threading
import logging

logger = logging.getLogger('FuzzyShell.TUI')

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
        if key in ('ctrl s', '\x13', 'ctrl S', '\x1f'):  # Multiple representations of Ctrl+S
            # Let the parent handle this global command
            logger.debug(f"WatermarkEdit: Passing through ctrl s: '{key}'")
            return key
        elif key == 'ctrl r':
            # Let the parent handle this global command  
            return key
        elif key in ('esc', 'escape'):
            # Let the parent handle escape
            return key
        elif key in ('up', 'down', 'page up', 'page down', 'home', 'end'):
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
    __version__ = "0.1.0"

class SearchModeIndicator(urwid.Text):
    def __init__(self):
        super().__init__("")
        self.search_mode = "hybrid"
        self.update()

    def set_search_mode(self, mode):
        self.search_mode = mode
        self.update()

    def update(self):
        mode_icons = {
            "semantic": "🧠",
            "keyword": "🔍",
            "hybrid": "⚡"
        }
        mode_colors = {
            "semantic": "light blue",
            "keyword": "light green",
            "hybrid": "light magenta"
        }
        icon = mode_icons.get(self.search_mode, "⚡")
        color = mode_colors.get(self.search_mode, "light magenta")
        
        text = [
            (color, f"{icon} "),
            ('bold', (color, self.search_mode.upper()))
        ]
        self.set_text(text)

class StatusFooter(urwid.WidgetWrap):
    def __init__(self):
        self.item_count = 0
        self.embedding_model = "unknown"
        self.search_time = 0.0
        self.show_scores = False

        self._left_text_widget = urwid.Text("")
        self._right_text_widget = urwid.Text("", align='right')
        
        super().__init__(urwid.Columns([
            self._left_text_widget,
            self._right_text_widget
        ]))
        self.update()

    def update(self, item_count=None, embedding_model=None, search_time=None, show_scores=None):
        if item_count is not None:
            self.item_count = item_count
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if search_time is not None:
            self.search_time = search_time
        if show_scores is not None:
            self.show_scores = show_scores

        left_text = [('bold', ('dark cyan', f"FuzzyShell v{__version__}"))]
        if self.item_count > 0:
            left_text.append(('dark gray', f" • {self.item_count:,} items"))
            if self.embedding_model != "unknown":
                model_short = self.embedding_model.replace("minilm-l6-v2-terminal-describer", "Terminal-MiniLM")
                model_short = model_short.replace("all-MiniLM-L6-v2", "MiniLM-L6")
                left_text.append(('dark blue', f" ({model_short})"))
        if self.search_time > 0:
            left_text.append(('dark green', f" • {self.search_time*1000:.0f}ms"))

        right_text = [
            ('bold', "ESC"), ('dark gray', " quit "),
            ('bold', "↑↓"), ('dark gray', " navigate "),
            ('bold', "ENTER"), ('dark gray', " select "),
            ('bold', "^S/S"), ('dark gray', " scores(ON)" if self.show_scores else " scores(OFF)")
        ]
        
        self._left_text_widget.set_text(left_text)
        self._right_text_widget.set_text(right_text)

class CommandDescriptionPane(urwid.Text):
    def __init__(self):
        super().__init__("")
        self._description_handler = None
        self._current_command = None
        self._init_lock = threading.Lock()
        self._generation_id = 0  # Track generation requests
        self._pending_generation = None  # Track pending delayed generation
        self._spinner_alarm = None  # Track spinner animation alarm
        self.is_loading = False
        self.description = ""
        self.update()

    def update(self):
        if self.is_loading:
            spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spinner_index = int(time.time() * 10) % len(spinner_chars)
            spinner = spinner_chars[spinner_index]
            text = [('dark cyan', f"{spinner} "), ('italic', "Generating description...")]
        elif self._pending_generation is not None:
            # Show a pending state during debounce delay
            text = [('dark gray', "⏳ "), ('italic', "Loading...")]
        elif self.description:
            text = [('light blue', "💡 "), ('white', self.description)]
        else:
            text = [('dark gray', "💡 "), ('dark gray', "Select a command to see its description")]
        self.set_text(text)
    
    def _start_spinner(self, loop):
        """Start the animated spinner."""
        if loop and not self._spinner_alarm:
            self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
    
    def _stop_spinner(self, loop):
        """Stop the animated spinner."""
        if loop and self._spinner_alarm:
            try:
                loop.remove_alarm(self._spinner_alarm)
            except:
                pass
            finally:
                self._spinner_alarm = None
    
    def _spinner_callback(self, loop, user_data):
        """Update spinner animation."""
        if self.is_loading:
            self.update()
            # Schedule next update
            self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
        else:
            self._spinner_alarm = None

    def _init_description_handler(self):
        with self._init_lock:
            if self._description_handler is None:
                try:
                    from .model_handler import DescriptionHandler
                    self._description_handler = DescriptionHandler()
                except Exception as e:
                    logger.error("Failed to initialize DescriptionHandler: %s", str(e))
                    self._description_handler = False
        return self._description_handler if self._description_handler is not False else None

    def generate_description_async(self, command: str, loop=None):
        if not command:
            return
        
        # DEBUG: Log every generation request
        logger.info(f"🔔 GENERATION REQUESTED: '{command}'")
        logger.info(f"   Current state: cmd='{self._current_command}', desc_len={len(self.description)}, loading={self.is_loading}")
            
        # Skip if same command and already loaded (optimization)
        # TEMPORARILY DISABLED FOR DEBUGGING - always generate
        if False and (command == self._current_command and 
            self.description and 
            not self.is_loading):
            logger.debug(f"Optimization skip for command: {command}")
            return
        
        # Cancel any pending delayed generation
        if self._pending_generation and loop:
            try:
                loop.remove_alarm(self._pending_generation)
                self._pending_generation = None
            except:
                pass
        
        # Update current command immediately for UI feedback
        self._current_command = command
        
        # Use a short delay to allow for rapid scrolling without constant regeneration
        # This gives better UX during fast navigation while still being responsive
        if loop:
            self._pending_generation = loop.set_alarm_in(
                0.1,  # 100ms delay - fast enough to feel responsive, slow enough to debounce
                lambda loop, data: self._start_generation(command, loop),
                command
            )
            
            # Clear old description and update UI to show pending state
            # Do this AFTER setting _pending_generation so update() shows the right state
            self.description = ""
            self.is_loading = False  # Reset loading state
            self.update()
            # Draw screen immediately to show the change
            try:
                loop.draw_screen()
            except:
                pass
        else:
            # No loop available, generate immediately
            self._start_generation(command, loop)
    
    def _start_generation(self, command: str, loop=None):
        """Start the actual description generation after debounce delay."""
        # Clear pending generation since we're starting now
        self._pending_generation = None
        
        # Increment generation ID to invalidate any previous requests
        self._generation_id += 1
        current_generation_id = self._generation_id
        
        # Set loading state
        self.is_loading = True
        self.description = ""
        self.update()
        
        # Start animated spinner
        self._start_spinner(loop)
        
        # Only draw screen if loop is available
        if loop:
            try:
                loop.draw_screen()
            except Exception as e:
                logger.debug(f"Could not draw screen: {e}")
                
        def generate_in_thread():
            try:
                # Check if this generation is still current
                if self._generation_id != current_generation_id:
                    logger.debug(f"Generation {current_generation_id} cancelled (current: {self._generation_id})")
                    return
                    
                handler = self._init_description_handler()
                if handler is None:
                    desc = f"Command: {command}"
                else:
                    desc = handler.generate_description(command)
                
                # Check again after generation completes
                if self._generation_id != current_generation_id:
                    logger.debug(f"Generation {current_generation_id} result discarded (current: {self._generation_id})")
                    return
                    
                # Use alarm to update from main thread instead of call_from_thread
                if loop:
                    try:
                        loop.set_alarm_in(0, lambda loop, data: self._update_description(desc, current_generation_id, loop))
                    except Exception as e:
                        logger.debug(f"Could not set alarm: {e}")
                        # Fallback - direct update
                        self._update_description(desc, current_generation_id, loop)
                else:
                    # Fallback for when no loop is available
                    self._update_description(desc, current_generation_id, loop)
            except Exception as e:
                logger.error("Error in description generation thread: %s", str(e))
                # Only show fallback if this generation is still current
                if self._generation_id == current_generation_id:
                    fallback_desc = f"Command: {command}"
                    if loop:
                        try:
                            loop.set_alarm_in(0, lambda loop, data: self._update_description(fallback_desc, current_generation_id, loop))
                        except Exception as e:
                            logger.debug(f"Could not set alarm for fallback: {e}")
                            self._update_description(fallback_desc, current_generation_id, loop)
                    else:
                        self._update_description(fallback_desc, current_generation_id, loop)
        
        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()

    def _update_description(self, desc: str, generation_id: int = None, loop=None):
        # Only update if this is the current generation (prevents race conditions)
        if generation_id is None or self._generation_id == generation_id:
            self.is_loading = False
            self.description = desc
            # Stop the animated spinner
            self._stop_spinner(loop)
            self.update()
        else:
            logger.debug(f"Ignoring description update from generation {generation_id} (current: {self._generation_id})")

class SearchResult(urwid.WidgetWrap):
    def __init__(self, command, score, search_mode, semantic_score, bm25_score, show_scores):
        self.command = command
        self.score = score
        self.search_mode = search_mode
        self.semantic_score = semantic_score
        self.bm25_score = bm25_score
        self.show_scores = show_scores
        
        # Create the actual widget content
        self._create_widget()
        super().__init__(self._widget)
    
    def _create_widget(self, focused=False):
        """Create the underlying widget with proper text content."""
        try:
            result = self._build_text_content(focused)
            if isinstance(result, tuple) and len(result) == 2:
                left_text, right_text = result
            else:
                # Fallback for unexpected return format
                left_text = [('default', self.command)]
                right_text = []
                logger.debug(f"Unexpected _build_text_content result: {result}")
        except Exception as e:
            # Defensive fallback
            left_text = [('default', self.command)]
            right_text = []
            logger.debug(f"Error in _build_text_content: {e}")
        
        # Create a columns layout with command on left, stats on right
        columns = urwid.Columns([
            ('weight', 1, urwid.Text(left_text, wrap='clip')),
            ('pack', urwid.Text(right_text, align='right'))
        ])
        
        # No background highlight - only arrow indicator shows selection
        self._widget = urwid.AttrMap(columns, None, focus_map=None)
        
    def selectable(self):
        """Make this widget selectable for navigation."""
        return True
        
    def keypress(self, size, key):
        """Handle key presses - return key to parent for handling."""
        return key
    
    def mouse_event(self, size, event, button, col, row, focus):
        """Handle mouse events safely."""
        try:
            # Handle mouse click as selection
            if event == 'mouse press' and button == 1:  # Left click
                # Don't return 'enter' - let the parent handle it
                # Just indicate that the event was handled
                return True
            return False
        except Exception as e:
            logger.debug(f"Error in SearchResult.mouse_event: {e}")
            return False
        

    def _build_text_content(self, focus=False, max_width=None):
        """Build the text content for this search result."""
        # Left side: indicator + command
        left_text = []
        if focus:
            left_text.append(('light cyan', "▶ "))
        else:
            left_text.append(('dark gray', "  "))
        
        # Use default color for command text
        left_text.append(('default', self.command))
        
        # Right side: match statistics
        right_text = []
        if self.show_scores and self.score > 0:
            semantic_pct = int(self.semantic_score * 100)
            bm25_pct = int(self.bm25_score * 100)

            # Determine which score is dominant
            if bm25_pct > semantic_pct * 1.2:
                # Keyword dominant
                right_text.extend([
                    ('dark gray', "Key: "),
                    ('light green', f"{bm25_pct}%")
                ])
            elif semantic_pct > bm25_pct * 1.2:
                # Semantic dominant  
                right_text.extend([
                    ('dark gray', "Sem: "),
                    ('light blue', f"{semantic_pct}%")
                ])
            else:
                # Hybrid - show both
                right_text.extend([
                    ('dark gray', "Sem: "),
                    ('light blue', f"{semantic_pct}% "),
                    ('dark gray', "Key: "),
                    ('light green', f"{bm25_pct}%")
                ])
        
        return left_text, right_text

    def render(self, size, focus=False):
        """Render with focus handling through WidgetWrap."""
        try:
            # Update widget content based on focus state
            if focus != getattr(self, '_last_focus_state', False):
                logger.debug(f"SearchResult render: focus changed to {focus} for '{self.command[:20]}...'")
                self._create_widget(focus)
                self._last_focus_state = focus
                super().__init__(self._widget)  # Update the wrapped widget
            
            return super().render(size, focus)
        except Exception as e:
            logger.debug(f"Error in SearchResult.render: {e}")
            # Return a fallback simple text widget
            fallback = urwid.Text(f"Error: {self.command}")
            return fallback.render(size, focus)

    def selectable(self):
        return True

    def keypress(self, size, key):
        return key

class LoadingIndicator(urwid.Text):
    """Shows loading state with animated spinner and progress information."""
    
    def __init__(self):
        super().__init__("")
        self.is_loading = False
        self.message = "Searching..."
        self._spinner_alarm = None
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_index = 0
        self._start_time = 0
        self._elapsed_time = 0
    
    def start_loading(self, loop, message="Searching..."):
        """Start the loading animation."""
        self.is_loading = True
        self.message = message
        self._spinner_index = 0
        self._start_time = time.time()
        self._elapsed_time = 0
        self._update_spinner()
        # Update spinner every 0.1 seconds
        self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
    
    def stop_loading(self, loop):
        """Stop the loading animation."""
        self.is_loading = False
        if self._spinner_alarm:
            loop.remove_alarm(self._spinner_alarm)
            self._spinner_alarm = None
        self.set_text("")
    
    def _spinner_callback(self, loop, user_data):
        """Update spinner animation."""
        if self.is_loading:
            self._elapsed_time = time.time() - self._start_time
            self._update_spinner()
            self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
    
    def _update_spinner(self):
        """Update the spinner display with elapsed time."""
        if self.is_loading:
            spinner = self._spinner_chars[self._spinner_index]
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_chars)
            
            # Show elapsed time for longer searches
            if self._elapsed_time > 1.0:
                time_text = f" ({self._elapsed_time:.1f}s)"
                message_color = 'yellow' if self._elapsed_time > 3.0 else 'dark gray'
            else:
                time_text = ""
                message_color = 'dark gray'
                
            text = [
                ('dark cyan', f"{spinner} "), 
                (message_color, self.message + time_text)
            ]
            self.set_text(text)

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
        # Cancel any pending search
        if self._pending_search:
            self.loop.remove_alarm(self._pending_search)
        
        # Cancel any currently running search
        self._current_search_cancelled = True
        
        if not new_text.strip():
            self._clear_results()
            return
        
        # Set up new search
        self._current_search_cancelled = False
        self._current_query = new_text.strip()
        self._pending_search = self.loop.set_alarm_in(0.05, self._perform_search_callback, new_text.strip())

    def _perform_search_callback(self, loop, query):
        """Callback for the alarm - proper urwid callback signature."""
        self._pending_search = None
        self._perform_search(query)
    
    def _perform_search(self, query):
        """Perform the actual search operation."""
        if not query.strip():
            return
        
        # Check if this search was already cancelled before we even started
        if self._current_search_cancelled or query != self._current_query:
            return
            
        # Show loading indicator (only if loop is available)
        if hasattr(self, 'loop') and self.loop:
            # Check if model is likely to be ready
            if hasattr(self.fuzzyshell, '_model') and self.fuzzyshell._model is not None:
                message = f"🔍 Initializing semantic search for '{query}'..."
            else:
                message = f"⚠️ Model not ready for '{query}'..."
            self.loading_indicator.start_loading(self.loop, message)
            self.loop.draw_screen()
            
        start_time = time.time()
        
        # Update search status
        self.search_status.set_text("Searching...")
        if hasattr(self, 'loop') and self.loop:
            self.loop.draw_screen()
        
        try:
            # The progress callback will handle all progress updates now
            # No more timeout-based messages that override the real progress
            
            # Create progress callback for real-time updates
            def progress_callback(current, total, stage, partial_results):
                # Check if search was cancelled
                if self._current_search_cancelled or query != self._current_query:
                    return  # Don't update UI for cancelled searches
                
                if hasattr(self, 'loop') and self.loop:
                    # Calculate percentage and elapsed time
                    percent = int((current / total) * 100) if total > 0 else 0
                    elapsed = time.time() - start_time
                    
                    # Update search status with elapsed time if > 0.5s
                    if elapsed > 0.5:
                        self.search_status.set_text(f"Searching ({elapsed:.0f}s)")
                    else:
                        self.search_status.set_text("Searching...")
                    
                    # Update loading message with progress
                    progress_msg = f"🔍 {stage} ({current:,}/{total:,} records, {percent}%)"
                    
                    # Update loading indicator
                    self.loading_indicator.stop_loading(self.loop)
                    self.loading_indicator.start_loading(self.loop, progress_msg)
                    
                    # Show partial results if available and processing is complete
                    if partial_results and len(partial_results) > 0 and current >= total and stage == "Complete":
                        # Only update results when processing is complete and search not cancelled
                        if not self._current_search_cancelled and query == self._current_query:
                            self._display_results(partial_results)
                            logger.debug(f"Progress callback updated results for '{query}' ({len(partial_results)} results)")
                    
                    # Force screen update
                    self.loop.draw_screen()
            
            # Call search with progress callback
            if hasattr(self.fuzzyshell, 'search'):
                results = self.fuzzyshell.search(query, top_k=100, return_scores=True, progress_callback=progress_callback)
            else:
                results = self.search_callback(query)
            search_time = time.time() - start_time
            
            # Validate that results is a list/tuple
            if results is None:
                logger.warning(f"Search callback returned None for query: '{query}'")
                results = []
            elif not isinstance(results, (list, tuple)):
                logger.error(f"Search callback returned invalid type: {type(results)}, expected list/tuple")
                results = []
            elif len(results) == 0:
                logger.debug(f"Search callback returned empty results for query: '{query}'")
            
            logger.debug(f"Search for '{query}' returned {len(results)} results in {search_time:.3f}s")
            logger.debug(f"Raw results type: {type(results)}, first few: {results[:3] if results else 'None'}")
            
            # Final check: only update results if search wasn't cancelled
            if not self._current_search_cancelled and query == self._current_query:
                self.footer.update(search_time=search_time)
                self.current_results = results
                self._display_results(results)
                # Update search status to show completion
                if len(results) > 0:
                    self.search_status.set_text(f"Found {len(results)}")
                else:
                    self.search_status.set_text("No results")
            else:
                logger.debug(f"Search for '{query}' was cancelled, not updating results")
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search error after {search_time:.3f}s: {e}", exc_info=True)
            self.footer.update(search_time=search_time)
            self.results_list.clear()
            error_msg = f"Search error: {str(e)}"
            if "timeout" in str(e).lower() or search_time > 25.0:
                error_msg = f"Model initialization timed out ({search_time:.1f}s). Try again in a moment."
            elif search_time > 10.0:
                error_msg = f"Search took {search_time:.1f}s - model may still be loading"
            self.results_list.append(urwid.Text(error_msg, align='center'))
            self.current_results = []
        finally:
            # Hide loading indicator (only if loop is available)
            if hasattr(self, 'loop') and self.loop:
                self.loading_indicator.stop_loading(self.loop)
        
        if hasattr(self, 'loop') and self.loop:
            self.loop.draw_screen()

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
        
        # Special debug mode - press F1 to enable key logging
        if key == 'f1':
            logger.info("Debug mode: Press any key to see its representation, ESC to continue")
            return
        
        # Global shortcuts that work regardless of focus
        if key in ('q', 'Q', 'escape'):
            raise urwid.ExitMainLoop()
        
        elif key == 'enter':
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

        elif key in ('ctrl s', '\x13', 'ctrl S', '\x1f', 's', 'S'):  # Multiple ways to toggle scores
            # Toggle score display
            logger.debug(f"unhandled_input: Got score toggle key '{key}', toggling scores from {self.show_scores}")
            self.show_scores = not self.show_scores
            self.footer.update(show_scores=self.show_scores)
            self._display_results(self.current_results)
            logger.debug(f"unhandled_input: Scores now {self.show_scores}")
            return
        
        elif key == 'ctrl r':
            # Refresh search
            if self.input_box.edit_text.strip():
                self._perform_search(self.input_box.edit_text.strip())
            return
        
        # Handle navigation keys for results - let urwid handle them naturally
        elif key in ('up', 'down', 'page up', 'page down', 'home', 'end'):
            if self.results_list and len(self.results_list) > 0:
                # Navigate to results if not already focused
                self._focus_results()
                # Let urwid handle the navigation naturally - focus change observer will handle description
                logger.debug(f"🔑 Navigation key '{key}' - letting urwid handle naturally")
            return
        
        # For all other keys, let the focused widget handle them
        self.main_layout.keypress((), key)
    
    
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
            
            # Focus styles - subtle highlighting
            ('focus_item', 'default', 'dark gray'),
            
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
