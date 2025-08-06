"""
Expert screen for FuzzyShell TUI.
Shows technical specifications and advanced configuration options.
"""

import urwid
from ..key_bindings import KEY_EXPERT_SCREEN, KEY_QUIT

# Import version at module level to avoid circular imports
try:
    from fuzzyshell import __version__
except ImportError:
    # Fallback for development/circular import issues
    __version__ = "1.0.0"

# Import constants from fuzzyshell
try:
    from ...fuzzyshell import USE_ANN_SEARCH, ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES, EMBEDDING_DTYPE, MODEL_OUTPUT_DIM
    # Convert numpy dtype to string for display
    EMBEDDING_DTYPE_STR = EMBEDDING_DTYPE.__name__
except ImportError:
    # Fallback values for development
    USE_ANN_SEARCH = True
    ANN_NUM_CLUSTERS = 32
    ANN_CLUSTER_CANDIDATES = 4
    EMBEDDING_DTYPE_STR = "float32"
    MODEL_OUTPUT_DIM = 384


class ExpertScreen(urwid.WidgetWrap):
    """Expert screen showing technical specifications and advanced configuration."""
    
    def __init__(self, fuzzyshell_instance=None):
        self.fuzzyshell = fuzzyshell_instance
        self.scoring_radio_group = []  # Radio button group for scoring preferences
        
        # Create the main content with radio buttons
        main_content = self._build_main_content()
        
        # Create a box with border and title
        box = urwid.LineBox(main_content, title='FuzzyShell - System Information')
        
        # Add instructions at the bottom
        expert_key_display = "^E" if KEY_EXPERT_SCREEN == 'ctrl e' else KEY_EXPERT_SCREEN.upper()
        instructions = urwid.Text([
            ('bold', expert_key_display), " or ", ('bold', "ESC"), " to close  |  ",
            ('bold', "↑↓"), " navigate  |  ", ('bold', "SPACE"), " select scoring mode"
        ], align='center')
        
        # Combine everything
        pile = urwid.Pile([
            ('weight', 1, box),
            ('pack', urwid.Divider()),
            ('pack', instructions)
        ])
        
        # Style with attribute map
        self.attr_map = urwid.AttrMap(pile, 'expert_screen')
        super().__init__(self.attr_map)
    
    def _build_main_content(self):
        """Build the main content with technical specs and scoring controls."""
        # Get dynamic info from fuzzyshell instance if available
        item_count = "unknown"
        embedding_model = "unknown"
        last_build = "unknown"
        db_uuid = "unknown"
        db_location = "unknown"
        db_created = "unknown"
        current_scoring_preference = "balanced"
        
        if self.fuzzyshell:
            try:
                # Get comprehensive expert screen data from MetadataManager
                expert_data = self.fuzzyshell.metadata_manager.get_expert_screen_data()
                
                # Extract information for display
                db_info = expert_data.get('database', {})
                item_count = f"{db_info.get('item_count', 0):,}"
                embedding_model = db_info.get('embedding_model', 'unknown')
                last_build = db_info.get('last_updated', 'unknown')
                db_uuid = db_info.get('uuid', 'unknown')
                db_created = db_info.get('created', 'unknown')
                db_location = db_info.get('db_path', '~/.fuzzyshell/fuzzyshell.db')
                
                # Get current scoring preference
                scoring_config = expert_data.get('scoring', {})
                current_scoring_preference = scoring_config.get('preference', 'balanced')
            except Exception as e:
                # Fallback to direct calls if MetadataManager fails
                try:
                    db_info = self.fuzzyshell.get_database_info()
                    item_count = f"{db_info.get('item_count', 0):,}"
                    embedding_model = db_info.get('embedding_model', 'unknown')
                    last_build = db_info.get('last_updated', 'unknown')
                    current_scoring_preference = self.fuzzyshell.metadata_dal.get_scoring_preference()
                    db_location = getattr(self.fuzzyshell, 'db_path', '~/.fuzzyshell/fuzzyshell.db')
                except:
                    pass
        
        # Create technical specs text widgets
        spec_lines = [
            f"Embedding Model: {embedding_model}",
            "Description Model: codet5-small-terminal-describer",
            "",
            f"Database Location: {db_location}",
            f"Database UUID: {db_uuid}",
            f"Database Created: {db_created}",
            "",
            f"Database Items: {item_count} commands indexed",
            f"Last ANN Build: {last_build}",
            "",
            f"Clusters: {ANN_NUM_CLUSTERS} k-means clusters, {ANN_CLUSTER_CANDIDATES} search candidates",
            f"Storage: {EMBEDDING_DTYPE_STR} embeddings in fuzzyshell.db",
            "",
            "Model Path: ~/.fuzzyshell/model/ ~/.fuzzyshell/description_model/",
            "",
        ]
        
        spec_widgets = []
        for line in spec_lines:
            if line == "":
                spec_widgets.append(urwid.Divider())
            else:
                spec_widgets.append(urwid.Text(line))
        
        # Create scoring preference radio buttons
        scoring_title = urwid.Text([('bold', 'Search Scoring Mode:')], align='left')
        spec_widgets.append(scoring_title)
        spec_widgets.append(urwid.Divider())
        
        # Define scoring options with descriptions
        scoring_options = [
            ('less_semantic', 'Less Semantic (30% semantic, 70% keyword)'),
            ('balanced', 'Balanced (50% semantic, 50% keyword)'),
            ('more_semantic', 'More Semantic (70% semantic, 30% keyword)'),
            ('semantic_only', 'Semantic Only (100% semantic, 0% keyword)')
        ]
        
        # Create radio buttons for scoring preferences
        for option_key, option_label in scoring_options:
            is_current = (option_key == current_scoring_preference)
            radio_btn = urwid.RadioButton(
                self.scoring_radio_group, 
                option_label, 
                state=is_current,
                on_state_change=self._on_scoring_change,
                user_data=option_key
            )
            spec_widgets.append(urwid.Padding(radio_btn, left=2))
        
        # Add all widgets to a padded list
        padded_widgets = [urwid.Padding(widget, left=2, right=2) for widget in spec_widgets]
        
        # Create scrollable list
        listbox = urwid.ListBox(urwid.SimpleFocusListWalker(padded_widgets))
        
        return listbox
    
    def _on_scoring_change(self, _radio_button, new_state, user_data):
        """Handle scoring preference radio button changes."""
        if new_state and self.fuzzyshell and user_data:
            try:
                # Use MetadataManager for better error handling
                self.fuzzyshell.metadata_manager.set_scoring_preference(user_data)
            except Exception as e:
                # Silently ignore preference update errors for now
                pass
    
    def keypress(self, size, key):
        """Handle key presses for the expert screen."""
        if key in (KEY_QUIT, 'q', KEY_EXPERT_SCREEN):
            return 'close_expert_screen'
        
        # Let the parent handle scrolling and radio button navigation
        return super().keypress(size, key)


# Define color palette for the expert screen
EXPERT_SCREEN_PALETTE = [
    ('title', 'light cyan,bold', 'default'),
    ('section_header', 'light green,bold', 'default'),
    ('expert_screen', 'default', 'default'),
]