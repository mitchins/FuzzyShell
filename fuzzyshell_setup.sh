#!/bin/bash
# FuzzyShell setup script for bash/zsh with automatic venv management
# This script should be sourced in your shell configuration file

# FuzzyShell home directory
FUZZYSHELL_HOME="$HOME/.fuzzyshell"
FUZZYSHELL_VENV="$FUZZYSHELL_HOME/venv"
FUZZYSHELL_DB="$HOME/.local/share/fuzzyshell/fuzzyshell.db"

# Ensure FuzzyShell is installed in its own venv
_ensure_fuzzyshell_installed() {
    # Create directories if needed
    mkdir -p "$FUZZYSHELL_HOME"
    mkdir -p "$(dirname "$FUZZYSHELL_DB")"
    
    # Check if venv exists and has fuzzy executable
    if [ ! -d "$FUZZYSHELL_VENV" ] || [ ! -f "$FUZZYSHELL_VENV/bin/fuzzy" ]; then
        echo "🔧 Setting up FuzzyShell environment..."
        
        # Find best Python executable with smart detection
        PYTHON_CMD=""
        PYTHON_SOURCE=""
        
        # Priority order: try to find the best available Python
        # 1. If user has fuzzyshell conda env available, prefer that
        # 2. Otherwise use any stable system Python
        # 3. Warn about Python 3.13 compatibility issues
        
        if [ -f "/opt/miniconda3/envs/fuzzyshell/bin/python3" ]; then
            # Use fuzzyshell conda environment if available (non-intrusive)
            PYTHON_CMD="/opt/miniconda3/envs/fuzzyshell/bin/python3"
            PYTHON_SOURCE="conda env (fuzzyshell)"
        elif [ -n "$CONDA_DEFAULT_ENV" ] && command -v python3 >/dev/null 2>&1; then
            # Use currently active conda environment
            PYTHON_CMD="python3"
            PYTHON_SOURCE="conda ($CONDA_DEFAULT_ENV)"
        elif [ -n "$CONDA_DEFAULT_ENV" ] && command -v python >/dev/null 2>&1; then
            # Fallback to python if python3 not available in conda
            PYTHON_CMD="python"
            PYTHON_SOURCE="conda ($CONDA_DEFAULT_ENV)"
        elif command -v python3.11 >/dev/null 2>&1; then
            PYTHON_CMD="python3.11"
            PYTHON_SOURCE="system (3.11)"
        elif command -v python3.10 >/dev/null 2>&1; then
            PYTHON_CMD="python3.10"
            PYTHON_SOURCE="system (3.10)"
        elif [ -f "/opt/miniconda3/bin/python3" ]; then
            PYTHON_CMD="/opt/miniconda3/bin/python3"
            PYTHON_SOURCE="miniconda base"
        elif command -v python3 >/dev/null 2>&1; then
            PYTHON_VERSION_CHECK=$("python3" --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
            if [[ "$PYTHON_VERSION_CHECK" == "3.13" ]]; then
                echo "⚠️  Python 3.13 detected - some ML dependencies may have build issues"
                echo "💡 For best compatibility, consider creating a Python 3.10/3.11 environment:"
                echo "   conda create -n fuzzyshell python=3.10 && conda activate fuzzyshell"
            fi
            PYTHON_CMD="python3"
            PYTHON_SOURCE="system"
        elif command -v python >/dev/null 2>&1 && python --version 2>&1 | grep -q "Python 3"; then
            PYTHON_CMD="python"
            PYTHON_SOURCE="system"
        else
            echo "❌ No suitable Python 3 found."
            echo "💡 Please install Python 3.7+ or create a conda environment:"
            echo "   conda create -n fuzzyshell python=3.10"
            echo "   conda activate fuzzyshell && source ~/.zshrc"
            return 1
        fi
        
        # Verify Python version
        PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
        echo "🐍 Using Python $PYTHON_VERSION from $PYTHON_SOURCE"
        
        # Create venv if it doesn't exist
        if [ ! -d "$FUZZYSHELL_VENV" ]; then
            echo "📦 Creating virtual environment..."
            "$PYTHON_CMD" -m venv "$FUZZYSHELL_VENV"
        fi
        
        # Install/upgrade fuzzyshell
        echo "📥 Installing FuzzyShell..."
        "$FUZZYSHELL_VENV/bin/pip" install --upgrade pip >/dev/null 2>&1
        
        # Check if we're in development mode (fuzzyshell repo)
        if [ -f "$(dirname "${BASH_SOURCE[0]}")/pyproject.toml" ]; then
            # Development install
            echo "🔧 Installing from local development directory..."
            if ! "$FUZZYSHELL_VENV/bin/pip" install -e "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1; then
                echo "❌ Failed to install FuzzyShell in development mode"
                return 1
            fi
        else
            # Production install from PyPI
            if ! "$FUZZYSHELL_VENV/bin/pip" install fuzzyshell >/dev/null 2>&1; then
                echo "❌ Failed to install FuzzyShell from PyPI"
                return 1
            fi
        fi
        
        # Verify installation
        if [ ! -f "$FUZZYSHELL_VENV/bin/fuzzy" ]; then
            echo "❌ FuzzyShell installation failed - executable not found"
            return 1
        fi
        
        # Initial ingestion if database doesn't exist
        if [ ! -f "$FUZZYSHELL_DB" ]; then
            echo "🎉 First time setup - ingesting command history..."
            if ! "$FUZZYSHELL_VENV/bin/fuzzy" --ingest; then
                echo "⚠️  Initial ingestion failed, but you can run 'fuzzy_ingest' later"
            fi
        fi
        
        echo "✅ FuzzyShell setup complete!"
    fi
}

# Ensure installation on first source
_ensure_fuzzyshell_installed

# Set the command path
FUZZYSHELL_CMD="$FUZZYSHELL_VENV/bin/fuzzy"

# Function to capture command before execution (zsh)
if [ -n "$ZSH_VERSION" ]; then
    # ZSH-specific implementation
    
    # Function to handle fuzzy search with Ctrl+F
    fuzzy_search() {
        local selected_cmd
        
        # Save current buffer
        local current_buffer="$BUFFER"
        local current_cursor="$CURSOR"
        
        # Run fuzzyshell and capture output
        selected_cmd=$("$FUZZYSHELL_CMD" 2>/dev/null)
        
        # Check if a command was selected (exit code 0)
        if [ $? -eq 0 ] && [ -n "$selected_cmd" ]; then
            # Replace buffer with selected command
            BUFFER="$selected_cmd"
            CURSOR=${#BUFFER}
        else
            # Restore original buffer if cancelled
            BUFFER="$current_buffer"
            CURSOR=$current_cursor
        fi
        
        # Redraw the prompt
        zle reset-prompt
    }
    
    # Create ZLE widget
    zle -N fuzzy_search
    
    # Bind to Ctrl+F
    bindkey '^F' fuzzy_search
    
    # Optional: Also bind to Ctrl+R for those who prefer it
    # bindkey '^R' fuzzy_search
    
    # Hook to capture commands for micro-ingest
    fuzzyshell_preexec() {
        # Only process non-empty commands
        if [ -n "$1" ]; then
            # Append to a temporary file that micro-ingest can read
            echo "$1" >> "$HOME/.fuzzyshell_recent_commands"
            
            # Keep only last 1000 lines
            if [ $(wc -l < "$HOME/.fuzzyshell_recent_commands" 2>/dev/null || echo 0) -gt 1000 ]; then
                tail -n 1000 "$HOME/.fuzzyshell_recent_commands" > "$HOME/.fuzzyshell_recent_commands.tmp"
                mv "$HOME/.fuzzyshell_recent_commands.tmp" "$HOME/.fuzzyshell_recent_commands"
            fi
        fi
    }
    
    # Add to preexec hooks
    preexec_functions+=(fuzzyshell_preexec)
    
# Bash-specific implementation
elif [ -n "$BASH_VERSION" ]; then
    # Bash implementation using readline
    
    fuzzy_search() {
        local selected_cmd
        
        # Run fuzzyshell and capture output
        selected_cmd=$("$FUZZYSHELL_CMD" 2>/dev/null </dev/tty)
        
        # Check if a command was selected
        if [ $? -eq 0 ] && [ -n "$selected_cmd" ]; then
            # Insert command at current position
            READLINE_LINE="$selected_cmd"
            READLINE_POINT=${#READLINE_LINE}
        fi
    }
    
    # Bind to Ctrl+F
    bind -x '"\C-f": fuzzy_search'
    
    # Optional: Also bind to Ctrl+R
    # bind -x '"\C-r": fuzzy_search'
    
    # Capture commands using DEBUG trap
    fuzzyshell_capture_command() {
        # Skip if command is empty or starts with space (private)
        if [ -n "$BASH_COMMAND" ] && [[ ! "$BASH_COMMAND" =~ ^[[:space:]] ]]; then
            echo "$BASH_COMMAND" >> "$HOME/.fuzzyshell_recent_commands"
            
            # Keep only last 1000 lines
            if [ $(wc -l < "$HOME/.fuzzyshell_recent_commands" 2>/dev/null || echo 0) -gt 1000 ]; then
                tail -n 1000 "$HOME/.fuzzyshell_recent_commands" > "$HOME/.fuzzyshell_recent_commands.tmp"
                mv "$HOME/.fuzzyshell_recent_commands.tmp" "$HOME/.fuzzyshell_recent_commands"
            fi
        fi
    }
    
    # Set up trap (be careful not to override existing DEBUG traps)
    if [[ -z "$(trap -p DEBUG)" ]]; then
        trap 'fuzzyshell_capture_command' DEBUG
    else
        echo "Warning: DEBUG trap already set, command capture disabled"
    fi
fi

# Convenience functions
fuzzy_ingest() {
    "$FUZZYSHELL_CMD" --ingest
}

fuzzy_rebuild_ann() {
    "$FUZZYSHELL_CMD" --rebuild-ann
}

fuzzy_update() {
    echo "🔄 Updating FuzzyShell..."
    "$FUZZYSHELL_VENV/bin/pip" install --upgrade fuzzyshell
    echo "✅ Update complete!"
}

# Main fuzzy command alias
fuzzy() {
    "$FUZZYSHELL_CMD" "$@"
}

# Show setup status (only on first setup or if FUZZYSHELL_VERBOSE is set)
if [ -n "$FUZZYSHELL_VERBOSE" ] || [ ! -f "$HOME/.fuzzyshell/.setup_complete" ]; then
    echo "✅ FuzzyShell ready! Use Ctrl+F to search your command history."
    touch "$HOME/.fuzzyshell/.setup_complete"
fi