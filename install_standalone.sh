#!/bin/bash
# FuzzyShell Standalone Installer - v1.0
# Downloads and installs FuzzyShell with its own venv
# Usage: curl -sSL https://raw.githubusercontent.com/mitchins/fuzzyshell/main/install_standalone.sh | bash

set -e

# Configuration
FUZZYSHELL_HOME="${FUZZYSHELL_HOME:-$HOME/.fuzzyshell}"
FUZZYSHELL_VERSION="${FUZZYSHELL_VERSION:-latest}"
GITHUB_REPO="mitchins/fuzzyshell"  # Replace with actual repo
INSTALL_DIR="$FUZZYSHELL_HOME/app"
VENV_DIR="$FUZZYSHELL_HOME/venv"
DATA_DIR="$HOME/.local/share/fuzzyshell"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Detect shell
detect_shell() {
    USER_SHELL=$(basename "$SHELL")
    
    if [ "$USER_SHELL" = "zsh" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
        SHELL_NAME="zsh"
    elif [ "$USER_SHELL" = "bash" ]; then
        if [ "$OS" = "macos" ]; then
            # macOS uses .bash_profile for login shells
            SHELL_CONFIG="$HOME/.bash_profile"
        else
            SHELL_CONFIG="$HOME/.bashrc"
        fi
        SHELL_NAME="bash"
    else
        log_error "Unsupported shell: $USER_SHELL"
        log_info "Currently only bash and zsh are supported"
        exit 1
    fi
    
    log_info "Detected shell: $SHELL_NAME ($SHELL_CONFIG)"
}

# Find best Python version
find_python() {
    local python_cmd=""
    local python_version=""
    
    # Check for Python 3.10-3.12 (optimal versions)
    for version in python3.12 python3.11 python3.10; do
        if command -v $version >/dev/null 2>&1; then
            python_cmd=$version
            break
        fi
    done
    
    # Fallback to python3
    if [ -z "$python_cmd" ] && command -v python3 >/dev/null 2>&1; then
        python_cmd="python3"
    fi
    
    # Check if we found Python
    if [ -z "$python_cmd" ]; then
        log_error "Python 3 not found"
        log_info "Please install Python 3.10 or later"
        exit 1
    fi
    
    # Verify version is 3.9+
    if ! $python_cmd -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then
        python_version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        log_error "Python $python_version is too old"
        log_info "FuzzyShell requires Python 3.9 or later"
        exit 1
    fi
    
    # Warn about Python 3.13
    if [[ "$python_version" == "3.13" ]]; then
        log_warning "Python 3.13 detected - some dependencies may have compatibility issues"
        log_info "Consider using Python 3.10-3.12 for best compatibility"
    fi
    
    PYTHON_CMD=$python_cmd
    log_success "Found Python $python_version at $PYTHON_CMD"
}

# Check if already installed
check_existing() {
    if [ -d "$FUZZYSHELL_HOME" ] && [ -f "$VENV_DIR/bin/fuzzy" ]; then
        log_warning "FuzzyShell appears to be already installed at $FUZZYSHELL_HOME"
        read -p "Do you want to reinstall/update? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled"
            exit 0
        fi
        log_info "Proceeding with reinstallation..."
    fi
}

# Download and extract release
download_fuzzyshell() {
    log_info "Downloading FuzzyShell..."
    
    # Create directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$DATA_DIR"
    
    # Download latest release tarball
    local download_url="https://github.com/$GITHUB_REPO/archive/refs/heads/main.tar.gz"
    if [ "$FUZZYSHELL_VERSION" != "latest" ]; then
        download_url="https://github.com/$GITHUB_REPO/archive/refs/tags/v${FUZZYSHELL_VERSION}.tar.gz"
    fi
    
    # Download and extract
    if command -v curl >/dev/null 2>&1; then
        curl -sSL "$download_url" | tar xz -C "$INSTALL_DIR" --strip-components=1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$download_url" | tar xz -C "$INSTALL_DIR" --strip-components=1
    else
        log_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    log_success "Downloaded FuzzyShell to $INSTALL_DIR"
}

# Create virtual environment and install
install_fuzzyshell() {
    log_info "Creating virtual environment..."
    
    # Create venv
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    
    # Upgrade pip
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    
    log_info "Installing FuzzyShell and dependencies..."
    
    # Install from local directory
    if ! "$VENV_DIR/bin/pip" install --quiet -e "$INSTALL_DIR"; then
        log_error "Failed to install FuzzyShell"
        exit 1
    fi
    
    # Verify installation
    if [ ! -f "$VENV_DIR/bin/fuzzy" ]; then
        log_error "Installation verification failed - fuzzy command not found"
        exit 1
    fi
    
    log_success "FuzzyShell installed successfully"
}

# Setup shell integration
setup_shell_integration() {
    log_info "Setting up shell integration..."
    
    # Check if already configured
    if grep -q "fuzzyshell/shell_integration.sh" "$SHELL_CONFIG" 2>/dev/null; then
        log_warning "Shell integration already configured in $SHELL_CONFIG"
        return
    fi
    
    # Create shell integration script
    cat > "$FUZZYSHELL_HOME/shell_integration.sh" << 'EOF'
#!/bin/bash
# FuzzyShell Shell Integration
# Auto-generated by installer - do not edit

# FuzzyShell paths
export FUZZYSHELL_HOME="$HOME/.fuzzyshell"
export FUZZYSHELL_CMD="$FUZZYSHELL_HOME/venv/bin/fuzzy"

# Ensure fuzzy command is available
if [ ! -f "$FUZZYSHELL_CMD" ]; then
    echo -e "⚠️  FuzzyShell not found at $FUZZYSHELL_CMD"
    echo -e "   Run: curl -sSL https://raw.githubusercontent.com/mitchins/fuzzyshell/main/install_standalone.sh | bash"
    return 1
fi

# ZSH integration
if [ -n "$ZSH_VERSION" ]; then
    # Ctrl+F handler for ZSH
    fuzzy_search_widget() {
        local selected_cmd
        local current_buffer="$BUFFER"
        local current_cursor="$CURSOR"
        
        # Run fuzzy and capture output
        selected_cmd=$("$FUZZYSHELL_CMD" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$selected_cmd" ]; then
            BUFFER="$selected_cmd"
            CURSOR=${#BUFFER}
        else
            BUFFER="$current_buffer"
            CURSOR=$current_cursor
        fi
        
        zle reset-prompt
    }
    
    # Register widget and bind to Ctrl+F
    zle -N fuzzy_search_widget
    bindkey '^F' fuzzy_search_widget
    
    # Command capture hook
    fuzzyshell_preexec() {
        if [ -n "$1" ]; then
            echo -e "$1" >> "$HOME/.fuzzyshell_recent_commands"
            # Keep only last 1000 commands
            if [ $(wc -l < "$HOME/.fuzzyshell_recent_commands" 2>/dev/null || echo 0) -gt 1000 ]; then
                tail -n 1000 "$HOME/.fuzzyshell_recent_commands" > "$HOME/.fuzzyshell_recent_commands.tmp"
                mv "$HOME/.fuzzyshell_recent_commands.tmp" "$HOME/.fuzzyshell_recent_commands"
            fi
        fi
    }
    
    # Add to preexec hooks
    if [[ ! " ${preexec_functions[@]} " =~ " fuzzyshell_preexec " ]]; then
        preexec_functions+=(fuzzyshell_preexec)
    fi
    
# BASH integration
elif [ -n "$BASH_VERSION" ]; then
    # Ctrl+F handler for Bash
    fuzzy_search_widget() {
        local selected_cmd
        selected_cmd=$("$FUZZYSHELL_CMD" 2>/dev/null </dev/tty)
        
        if [ $? -eq 0 ] && [ -n "$selected_cmd" ]; then
            READLINE_LINE="$selected_cmd"
            READLINE_POINT=${#READLINE_LINE}
        fi
    }
    
    # Bind to Ctrl+F
    bind -x '"\C-f": fuzzy_search_widget'
    
    # Command capture using DEBUG trap
    fuzzyshell_capture() {
        if [ -n "$BASH_COMMAND" ] && [[ ! "$BASH_COMMAND" =~ ^[[:space:]] ]]; then
            echo -e "$BASH_COMMAND" >> "$HOME/.fuzzyshell_recent_commands"
            # Keep only last 1000 commands
            if [ $(wc -l < "$HOME/.fuzzyshell_recent_commands" 2>/dev/null || echo 0) -gt 1000 ]; then
                tail -n 1000 "$HOME/.fuzzyshell_recent_commands" > "$HOME/.fuzzyshell_recent_commands.tmp"
                mv "$HOME/.fuzzyshell_recent_commands.tmp" "$HOME/.fuzzyshell_recent_commands"
            fi
        fi
    }
    
    # Set trap if not already set
    if [[ -z "$(trap -p DEBUG)" ]]; then
        trap 'fuzzyshell_capture' DEBUG
    fi
fi

# Convenience aliases
alias fuzzy="$FUZZYSHELL_CMD"
alias fuzzy-ingest="$FUZZYSHELL_CMD --ingest"
alias fuzzy-rebuild="$FUZZYSHELL_CMD --rebuild-ann"
alias fuzzy-update="curl -sSL https://raw.githubusercontent.com/mitchins/fuzzyshell/main/install_standalone.sh | bash"

# Silent check on startup (only show if verbose mode)
if [ -n "$FUZZYSHELL_VERBOSE" ]; then
    echo -e "✓ FuzzyShell ready (Ctrl+F to search)"
fi
EOF
    
    # Make integration script executable
    chmod +x "$FUZZYSHELL_HOME/shell_integration.sh"
    
    # Add to shell config
    echo -e "" >> "$SHELL_CONFIG"
    echo -e "# FuzzyShell - Semantic command search (installed $(date +%Y-%m-%d))" >> "$SHELL_CONFIG"
    echo -e "[ -f \"$FUZZYSHELL_HOME/shell_integration.sh\" ] && source \"$FUZZYSHELL_HOME/shell_integration.sh\"" >> "$SHELL_CONFIG"
    
    log_success "Added FuzzyShell to $SHELL_CONFIG"
}

# Initial setup and ingestion
initial_setup() {
    log_info "Running initial setup..."
    
    # Check if database already exists
    if [ -f "$DATA_DIR/fuzzyshell.db" ]; then
        log_info "Existing database found, skipping initial ingestion"
        return
    fi
    
    log_info "Ingesting command history (this may take a moment)..."
    
    # Run ingestion
    if "$VENV_DIR/bin/fuzzy" --ingest >/dev/null 2>&1; then
        log_success "Command history ingested successfully"
    else
        log_warning "Initial ingestion failed - you can run 'fuzzy-ingest' manually later"
    fi
}

# Print success message
print_success() {
    echo
    echo -e "════════════════════════════════════════════════════════════════"
    echo
    log_success "FuzzyShell v1.0 installed successfully!"
    echo
    echo -e "📍 Installation location: $FUZZYSHELL_HOME"
    echo -e "📂 Data location: $DATA_DIR"
    echo
    echo -e "🚀 To get started:"
    echo
    echo -e "   1. Reload your shell configuration:"
    echo -e "      ${GREEN}source $SHELL_CONFIG${NC}"
    echo
    echo -e "   2. Use Ctrl+F to search your command history!"
    echo
    echo -e "📝 Available commands:"
    echo -e "   • ${BLUE}fuzzy${NC}          - Open search interface"
    echo -e "   • ${BLUE}fuzzy-ingest${NC}   - Re-ingest command history"
    echo -e "   • ${BLUE}fuzzy-rebuild${NC}  - Rebuild search index"
    echo -e "   • ${BLUE}fuzzy-update${NC}   - Update FuzzyShell"
    echo
    echo -e "💡 Tips:"
    echo -e "   • Ctrl+F works like Ctrl+R but with semantic search"
    echo -e "   • Type naturally: 'list files' finds 'ls -la'"
    echo -e "   • Commands are auto-captured for future searches"
    echo
    echo -e "════════════════════════════════════════════════════════════════"
    echo
}

# Main installation flow
main() {
    echo
    echo -e "═══════════════════════════════════════════════════════════════"
    echo -e "                    FuzzyShell Installer v1.0                  "
    echo -e "═══════════════════════════════════════════════════════════════"
    echo
    
    # Run installation steps
    detect_os
    detect_shell
    find_python
    check_existing
    download_fuzzyshell
    install_fuzzyshell
    setup_shell_integration
    initial_setup
    print_success
}

# Run main function
main "$@"