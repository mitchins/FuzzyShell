#!/bin/bash
# FuzzyShell installer - sets up everything automatically

set -e

echo "🚀 Installing FuzzyShell..."

# Detect user's actual shell (not the script's shell)
USER_SHELL=$(basename "$SHELL")

if [ "$USER_SHELL" = "zsh" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [ "$USER_SHELL" = "bash" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    # Fallback: try to detect from current process
    if [ -f "$HOME/.zshrc" ] && ps -p $$ | grep -q zsh; then
        SHELL_CONFIG="$HOME/.zshrc"
        SHELL_NAME="zsh"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
        SHELL_NAME="bash"
    else
        echo "❌ Could not detect shell. Please manually add to your .zshrc or .bashrc:"
        echo "   source $SCRIPT_DIR/fuzzyshell_setup.sh"
        exit 1
    fi
fi

echo "🔍 Detected shell: $SHELL_NAME"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if already installed
if grep -q "fuzzyshell_setup.sh" "$SHELL_CONFIG" 2>/dev/null; then
    echo "⚠️  FuzzyShell already appears to be installed in your $SHELL_NAME config."
    echo "   To reinstall, remove the fuzzyshell line from $SHELL_CONFIG first."
    exit 1
fi

# Add source line to shell config
echo "" >> "$SHELL_CONFIG"
echo "# FuzzyShell - Semantic command search (Ctrl+F)" >> "$SHELL_CONFIG"
echo "source $SCRIPT_DIR/fuzzyshell_setup.sh" >> "$SHELL_CONFIG"

echo "✅ Added FuzzyShell to $SHELL_CONFIG"
echo ""
echo "🎯 Next steps:"
echo "   1. Reload your shell: source $SHELL_CONFIG"
echo "   2. Press Ctrl+F to search your command history!"
echo ""
echo "💡 Tips:"
echo "   - First time setup will automatically create a venv and ingest your history"
echo "   - Use 'fuzzy_update' to update FuzzyShell anytime"
echo "   - Use 'fuzzy_rebuild_ann' if search performance degrades"