# FuzzyShell Installation & Usage Guide

## Quick Install (Recommended)

```bash
curl -sSL https://raw.githubusercontent.com/mitchins/fuzzyshell/main/install_standalone.sh | bash
```

This one command:
- ✅ Creates isolated venv in `~/.fuzzyshell/venv/`
- ✅ Installs all dependencies automatically  
- ✅ Sets up Ctrl+F key binding for instant search
- ✅ Configures automatic command capture
- ✅ Ingests your existing command history

After installation, reload your shell and press **Ctrl+F** to start searching!

## Alternative Installation Methods

### Manual pip install
```bash
pip install fuzzyshell
fuzzy --install-hook  # Sets up basic shell integration
```

### Development install
```bash
git clone https://github.com/mitchins/fuzzyshell
cd fuzzyshell
pip install -e ".[test]"
```

## Usage

### Basic Search
- **Press Ctrl+F** anywhere in your terminal to open search
- **Type your query** - supports both exact matching and semantic search
- **Navigate** with arrow keys or Tab/Shift+Tab  
- **Select** with Enter - command appears in your prompt ready to edit
- **Cancel** with Escape or Ctrl+C

### Search Examples
- `"list files"` → finds `ls -la`, `find . -type f`
- `"git uncommit"` → finds `git reset HEAD~1` 
- `"show containers"` → finds `docker ps`
- `"stage changes"` → finds `git add .`

### Available Commands
- `fuzzy` - Open search interface manually
- `fuzzy-ingest` - Re-scan command history
- `fuzzy-rebuild` - Rebuild search index for better performance
- `fuzzy-update` - Update to latest version

## How It Works

### Shell Integration
FuzzyShell uses shell-specific features for seamless command insertion:

- **Zsh**: Uses `print -z` to place commands in the command line buffer
- **Bash**: Uses `READLINE_LINE` to insert commands at cursor position

### Command Capture
Commands are automatically captured as you run them via shell hooks:
- **Zsh**: `preexec` hooks capture commands before execution
- **Bash**: `DEBUG` trap captures commands (if available)

### Search Technology
- **BM25**: Fast keyword matching for exact queries
- **Semantic Search**: ML embeddings understand command meaning
- **Hybrid Scoring**: Combines both approaches with recency weighting

## Troubleshooting

### Ctrl+F Not Working
1. Reload your shell: `source ~/.zshrc` or `source ~/.bashrc`
2. Check integration: `type fuzzy_search_widget` (should show function)
3. Try manual search: `fuzzy`

### Commands Not Being Captured
1. Check recent commands file: `ls -la ~/.fuzzyshell_recent_commands`
2. For bash: Ensure no conflicting DEBUG traps (`trap -p DEBUG`)
3. Run manual ingestion: `fuzzy-ingest`

### Search Results Poor
1. Rebuild search index: `fuzzy-rebuild`
2. Re-ingest history: `fuzzy-ingest`
3. Check database status: Run `fuzzy` and look for any error messages

### Performance Issues
1. Check database size: Look for `~/.local/share/fuzzyshell/fuzzyshell.db`
2. Rebuild ANN index: `fuzzy-rebuild`
3. Consider reducing shell history size if very large (50k+ commands)

## Advanced Configuration

### Custom Installation Location
```bash
FUZZYSHELL_HOME=~/my-fuzzy-location curl -sSL https://raw.githubusercontent.com/mitchins/fuzzyshell/main/install_standalone.sh | bash
```

### Using Different Python Version
The installer automatically finds the best Python 3.9+ version. To use a specific version:
```bash
# Ensure your preferred Python is first in PATH before running installer
export PATH="/usr/bin/python3.11:$PATH"
# Then run installer
```

### Manual Shell Integration

If you prefer manual setup, add this to your shell config:

#### Zsh (~/.zshrc)
```bash
fuzzy() {
    local selected_cmd
    selected_cmd=$("$HOME/.fuzzyshell/venv/bin/fuzzy" "$@")
    if [[ -n "$selected_cmd" ]]; then
        print -z "$selected_cmd"
    fi
}

# Optional: Bind to Ctrl+F
fuzzy_search_widget() {
    local selected_cmd
    selected_cmd=$("$HOME/.fuzzyshell/venv/bin/fuzzy" 2>/dev/null)
    if [[ -n "$selected_cmd" ]]; then
        BUFFER="$selected_cmd"
        CURSOR=${#BUFFER}
    fi
    zle reset-prompt
}
zle -N fuzzy_search_widget
bindkey '^F' fuzzy_search_widget
```

#### Bash (~/.bashrc)
```bash
fuzzy() {
    local selected_cmd
    selected_cmd=$("$HOME/.fuzzyshell/venv/bin/fuzzy" "$@")
    if [[ -n "$selected_cmd" ]]; then
        history -s "$selected_cmd"
        echo "Selected: $selected_cmd"
    fi
}

# Optional: Bind to Ctrl+F  
fuzzy_search_widget() {
    local selected_cmd
    selected_cmd=$("$HOME/.fuzzyshell/venv/bin/fuzzy" 2>/dev/null </dev/tty)
    if [[ -n "$selected_cmd" ]]; then
        READLINE_LINE="$selected_cmd"
        READLINE_POINT=${#READLINE_LINE}
    fi
}
bind -x '"\C-f": fuzzy_search_widget'
```

## Uninstall

```bash
# Remove FuzzyShell
rm -rf ~/.fuzzyshell

# Remove shell integration (edit your shell config file)
# Delete the lines that source fuzzyshell/shell_integration.sh
```