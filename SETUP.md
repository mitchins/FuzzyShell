# FuzzyShell Setup Guide

## Quick Setup

### 1. Install FuzzyShell
```bash
pip install fuzzyshell
```

### 2. Initial Ingestion
```bash
# Ingest your shell history
fuzzyshell --ingest
```

### 3. Shell Integration

#### For Zsh (recommended)
Add to your `~/.zshrc`:
```bash
# Source FuzzyShell setup
[ -f /path/to/fuzzyshell_setup.sh ] && source /path/to/fuzzyshell_setup.sh
```

#### For Bash
Add to your `~/.bashrc`:
```bash
# Source FuzzyShell setup
[ -f /path/to/fuzzyshell_setup.sh ] && source /path/to/fuzzyshell_setup.sh
```

### 4. Reload Shell
```bash
# For zsh
source ~/.zshrc

# For bash
source ~/.bashrc
```

## Usage

- **Search**: Press `Ctrl+F` in your terminal to search command history
- **Navigate**: Use arrow keys to browse results
- **Select**: Press Enter to insert command into your prompt
- **Cancel**: Press Esc to exit without selecting

## Features

### Automatic Command Capture
FuzzyShell automatically captures new commands as you execute them, so your search results stay up-to-date without manual ingestion.

### Micro-Ingestion
On startup, FuzzyShell quickly checks for new commands since the last update. This is fast and automatic.

### ANN Index Rebuilding
FuzzyShell will suggest when to rebuild the ANN (Approximate Nearest Neighbor) index for optimal performance:
```bash
# Manual rebuild when suggested
fuzzyshell --rebuild-ann
```

## Troubleshooting

### Commands Not Captured
If commands aren't being captured automatically:
1. Check that the shell integration is properly sourced
2. Verify `~/.fuzzyshell_recent_commands` is being written
3. For bash, ensure no conflicting DEBUG traps exist

### Search Not Working
1. Ensure initial ingestion was run: `fuzzyshell --ingest`
2. Check the database exists: `fuzzyshell --status`
3. Try clearing cache: `fuzzyshell --clear-cache`

### Performance Issues
1. Rebuild ANN index: `fuzzyshell --rebuild-ann`
2. Check database size: `fuzzyshell --info`
3. Consider adjusting history size in your shell config