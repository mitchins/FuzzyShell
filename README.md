# FuzzyShell

<div style="display: flex; justify-content: center; align-items: center; gap: 24px; margin-bottom: 32px;">
  <picture>
    <source srcset="mascot_dark.png" media="(prefers-color-scheme: dark)">
    <img src="mascot_light.png" alt="FuzzyShell Mascot" width="90" height="90">
  </picture>
  <div style="font-size:1.24em; color:var(--fgColor-muted,#6e7781);">
    Lightning-fast semantic search for your command history.<br>
    Find that command you ran last week with just a few keystrokes.
  </div>
</div>



## Quick Start

### Install
```bash
pip install fuzzyshell
```

### Setup Shell Integration (Recommended)
```bash
# For Zsh users - add to ~/.zshrc:
echo 'source $(python -c "import fuzzyshell; print(fuzzyshell.__path__[0])")/fuzzyshell-wrapper.zsh' >> ~/.zshrc
source ~/.zshrc

# For Bash users - add to ~/.bashrc:
echo 'source $(python -c "import fuzzyshell; print(fuzzyshell.__path__[0])")/fuzzyshell-wrapper.bash' >> ~/.bashrc
source ~/.bashrc
```

### First Run
```bash
# Ingest your command history
fuzzy --ingest

# Start searching!
fuzzy
# OR use the keyboard shortcut: Ctrl+F
```

### Usage
- **Search**: Type to find commands (try "git", "docker", "python")
- **Navigate**: Arrow keys or Tab/Shift+Tab
- **Select**: Enter key - command appears in your shell prompt
- **Exit**: Escape or Ctrl+C

## Features

### 🎯 Smart Search
- **Semantic**: "list files" finds `ls -la`.
- **Exact**: "git" prioritizes git commands.
- **Prefix**: "doc" finds "docker run", "docker ps".

### ⚡ Performance
- Sub-100ms search on 10k+ commands.
- Fast loading with intelligent caching and quantized embeddings.

### 🔧 Shell Integration
- **Ctrl+F**: Search from anywhere.
- **Command Population**: Selected commands appear in your prompt.
- **Familiar Workflow**: Works like Ctrl+R, but better.

## How It Works

FuzzyShell uses a hybrid approach for command search:

1.  **Keyword Search (BM25)**: For fast, exact matches.
2.  **Semantic Search**: AI-powered understanding of command intent.

Commands are indexed using both methods, and results are ranked to prioritize exact matches while still understanding context.

**Tech Stack**: Python 3.10+, SQLite (with VSS), ONNX quantized embeddings.

## Development

### Setup Development Environment
```bash
git clone https://github.com/your-username/fuzzyshell
cd fuzzyshell
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

### Setup Shell Integration (Development)
For development, point directly to your working copy:

```bash
# For Zsh - add to ~/.zshrc:
echo "source $(pwd)/fuzzyshell-wrapper.zsh" >> ~/.zshrc
source ~/.zshrc

# For Bash - add to ~/.bashrc:
echo "source $(pwd)/fuzzyshell-wrapper.bash" >> ~/.bashrc
source ~/.bashrc

# Test the integration
fuzzy --ingest  # Initial setup
# Then try Ctrl+F in your terminal
```

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Test search quality
python tests/test_both_fixes.py

# Test specific functionality
python tests/test_integration.py
```

### Key Files
- `src/fuzzyshell/fuzzyshell.py` - Core search engine
- `src/fuzzyshell/fuzzy_tui.py` - Terminal UI
- `src/fuzzyshell/model_handler.py` - AI model handling
- `tests/` - Comprehensive test suite

### Contributing
1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**Made with ❤️ for developers who live in the terminal**

*FuzzyShell: Because `history | grep` isn't enough anymore.*