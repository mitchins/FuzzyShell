<p align="center">
  <picture>
    <source srcset="mascot_monogram_dark.png" media="(prefers-color-scheme: dark)">
    <img src="mascot_monogram_light.png" alt="FuzzyShell Mascot" width="96" height="96">
  </picture>
    <div>
  # FuzzyShell 

Lightning-fast semantic search for your command history. Find that command you ran last week with just a few keystrokes.
  </div>
</p>



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

### 🎯 **Smart Search**
- **Semantic understanding**: "list files" finds `ls -la`
- **Exact matching**: "git" prioritizes git commands
- **Prefix matching**: "doc" finds "docker run", "docker ps"

### ⚡ **Performance**
- Sub-100ms search on 10k+ commands
- Intelligent caching and quantized embeddings
- Background model loading

### 🔧 **Shell Integration**
- **Ctrl+F**: Search from anywhere in your terminal
- **Command population**: Selected commands appear in your prompt
- **Works like Ctrl+R**: Familiar workflow, better results

## How It Works

FuzzyShell combines two powerful search techniques:

1. **BM25 (Keyword Search)**: Fast exact matching for precise queries
2. **Semantic Search**: AI-powered understanding of command meaning

Commands are indexed with both approaches, then results are ranked using a hybrid scoring system that prioritizes exact matches while understanding context.

**Tech Stack**: Python 3.10+, SQLite with VSS extension, ONNX quantized embeddings

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