"""Command-line interface commands and utilities for FuzzyShell."""

#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import logging

# Suppress tokenizers warnings
logging.getLogger("tokenizers").setLevel(logging.ERROR)

from . import __version__

logger = logging.getLogger('FuzzyShell')


def install_shell_hook():
    """Install shell integration hook for command insertion."""
    import os
    
    print("🔧 FuzzyShell Shell Integration Setup")
    print("=" * 40)
    print("⚠️  Note: For full functionality, consider using the complete installer:")
    print("   ./install.sh (sets up venv, Ctrl+F binding, command capture)")
    print()
    
    # Detect shell
    shell = os.environ.get('SHELL', '')
    if 'zsh' in shell:
        shell_name = 'zsh'
        rc_file = os.path.expanduser('~/.zshrc')
    elif 'bash' in shell:
        shell_name = 'bash'  
        rc_file = os.path.expanduser('~/.bashrc')
    else:
        print("⚠️  Could not detect shell. Supported shells: zsh, bash")
        return
        
    print(f"Detected shell: {shell_name}")
    print(f"Config file: {rc_file}")
    
    # Integration code for each shell
    if shell_name == 'zsh':
        integration_code = '''
# FuzzyShell integration
fuzzy() {
    local fuzzy_cmd="$HOME/.fuzzyshell/venv/bin/fuzzy"
    if [[ ! -f "$fuzzy_cmd" ]]; then
        echo "❌ FuzzyShell not installed. Please run the install.sh script first."
        return 1
    fi
    local selected_command
    selected_command=$("$fuzzy_cmd" "$@")
    if [[ -n "$selected_command" && "$selected_command" != "" ]]; then
        print -z "$selected_command"
    fi
}
'''
    else:  # bash
        integration_code = '''
# FuzzyShell integration  
fuzzy() {
    local fuzzy_cmd="$HOME/.fuzzyshell/venv/bin/fuzzy"
    if [[ ! -f "$fuzzy_cmd" ]]; then
        echo "❌ FuzzyShell not installed. Please run the install.sh script first."
        return 1
    fi
    local selected_command
    selected_command=$("$fuzzy_cmd" "$@")
    if [[ -n "$selected_command" && "$selected_command" != "" ]]; then
        history -s "$selected_command"
        echo "Selected: $selected_command"
    fi
}
'''
    
    # Check if already installed
    if os.path.exists(rc_file):
        with open(rc_file, 'r') as f:
            content = f.read()
        if '# FuzzyShell integration' in content:
            print("✅ FuzzyShell integration already exists in", rc_file)
            return
    
    # Add integration
    try:
        with open(rc_file, 'a') as f:
            f.write(integration_code)
        print("✅ Integration added to", rc_file)
        print()
        print("To activate the integration, either:")
        print("  1. Restart your terminal, or")
        print(f"  2. Run: source {rc_file}")
        print()
        print("Usage: fuzzy")
        print("The selected command will be inserted into your prompt!")
        print()
        print("💡 This basic integration requires manual venv setup.")
        print("   For automatic setup with Ctrl+F binding, use: ./install.sh")
    except Exception as e:
        print(f"❌ Failed to add integration: {e}")


def print_system_info(fuzzyshell_instance, detailed: bool = False):
    """Print formatted system information."""
    info = fuzzyshell_instance.get_system_info()
    
    print(f"FuzzyShell v{info['version']} - System Status")
    print("=" * 45)
    
    # Database info
    print("DATABASE:")
    print(f"  Path: {info['database']['path']}")
    print(f"  Commands: {info['database'].get('actual_command_count', 'Unknown')}")
    print(f"  Embeddings: {info['database'].get('embedding_count', 'Unknown')} ({info['database'].get('embedding_coverage', 'Unknown')} coverage)")
    if detailed:
        print(f"  Cached queries: {info['database'].get('cached_queries', 'Unknown')}")
    
    # Embedding model info  
    print("\nEMBEDDING MODEL:")
    emb_model = info.get('embedding_model', {})
    if 'error' in emb_model:
        print(f"  Error: {emb_model['error']}")
    else:
        # Use description if available, fallback to model name
        if 'description' in emb_model:
            print(f"  Model: {emb_model['description']}")
        else:
            model_name = emb_model.get('model_name', 'Unknown')
            print(f"  Model: {model_name}")
        
        print(f"  Status: {emb_model.get('status', 'Unknown')}")
        print(f"  Dimensions: {emb_model.get('dimensions', 'Unknown')}")
        
        if detailed:
            print(f"  Repository: {emb_model.get('repository', 'Unknown')}")
            print(f"  Path: {emb_model.get('model_path', 'Unknown')}")
            print(f"  File size: {emb_model.get('file_size', 'Unknown')}")
    
    # Search configuration
    print("\nSEARCH CONFIG:")
    search_config = info.get('search_configuration', {})
    print(f"  ANN enabled: {search_config.get('ann_enabled', 'Unknown')}")
    print(f"  Clusters: {search_config.get('ann_clusters', 'Unknown')}")
    print(f"  Search candidates: {search_config.get('ann_candidates', 'Unknown')}")
    print(f"  Storage format: {search_config.get('embedding_storage', 'Unknown')}")
    
    if detailed:
        print(f"  K1 (BM25): {search_config.get('bm25_k1', 'Unknown')}")
        print(f"  B (BM25): {search_config.get('bm25_b', 'Unknown')}")
        
        # Performance info
        print("\nPERFORMANCE:")
        perf = info.get('performance', {})
        print(f"  ANN command count: {perf.get('ann_command_count', 'Unknown')}")
        print(f"  Poorly clustered: {perf.get('poorly_clustered_commands', 'Unknown')}")
        print(f"  Last updated: {perf.get('last_updated', 'Unknown')}")


def handle_cli_commands():
    """Handle command-line arguments and execute CLI commands."""
    parser = argparse.ArgumentParser(description="FuzzyShell: A semantic search for your command history.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--ingest', action='store_true', help='Full re-ingestion from shell history (use for repairs/model changes).')
    parser.add_argument('--rebuild-ann', action='store_true', help='Rebuild the ANN index for optimal search performance.')
    parser.add_argument('--scoring', action='store_true', help='Show semantic and BM25 scores for each result.')
    parser.add_argument('--status', action='store_true', help='Show system status and model information.')
    parser.add_argument('--info', action='store_true', help='Show detailed system information including model details.')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches and force fresh results.')
    parser.add_argument('--no-ann', action='store_true', help='Disable ANN search for debugging (uses full linear search).')
    parser.add_argument('--profile', action='store_true', help='Show detailed timing and performance information.')
    parser.add_argument('--no-random', action='store_true', help='Disable random funny messages during ingestion and show raw progress.')
    parser.add_argument('--install-hook', action='store_true', help='Install shell integration hook for command insertion.')

    args = parser.parse_args()
    
    # Handle cache clearing (do this before creating FuzzyShell instance)
    if args.clear_cache:
        from .fuzzyshell import FuzzyShell
        fuzzyshell_temp = FuzzyShell()
        fuzzyshell_temp.clear_all_caches()
        print("✅ All caches cleared. Search results will be recalculated from scratch.")
        sys.exit(0)
    
    # Handle install hook
    if args.install_hook:
        install_shell_hook()
        sys.exit(0)
        
    # Handle different command modes
    if args.status or args.info or args.ingest or args.rebuild_ann:
        # Non-interactive modes - create FuzzyShell normally
        from .fuzzyshell import FuzzyShell
        fuzzyshell = FuzzyShell()
        
        if args.status:
            print_system_info(fuzzyshell, detailed=False)
            sys.exit(0)
        elif args.info:
            print_system_info(fuzzyshell, detailed=True)  
            sys.exit(0)
        elif args.ingest:
            print("🔄 Starting full re-ingestion from shell history...")
            print("⚠️  This will clear all caches and rebuild the database.")
            fuzzyshell.ingest_history(no_random=args.no_random)
            print("✅ Ingestion complete!")
            sys.exit(0)
        elif args.rebuild_ann:
            print("🔧 Rebuilding ANN index...")
            success = fuzzyshell.rebuild_ann_index()
            if success:
                fuzzyshell.metadata_dal.set('ann_command_count', str(fuzzyshell.get_indexed_count()))
                fuzzyshell.metadata_dal.set('poorly_clustered_commands', '0')
                print("✅ ANN index rebuilt successfully!")
            else:
                print("❌ ANN index rebuild failed!")
                sys.exit(1)
            sys.exit(0)
    else:
        # Interactive mode - return args for further processing
        return args


def main():
    """Main entry point for the application"""
    args = handle_cli_commands()
    
    if args is None:
        # CLI command was handled, exit
        return
    
    # Interactive mode
    from .fuzzyshell import interactive_search, USE_ANN_SEARCH
    
    # Handle ANN disable flag
    if args.no_ann:
        import fuzzyshell.fuzzyshell as fs_module
        fs_module.USE_ANN_SEARCH = False
        print("🔍 ANN search disabled - using full linear search for debugging")
        
    # Handle profiling flag
    if args.profile:
        logger.setLevel(logging.INFO)  # Enable more detailed logging
        print("🔍 Profiling mode enabled - detailed timing will be shown")
    
    result = interactive_search(show_profiling=args.profile)
    if result:
        # Print the selected command so it can be captured by the shell wrapper
        print(result)
    sys.exit(0)