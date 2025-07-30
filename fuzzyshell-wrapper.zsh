#!/bin/zsh
# FuzzyShell shell integration for zsh
# Add this to your ~/.zshrc:
# source /path/to/fuzzyshell-wrapper.zsh

fuzzyshell-widget() {
    # Clear the current line
    BUFFER=""
    CURSOR=0
    zle reset-prompt
    
    # Run fuzzy and capture the selected command
    local selected_command
    selected_command=$(fuzzy 2>/dev/null)
    
    # If a command was selected, put it in the buffer
    if [[ -n "$selected_command" ]]; then
        BUFFER="$selected_command"
        CURSOR=$#BUFFER
    fi
    
    # Redraw the prompt
    zle reset-prompt
}

# Create a zsh widget
zle -N fuzzyshell-widget

# Bind to Ctrl-F (or choose your preferred key)
bindkey '^F' fuzzyshell-widget

# Optional: Add an alias for manual invocation
alias fs='fuzzy'