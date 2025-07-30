#!/bin/bash
# FuzzyShell shell integration for bash
# Add this to your ~/.bashrc:
# source /path/to/fuzzyshell-wrapper.bash

__fuzzyshell_widget() {
    # Clear the current line
    READLINE_LINE=""
    READLINE_POINT=0
    
    # Run fuzzy and capture the selected command
    local selected_command
    selected_command=$(fuzzy 2>/dev/null)
    
    # If a command was selected, put it in the buffer
    if [[ -n "$selected_command" ]]; then
        READLINE_LINE="$selected_command"
        READLINE_POINT=${#READLINE_LINE}
    fi
}

# Bind to Ctrl-F (or choose your preferred key)
bind -x '"\C-f": __fuzzyshell_widget'

# Optional: Add an alias for manual invocation
alias fs='fuzzy'