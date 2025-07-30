#!/bin/zsh

# Function to handle the fuzzy search
function fuzzy_search() {
    # Get the command from fuzzyshell.py
    local cmd=$(python3 $(dirname $0)/fuzzyshell.py)
    if [ $? -eq 0 ] && [ ! -z "$cmd" ]; then
        # Put the command in the buffer and move cursor to end
        BUFFER="$cmd"
        CURSOR=${#BUFFER}
    fi
    # Force a redraw of the prompt
    zle reset-prompt
}

# Create a new ZLE widget
zle -N fuzzy_search

# Bind it to Ctrl+F (you can change this to your preferred key binding)
bindkey '^F' fuzzy_search

# Function to ingest history
function fuzzy_ingest() {
    python3 $(dirname $0)/fuzzyshell.py --ingest
}
