#!/bin/bash
# training_project/scripts/completions/completion.bash

_train_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD - 1]}"

    # Available options
    opts="--config --epochs --batch --device --no-resume --model --run-name --workers --amp --no-amp --check-mps --optimize-for-mps --help"

    # Config file completion
    if [[ ${prev} == "--config" ]]; then
        local config_files=$(find config/ -name "*.yaml" 2>/dev/null | sed 's|config/||')
        COMPREPLY=("$(compgen -W "$config_files" -- "$cur")")
        return 0
    fi

    # Device completion
    if [[ ${prev} == "--device" ]]; then
        COMPREPLY=("$(compgen -W "auto cpu mps cuda" -- "$cur")")
        return 0
    fi

    # Model completion
    if [[ ${prev} == "--model" ]]; then
        local models="yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt"
        COMPREPLY=("$(compgen -W "$models" -- "$cur")")
        return 0
    fi

    # Batch size suggestions
    if [[ ${prev} == "--batch" ]]; then
        COMPREPLY=("$(compgen -W "-1 8 16 32 64" -- "$cur")")
        return 0
    fi

    # Epochs suggestions
    if [[ ${prev} == "--epochs" ]]; then
        COMPREPLY=("$(compgen -W "10 50 100 200 300" -- "$cur")")
        return 0
    fi

    # Workers suggestions
    if [[ ${prev} == "--workers" ]]; then
        COMPREPLY=("$(compgen -W "0 2 4 8" -- "$cur")")
        return 0
    fi

    # Default completion
    COMPREPLY=("$(compgen -W "$opts" -- "$cur")")
    return 0
}

# Register completion
complete -F _train_completion python scripts/train.py
complete -F _train_completion ./scripts/train.py
complete -F _train_completion train.py
