#compdef uv\ run\ python\ training_project/scripts/train.py python\ training_project/scripts/train.py training_project/scripts/train.py

# YOLO Training Autocompletion for Zsh with fzf-tab support
# This function should only be called for the specific commands above

function _train_fzf_completion() {
    # Only run if we're actually completing the train script
    local service="$words[1]"
    if [[ "$service" != *"train.py"* && "$words" != *"training_project/scripts/train.py"* ]]; then
        return 1
    fi
    
    local -a args
    args=(
        '--config[Configuration file]:config file:_files -g "training_project/config/*.yaml"'
        '--epochs[Number of epochs]:epochs:(10 50 100 200 300)'
        '--batch[Batch size]:batch:(-1 8 16 32 64)'
        '--device[Device]:device:(auto cpu mps cuda)'
        '--model[Model]:model:(yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt)'
        '--workers[Workers]:workers:(0 2 4 8)'
        '--run-name[Run name]:name:'
        '--amp[Enable AMP]'
        '--no-amp[Disable AMP]'
        '--check-mps[Check MPS]'
        '--optimize-for-mps[Optimize for MPS]'
        '--no-resume[No resume]'
        '--help[Help]'
    )
    
    _arguments -S -s $args
}

# Only register for specific commands
compdef _train_fzf_completion 'uv run python training_project/scripts/train.py'
compdef _train_fzf_completion 'python training_project/scripts/train.py'
compdef _train_fzf_completion 'training_project/scripts/train.py'
