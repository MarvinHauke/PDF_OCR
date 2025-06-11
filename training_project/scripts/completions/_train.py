#compdef train.py uv\ run\ python\ training_project/scripts/train.py python\ training_project/scripts/train.py

# YOLO Training Autocompletion for Zsh
_train_py() {
    local context state line
    typeset -A opt_args

    # Function to find config files
    _config_files() {
        local config_dir
        if [[ -d "training_project/config" ]]; then
            config_dir="training_project/config"
        elif [[ -d "config" ]]; then
            config_dir="config"
        else
            return 1
        fi
        
        local -a config_files
        config_files=(${config_dir}/*.yaml(.N:t))
        _describe 'config files' config_files
    }

    _arguments -C \
        '--config[Configuration YAML file]:config file:_config_files' \
        '--epochs[Number of epochs]:epochs:(10 50 100 200 300)' \
        '--batch[Batch size]:batch:(-1 8 16 32 64)' \
        '--device[Device to use]:device:(auto cpu mps cuda)' \
        '--no-resume[Disable auto-resume]' \
        '--model[Model architecture]:model:(yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt)' \
        '--run-name[Name for training run]:run name:' \
        '--workers[Number of workers]:workers:(0 2 4 8)' \
        '--amp[Enable AMP]' \
        '--no-amp[Disable AMP]' \
        '--check-mps[Check MPS availability]' \
        '--optimize-for-mps[Apply MPS optimizations]' \
        '--help[Show help]' \
        '*::args:_files'
}

_train_py "$@"
