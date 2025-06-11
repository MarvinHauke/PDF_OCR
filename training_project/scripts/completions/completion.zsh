#!/bin/zsh
# training_project/scripts/completions/completion.zsh

#compdef train.py

_train_completion() {
  local context state line
  typeset -A opt_args

  _arguments \
    '--config[Configuration YAML file]:config file:_files -g "config/*.yaml"' \
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
    '--help[Show help]'
}

_train_completion "$@"
