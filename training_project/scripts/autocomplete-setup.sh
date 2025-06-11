#!/bin/bash
# training_project/scripts/autocomplete-setup.sh

echo "Setting up autocompletion for YOLO training scripts..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPLETIONS_DIR="$SCRIPT_DIR/completions"

# Install argcomplete if not present
if ! python -c "import argcomplete" 2>/dev/null; then
    echo "Installing argcomplete..."
    pip install argcomplete
fi

# Setup shell completion
SHELL_NAME=$(basename "$SHELL")

case "$SHELL_NAME" in
"bash")
    echo "Setting up Bash completion..."

    # Check if completion line already exists
    if ! grep -q "source.*completion.bash" ~/.bashrc; then
        echo "# YOLO Training Autocompletion" >>~/.bashrc
        echo "source $COMPLETIONS_DIR/completion.bash" >>~/.bashrc
        echo "Added to ~/.bashrc"
    fi

    # Source immediately
    source "$COMPLETIONS_DIR/completion.bash"
    ;;

"zsh")
    echo "Setting up Zsh completion..."

    # Add to fpath
    if ! grep -q "fpath.*training_project.*completions" ~/.zshrc; then
        echo "# YOLO Training Autocompletion" >>~/.zshrc
        echo "fpath=($COMPLETIONS_DIR \$fpath)" >>~/.zshrc
        echo "autoload -U compinit && compinit" >>~/.zshrc
        echo "Added to ~/.zshrc"
    fi
    ;;

*)
    echo "Unsupported shell: $SHELL_NAME"
    echo "Please manually add completion support"
    ;;
esac

# Enable argcomplete globally
if command -v activate-global-python-argcomplete >/dev/null 2>&1; then
    echo "Enabling global Python argcomplete..."
    activate-global-python-argcomplete --user
fi

echo "✅ Autocompletion setup complete!"
echo "Please restart your terminal or run: source ~/.${SHELL_NAME}rc"
