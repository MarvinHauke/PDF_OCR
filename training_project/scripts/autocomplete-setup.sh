#!/bin/bash
# training_project/scripts/autocomplete-setup.sh

echo "Setting up autocompletion for YOLO training scripts..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPLETIONS_DIR="$SCRIPT_DIR/completions"

# Verify completion files exist
echo "🔍 Checking completion files..."
if [[ ! -f "$COMPLETIONS_DIR/completion.bash" ]]; then
    echo "❌ Missing: $COMPLETIONS_DIR/completion.bash"
    echo "Please ensure completion files are committed to the repository"
    exit 1
fi

if [[ ! -f "$COMPLETIONS_DIR/_train.py" ]]; then
    echo "❌ Missing: $COMPLETIONS_DIR/_train.py"
    echo "Please ensure completion files are committed to the repository"
    exit 1
fi

if [[ ! -f "$COMPLETIONS_DIR/_train_fzf.py" ]]; then
    echo "❌ Missing: $COMPLETIONS_DIR/_train_fzf.py"
    echo "Please ensure completion files are committed to the repository"
    exit 1
fi

echo "✅ All completion files found"

# Install argcomplete if not present
echo "🔧 Checking argcomplete..."
if ! python -c "import argcomplete" 2>/dev/null; then
    echo "Installing argcomplete..."
    if command -v uv >/dev/null 2>&1; then
        uv add argcomplete
    else
        pip install argcomplete
    fi
    echo "✅ argcomplete installed"
else
    echo "✅ argcomplete already available"
fi

# Detect if fzf-tab is installed
detect_fzf_tab() {
    if [[ "$SHELL_NAME" == "zsh" ]]; then
        # Check for fzf-tab in various common locations
        if grep -q "fzf-tab" ~/.zshrc 2>/dev/null ||
            grep -q "aloxaf/fzf-tab" ~/.zshrc 2>/dev/null ||
            [[ -n "$ZAP_DIR" ]] ||
            [[ -n "$ZSH" ]] ||
            command -v fzf >/dev/null 2>&1; then
            return 0 # fzf-tab likely installed
        fi
    fi
    return 1 # fzf-tab not detected
}

# Get absolute path for completions
COMPLETIONS_ABS_PATH="$(cd "$COMPLETIONS_DIR" && pwd)"

# Setup shell completion
SHELL_NAME=$(basename "$SHELL")

case "$SHELL_NAME" in
"bash")
    echo "🐚 Setting up Bash completion..."

    # Remove any existing YOLO completion entries
    if [ -f ~/.bashrc ]; then
        # Create a backup
        cp ~/.bashrc ~/.bashrc.backup."$(date +%Y%m%d_%H%M%S)"
        echo "📋 Created backup: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

        # Remove old entries
        sed -i '/# YOLO Training Autocompletion/,+1d' ~/.bashrc 2>/dev/null || true
    fi

    # Add new completion
    echo "" >>~/.bashrc
    echo "# YOLO Training Autocompletion" >>~/.bashrc
    echo "source $COMPLETIONS_ABS_PATH/completion.bash" >>~/.bashrc
    echo "✅ Added Bash completion to ~/.bashrc"

    # Source immediately for current session
    source "$COMPLETIONS_ABS_PATH/completion.bash"
    echo "✅ Bash completion activated for current session"
    ;;

"zsh")
    echo "🐚 Setting up Zsh completion..."

    # Remove any existing YOLO completion entries
    if [ -f ~/.zshrc ]; then
        # Create a backup
        cp ~/.zshrc ~/.zshrc.backup."$(date +%Y%m%d_%H%M%S)"
        echo "📋 Created backup: ~/.zshrc.backup.$(date +%Y%m%d_%H%M%S)"

        # Remove old entries (handle various patterns)
        sed -i '/# YOLO Training Autocompletion/,+5d' ~/.zshrc 2>/dev/null || true
        sed -i '/fzf-tab compatible/,+3d' ~/.zshrc 2>/dev/null || true
    fi

    # Detect if fzf-tab is available
    if detect_fzf_tab; then
        echo "🎯 Detected fzf-tab installation - using enhanced completion"

        # Add fzf-tab friendly completion
        echo "" >>~/.zshrc
        echo "# YOLO Training Autocompletion (fzf-tab compatible)" >>~/.zshrc
        echo "fpath=($COMPLETIONS_ABS_PATH \$fpath)" >>~/.zshrc
        echo "autoload -U compinit && compinit" >>~/.zshrc
        echo "source $COMPLETIONS_ABS_PATH/_train_fzf.py" >>~/.zshrc

        # Source immediately for current session
        fpath=("$COMPLETIONS_ABS_PATH" "$fpath")
        autoload -U compinit && compinit >/dev/null 2>&1
        source "$COMPLETIONS_ABS_PATH/_train_fzf.py"
        echo "✅ fzf-tab compatible completion activated"
    else
        echo "📁 Using standard zsh completion"

        # Add standard zsh completion
        echo "" >>~/.zshrc
        echo "# YOLO Training Autocompletion" >>~/.zshrc
        echo "fpath=($COMPLETIONS_ABS_PATH \$fpath)" >>~/.zshrc
        echo "autoload -U compinit && compinit" >>~/.zshrc

        # Source immediately for current session
        fpath=("$COMPLETIONS_ABS_PATH" "$fpath")
        autoload -U compinit && compinit >/dev/null 2>&1
        echo "✅ Standard zsh completion activated"
    fi

    # Clear completion cache to ensure new completions are loaded
    rm -f ~/.zcompdump* 2>/dev/null
    ;;

*)
    echo "❌ Unsupported shell: $SHELL_NAME"
    echo "Supported shells: bash, zsh"
    echo "Please manually add completion support or switch to bash/zsh"
    exit 1
    ;;
esac

# Enable argcomplete globally if available
echo "🌐 Setting up global argcomplete..."
if command -v activate-global-python-argcomplete >/dev/null 2>&1; then
    echo "🔧 Enabling global Python argcomplete..."
    activate-global-python-argcomplete --user 2>/dev/null || echo "ℹ️  Note: Global argcomplete setup skipped (may require different permissions)"
else
    echo "ℹ️  Global argcomplete not available - install argcomplete system-wide if needed"
fi

# Test completion
echo ""
echo "🧪 Testing completion setup..."
case "$SHELL_NAME" in
"bash")
    if type _train_completion >/dev/null 2>&1; then
        echo "✅ Bash completion function loaded successfully"
    else
        echo "⚠️  Bash completion function not found - may need to restart terminal"
    fi
    ;;
"zsh")
    if type _train_py >/dev/null 2>&1 || type _train_fzf >/dev/null 2>&1; then
        echo "✅ Zsh completion function loaded successfully"
    else
        echo "⚠️  Zsh completion function not found - may need to restart terminal"
    fi
    ;;
esac

# Show current configuration
echo ""
echo "📋 Current setup:"
echo "   Shell: $SHELL_NAME"
echo "   Completions dir: $COMPLETIONS_ABS_PATH"
echo "   Files: completion.bash, _train.py, _train_fzf.py"
if [[ "$SHELL_NAME" == "zsh" ]]; then
    if detect_fzf_tab; then
        echo "   Mode: fzf-tab enhanced"
    else
        echo "   Mode: standard zsh"
    fi
fi

echo ""
echo "🎉 Autocompletion setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your terminal or run: source ~/.${SHELL_NAME}rc"
echo "2. Test completion with: uv run python training_project/scripts/train.py --<TAB>"
echo ""
echo "🔧 Troubleshooting:"
echo "- If completion doesn't work immediately, restart your terminal"
echo "- For zsh users: try 'rm ~/.zcompdump* && autoload -U compinit && compinit'"
echo "- For fzf-tab users: ensure fzf-tab is loaded before testing"
echo "- Check that you're running from the correct directory (PDF_OCR/)"
echo ""
echo "📁 Files modified:"
echo "   ~/.${SHELL_NAME}rc (completion setup added)"
echo "   ~/.${SHELL_NAME}rc.backup.$(date +%Y%m%d_%H%M%S) (backup created)"
echo ""
echo "🎯 Happy training!"
