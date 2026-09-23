#!/bin/bash
echo "=== Completion Debug Script ==="
echo "Current directory: $PWD"
echo ""

echo "=== 1. Check Python and argcomplete ==="
which python
python --version
python -c "import argcomplete; print(f'✅ Argcomplete version: {argcomplete.__version__}')" 2>/dev/null || echo "❌ Argcomplete not available"
echo ""

echo "=== 2. Check script exists and has magic comment ==="
if [[ -f "training_project/scripts/train.py" ]]; then
    echo "✅ Script exists"
    head -3 training_project/scripts/train.py | grep -n "ARGCOMPLETE"
else
    echo "❌ Script not found at training_project/scripts/train.py"
fi
echo ""

echo "=== 3. Test argcomplete directly ==="
echo "Testing: COMP_WORDS='python training_project/scripts/train.py --' COMP_CWORD=2 python training_project/scripts/train.py"
COMP_WORDS="python training_project/scripts/train.py --" COMP_CWORD=2 python training_project/scripts/train.py 2>/dev/null || echo "Direct test failed"
echo ""

echo "=== 4. Test register-python-argcomplete ==="
if command -v register-python-argcomplete >/dev/null 2>&1; then
    echo "✅ register-python-argcomplete found"
    echo "Testing registration:"
    register-python-argcomplete training_project/scripts/train.py
else
    echo "❌ register-python-argcomplete not found"
fi
echo ""

echo "=== 5. Check shell completion functions ==="
type _python_argcomplete 2>/dev/null && echo "✅ _python_argcomplete function exists" || echo "❌ _python_argcomplete function missing"
echo ""

echo "=== 6. Manual test ==="
echo "Try this manually:"
echo "eval \"\$(register-python-argcomplete training_project/scripts/train.py)\""
echo "python training_project/scripts/train.py --<TAB>"
