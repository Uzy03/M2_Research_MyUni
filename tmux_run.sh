#!/bin/bash
# Run a command in a named tmux session.
# On success (exit 0): kill the session automatically.
# On failure (exit != 0): keep the session for inspection.
#
# Usage: bash team/tmux_run.sh <session_name> <command...>

SESSION="$1"
shift
CMD="$*"

# Kill existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Write command to a temp script to avoid escaping issues
TMPSCRIPT=$(mktemp /tmp/tmux_XXXXXX.sh)
cat > "$TMPSCRIPT" << SCRIPT_EOF
#!/bin/bash
$CMD
EXIT=\$?
echo ""
if [ \$EXIT -eq 0 ]; then
    echo "[SUCCESS] Closing session in 3s..."
    sleep 3
    tmux kill-session -t '$SESSION'
else
    echo "[FAILED exit=\$EXIT] Session '$SESSION' kept for inspection"
fi
rm -f $TMPSCRIPT
SCRIPT_EOF

chmod +x "$TMPSCRIPT"
tmux new-session -d -s "$SESSION" bash "$TMPSCRIPT"

echo "============================================"
echo " tmux session: $SESSION"
echo "   attach: tmux attach -t $SESSION"
echo "   list:   tmux ls"
echo "   kill:   tmux kill-session -t $SESSION"
echo "============================================"
