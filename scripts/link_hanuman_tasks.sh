#!/usr/bin/env bash
# Link HANUMAN's Layer 1 task module (layer_1/system1) into the installed mjlab
# task registry so the Hanuman-* tasks register.
#
# mjlab is installed from git as a normal package (no editable/dev checkout), so
# its task package lives in the pixi env's site-packages. This script symlinks
# our task module in and adds the import that triggers registration.
#
# Runs automatically on `pixi` activation (see [activation] in pixi.toml) and is
# safe to run by hand. Idempotent and best-effort — it never aborts the shell.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
SYSTEM1="$REPO_ROOT/layer_1/system1"

# Locate the installed mjlab tasks package (follows the active python).
MJLAB_TASKS="$(python -c 'import os, mjlab.tasks; print(os.path.dirname(mjlab.tasks.__file__))' 2>/dev/null)"
if [ -z "$MJLAB_TASKS" ]; then
    echo "[link_hanuman_tasks] mjlab not importable yet — run 'pixi install', then reactivate." >&2
    return 0 2>/dev/null || exit 0
fi

# (Re)create the symlink to our task module.
ln -sfn "$SYSTEM1" "$MJLAB_TASKS/hanuman"

# Ensure the tasks package imports it (this is what registers the gym ids).
if ! grep -q "from . import hanuman" "$MJLAB_TASKS/__init__.py" 2>/dev/null; then
    echo "from . import hanuman  # HANUMAN tasks (linked by link_hanuman_tasks.sh)" >> "$MJLAB_TASKS/__init__.py"
fi

echo "[link_hanuman_tasks] HANUMAN tasks linked into $MJLAB_TASKS"
