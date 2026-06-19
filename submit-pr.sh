#!/usr/bin/env bash
# submit-pr.sh — fork, push, and open the migration PR for RVC-GUI
#
# Prerequisites:
#   1. `gh` (GitHub CLI) installed and authenticated: https://cli.github.com/
#   2. Run this script from the RVC-GUI repo root on the
#      `feature/migration-to-uv-and-cuda-update` branch.
#
# What it does:
#   - Forks FranzFischer78/RVC-GUI to your personal GitHub account (if not
#     already forked).
#   - Adds your fork as a new `fork` remote.
#   - Pushes the feature branch to your fork.
#   - Opens a Pull Request against FranzFischer78/RVC-GUI:main using the
#     PR template in PULL_REQUEST_TEMPLATE.md.

set -euo pipefail

UPSTREAM="FranzFischer78/RVC-GUI"
BRANCH="feature/migration-to-uv-and-cuda-update"

echo "==> Verifying gh CLI auth..."
gh auth status

echo "==> Current branch: $(git rev-parse --abbrev-ref HEAD)"
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]]; then
  echo "ERROR: not on $BRANCH. Run: git checkout $BRANCH"
  exit 1
fi

echo "==> Forking $UPSTREAM (if not already forked)..."
gh repo fork "$UPSTREAM" --remote=true --remote-name=fork || true

# Make sure the fork remote URL uses SSH if the user prefers SSH, otherwise HTTPS.
FORK_URL=$(gh repo view --json url --jq '.url' 2>/dev/null || true)
echo "==> Fork URL: $FORK_URL"

echo "==> Pushing $BRANCH to fork..."
git push -u fork "$BRANCH"

echo "==> Opening Pull Request..."
gh pr create \
  --repo "$UPSTREAM" \
  --base main \
  --head "$BRANCH" \
  --title "Migrate to uv + Python 3.10 + PyTorch 2.12.1 (CUDA 13.2)" \
  --body-file PULL_REQUEST_TEMPLATE.md

echo ""
echo "==> Done! PR opened against $UPSTREAM:main"
