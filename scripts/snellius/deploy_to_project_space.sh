#!/bin/bash
# Usage
# ./scripts/snellius/deploy_to_shared.sh

LOCAL_DIR="$PWD"
SHARED_DIR="/INSERT-GPT-NL-TRAINING-DIR/pytorch-native"
REQUIRED_BRANCH="main"

abort() {
  echo "Cancelled because of $1"
  exit 1
}

if [ ! -d "$LOCAL_DIR" ]; then
  abort "Local directory '$LOCAL_DIR' does not exist."
fi

cd "$LOCAL_DIR" || abort "Failed to enter local directory."

if [ ! -d ".git" ]; then
  abort "Directory '$LOCAL_DIR' is not a Git repository."
fi

if [[ -n $(git status --porcelain | grep -v '^??') ]]; then
  abort "You have uncommitted changes. Please commit or stash them before deploying."
fi

current_branch=$(git symbolic-ref --short HEAD)
if [ "$current_branch" != "$REQUIRED_BRANCH" ]; then
  abort "You must be on the '$REQUIRED_BRANCH' branch to deploy. Current: '$current_branch'"
fi

echo "Pulling latest changes from origin/$REQUIRED_BRANCH..."
git pull origin "$REQUIRED_BRANCH" || abort "Git pull failed."

# Collect Git-tracked files and their parent directories
{
  git ls-files
  git ls-files | xargs -n1 dirname
} | sort -u > /tmp/git-tracked-paths.txt

# Add --delete flag if you know what you are doing and want an exact mirror!
rsync -av \
  --exclude="venv*" \
  --exclude="wandb*" \
  --exclude=".git" \
  --files-from=<(git ls-files) ./ "$SHARED_DIR/"
rm /tmp/git-tracked-paths.txt

echo "Deployment successful!"