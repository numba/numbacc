set -xe
# Usage: checkout.sh <repo-url> <branch-or-commit> <target-dir>

if [ -d "$3" ]; then
    echo "Directory $3 already exists. Checking out $2..."
    cd "$3"
    git fetch origin
    if git show-ref --verify --quiet "refs/remotes/origin/$2"; then
        # It's a branch
        git checkout -B "$2" "origin/$2"
        git pull origin "$2"
    else
        # Assume it's a commit hash
        git checkout "$2"
    fi
else
    echo "Cloning repository to $3..."
    git clone "$1" "$3"
    cd "$3"
    git checkout "$2"
fi
