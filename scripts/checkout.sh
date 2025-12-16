set -xe
# Check if target directory already exists
if [ -d "$3" ]; then
    echo "Directory $3 already exists. Checking out branch $2..."
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
    echo "Cloning repository to $3 with branch $2..."
    git clone $1 --revision="$2" --depth=1 "$3"
fi
