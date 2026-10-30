#!/bin/sh
set -eu

hook_dir=$(CDPATH= cd -- "$(dirname "$0")/../.githooks" && pwd)
test_root=$(mktemp -d)
trap 'rm -rf -- "$test_root"' EXIT
trap 'exit 1' HUP INT TERM
separator=$(printf '%100s' '' | tr ' ' '-')
file=learnings/learnings.txt

new_repo() {
    mkdir "$test_root/$1"
    cd "$test_root/$1"
    git init -q
    git config user.name "Hook Test"
    git config user.email "hook-test@example.invalid"
    git config commit.gpgsign false
    git config core.hooksPath "$hook_dir"
    mkdir learnings
}

commit() {
    git commit -qm test
}

assert_committed() {
    git show "HEAD:$file" > "$test_root/actual"
    cmp "$test_root/expected" "$test_root/actual"
    cmp "$test_root/expected" "$file"
}

new_repo initial
printf 'First learning\n' > "$file"
git add -- "$file"
commit
printf '%s\nFirst learning\n' "$separator" > "$test_root/expected"
assert_committed
[ "$(head -n 1 "$file" | tr -d '\n' | wc -c)" -eq 100 ]

# Unchanged learnings and already formatted changes must not gain separators.
git commit --allow-empty -qm unchanged
assert_committed
printf '%s\nAn edited learning\n' "$separator" > "$file"
git add -- "$file"
commit
cp "$file" "$test_root/expected"
assert_committed

# New entries go above the previous separator.
{
    printf 'Newest learning\n'
    cat "$test_root/expected"
} > "$file"
git add -- "$file"
"$hook_dir/pre-commit"
"$hook_dir/pre-commit"
commit
printf '%s\nNewest learning\n%s\nAn edited learning\n' "$separator" "$separator" > "$test_root/expected"
assert_committed

# Removing content alone does not count as a new entry.
printf 'An edited learning\n' > "$file"
git add -- "$file"
commit
printf 'An edited learning\n' > "$test_root/expected"
assert_committed

# Partial staging must fail without modifying either version.
printf 'Staged learning\nAn edited learning\n' > "$file"
git add -- "$file"
printf 'Unstaged learning\n' >> "$file"
cp "$file" "$test_root/worktree"
git show ":$file" > "$test_root/index"
if commit > "$test_root/output" 2>&1; then
    echo "FAIL: partially staged learnings were committed" >&2
    exit 1
fi
grep -q 'stage all changes' "$test_root/output"
cmp "$test_root/worktree" "$file"
git show ":$file" > "$test_root/actual"
cmp "$test_root/index" "$test_root/actual"

new_repo empty
: > "$file"
git add -- "$file"
commit
: > "$test_root/expected"
assert_committed
printf ' \n\n' > "$file"
git add -- "$file"
commit
printf ' \n\n' > "$test_root/expected"
assert_committed

new_repo unstaged
printf 'Not staged\n' > "$file"
git commit --allow-empty -qm unrelated
printf 'Not staged\n' > "$test_root/expected"
cmp "$test_root/expected" "$file"
if git cat-file -e "HEAD:$file" 2>/dev/null; then
    echo "FAIL: unstaged learnings were committed" >&2
    exit 1
fi

new_repo deleted
printf 'A learning\n' > "$file"
git add -- "$file"
commit
git rm -q -- "$file"
commit
[ ! -e "$file" ]

echo "All pre-commit hook tests passed."
