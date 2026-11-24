#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'Usage: %s OUTPUT.tar.gz [REVISION]\n' "$0" >&2
    exit 2
fi

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$root" rev-parse --verify "${2:-HEAD}^{commit}")
output=$1
module=packages/anndatapp
entry=$(git -C "$root" ls-tree "$commit" -- "$module")
if [ -z "$entry" ]; then
    module=extern/anndatapp
    entry=$(git -C "$root" ls-tree "$commit" -- "$module")
fi
case "$entry" in
    '160000 commit '*) ;;
    *)
        printf 'Revision %s has no AnnData++ gitlink at %s.\n' "$commit" "$module" >&2
        exit 1
        ;;
esac
pin=$(git -C "$root" rev-parse --verify "$commit:$module")
checkout=packages/anndatapp
if [ ! -e "$root/$checkout/.git" ]; then checkout=$module; fi
if [ "$(git -C "$root/$checkout" rev-parse --show-toplevel 2>/dev/null || true)" != "$root/$checkout" ] ||
    ! git -C "$root/$checkout" cat-file -e "$pin^{commit}" 2>/dev/null; then
    printf 'Initialize %s and make pinned commit %s available before archiving.\n' "$checkout" "$pin" >&2
    exit 1
fi

staging=$(mktemp -d "$(dirname "$output")/.cellerator-source.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
git -C "$root" archive --format=tar --prefix=Cellerator/ "$commit" > "$staging/root.tar"
git -C "$root/$checkout" archive --format=tar --prefix="Cellerator/$module/" "$pin" > "$staging/ann.tar"
tar --concatenate --file="$staging/root.tar" "$staging/ann.tar"
gzip -n -c "$staging/root.tar" > "$staging/source.tar.gz"
mv -- "$staging/source.tar.gz" "$output"
