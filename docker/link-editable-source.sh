#!/bin/sh
set -eu

REPO_ROOT="${1:-/workspace/robot-control-stack}"

if [ ! -d "$REPO_ROOT" ]; then
    echo "Mounted repo not found at $REPO_ROOT; leaving installed packages unchanged."
    exit 0
fi

SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

link_mixed_package() {
    src_dir="$1"
    dst_dir="$2"
    keep_dir_name="$3"

    if [ ! -d "$src_dir" ] || [ ! -d "$dst_dir/$keep_dir_name" ]; then
        return
    fi

    tmp_keep="$(mktemp -d)"
    mv "$dst_dir/$keep_dir_name" "$tmp_keep/$keep_dir_name"
    rm -rf "$dst_dir"
    mkdir -p "$dst_dir"
    cp -as "$src_dir/." "$dst_dir/"
    rm -rf "$dst_dir/$keep_dir_name"
    mv "$tmp_keep/$keep_dir_name" "$dst_dir/$keep_dir_name"
    rmdir "$tmp_keep"
}

link_pure_python_package() {
    src_dir="$1"
    dst_dir="$2"

    if [ ! -d "$src_dir" ]; then
        return
    fi

    rm -rf "$dst_dir"
    ln -s "$src_dir" "$dst_dir"
}

link_mixed_package "$REPO_ROOT/python/rcs" "$SITE_PACKAGES/rcs" "_core"
link_mixed_package "$REPO_ROOT/extensions/rcs_fr3/src/rcs_fr3" "$SITE_PACKAGES/rcs_fr3" "_core"
link_pure_python_package "$REPO_ROOT/extensions/rcs_realsense/src/rcs_realsense" "$SITE_PACKAGES/rcs_realsense"
link_pure_python_package "$REPO_ROOT/extensions/rcs_robotiq2f85/src/rcs_robotiq2f85" "$SITE_PACKAGES/rcs_robotiq2f85"
