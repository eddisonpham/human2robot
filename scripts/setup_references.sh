#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/references"
mkdir -p "$REF"

clone() {
  local name="$1" url="$2"
  if [ -d "$REF/$name/.git" ]; then
    echo "[$name] already present, skipping"
  else
    echo "[$name] cloning"
    git clone --depth 1 "$url" "$REF/$name"
  fi
}

clone dex-retargeting https://github.com/dexsuite/dex-retargeting.git
clone mujoco_menagerie https://github.com/google-deepmind/mujoco_menagerie.git
clone cleanrl https://github.com/vwxyzjn/cleanrl.git
clone dexmv-sim https://github.com/yzqin/dexmv-sim.git
clone dexmv-learn https://github.com/yzqin/dexmv-learn.git
clone pddm https://github.com/google-research/pddm.git
clone dex-ycb-toolkit https://github.com/NVlabs/dex-ycb-toolkit.git
clone arctic https://github.com/zc-alexfan/arctic.git
clone grab https://github.com/otaheri/GRAB.git

echo "Done. See agents/03_EXISTING_REPOS_TO_CLONE.md for roles and licenses."
