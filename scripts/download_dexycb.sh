#!/usr/bin/env bash
set -u
DEST="${1:-data/raw/dexycb}"
shift || true
mkdir -p "$DEST"
declare -A FILES=(
  [dex-ycb-20210415.tar.gz]=18fD8RtWJM_fBi3bsuQTzdjIKxdRYtl57
  [20200709-subject-01.tar.gz]=1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk
  [20200813-subject-02.tar.gz]=1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62
  [20200820-subject-03.tar.gz]=1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq
  [20200903-subject-04.tar.gz]=14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA
  [20200908-subject-05.tar.gz]=1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS
  [20200918-subject-06.tar.gz]=1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ
  [20200928-subject-07.tar.gz]=1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0
  [20201002-subject-08.tar.gz]=1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv
  [20201015-subject-09.tar.gz]=1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH
  [20201022-subject-10.tar.gz]=1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-
  [bop.tar.gz]=1CPqLjsaYNjE3xSJbuWmqaMsGvyGIxiKL
  [calibration.tar.gz]=1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ
  [models.tar.gz]=1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM
)
if [ "$#" -eq 0 ]; then
  TARGETS=(models 20200709-subject-01 20200813-subject-02)
else
  TARGETS=("$@")
fi
for name in "${TARGETS[@]}"; do
  f="${name}.tar.gz"
  id="${FILES[$f]:-}"
  if [ -z "$id" ]; then echo "unknown target: $name"; exit 1; fi
  if [ -f "$DEST/$f" ] && [ -s "$DEST/$f" ]; then
    echo "skip $f (exists)"; continue
  fi
  echo "downloading $f"
  uv run gdown "$id" -O "$DEST/$f" || { echo "FAILED $f"; exit 1; }
done
echo "all downloads done"
