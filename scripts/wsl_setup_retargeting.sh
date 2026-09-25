#!/usr/bin/env bash
set -e
if [ ! -f "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv "$HOME/h2r-wsl" --clear --python 3.12 --allow-python-downloads
VIRTUAL_ENV="$HOME/h2r-wsl" uv pip install dex-retargeting pyyaml numpy torch --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$HOME/h2r-wsl/bin/python" -c "import dex_retargeting; print('dex-retargeting OK')"
