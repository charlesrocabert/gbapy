#!/usr/bin/env bash

pip uninstall gba --break-system-packages
rm -rf dist
rm -rf build
rm -rf gba.egg-info
rm -rf gba/__pycache__
python -m build
pip install --no-cache-dir . --break-system-packages

