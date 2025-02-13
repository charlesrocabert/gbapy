rm -rf dist
rm -rf build
pip uninstall gba --break-system-packages
python -m build
pip install . --break-system-packages

