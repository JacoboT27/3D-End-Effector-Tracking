#!/bin/bash
# download_assets.sh
# Uses the robot_descriptions package to download robot model files.
# Run this once before starting the Docker container.

set -e

echo "Installing robot_descriptions..."
pip install robot_descriptions -q

echo "Downloading Franka Panda model..."
python3 - << 'PYEOF'
from robot_descriptions import panda_mj_description
import shutil, os

src = os.path.dirname(panda_mj_description.MJCF_PATH)
dst = "assets/franka"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"  Copied from {src} -> {dst}")
PYEOF

echo "Downloading UR5e model..."
python3 - << 'PYEOF'
from robot_descriptions import ur5e_mj_description
import shutil, os

src = os.path.dirname(ur5e_mj_description.MJCF_PATH)
dst = "assets/ur5e"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"  Copied from {src} -> {dst}")
PYEOF

echo ""
echo "All assets ready. You can now run: docker-compose up train"