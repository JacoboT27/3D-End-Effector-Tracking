#!/bin/bash
# download_assets.sh
# Downloads the minimum required robot model files from MuJoCo Menagerie.
# Run this once before starting the Docker container.

set -e

BASE_URL="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"

download_file() {
    local url=$1
    local dest=$2
    mkdir -p "$(dirname "$dest")"
    echo "  Downloading $dest..."
    curl -fsSL "$url" -o "$dest"
}

# --- Franka Emika Panda ---
echo "Downloading Franka Emika Panda model..."
FRANKA_URL="$BASE_URL/franka_emika_panda"
FRANKA_DIR="assets/franka"

download_file "$FRANKA_URL/panda.xml"               "$FRANKA_DIR/panda.xml"
download_file "$FRANKA_URL/panda_arm.xml"            "$FRANKA_DIR/panda_arm.xml"
download_file "$FRANKA_URL/panda_arm_hand.xml"       "$FRANKA_DIR/panda_arm_hand.xml"

# meshes
FRANKA_ASSETS=(
    "link0.stl" "link1.stl" "link2.stl" "link3.stl"
    "link4.stl" "link5.stl" "link6.stl" "link7.stl"
    "hand.stl" "finger.stl"
)
for mesh in "${FRANKA_ASSETS[@]}"; do
    download_file "$FRANKA_URL/assets/$mesh" "$FRANKA_DIR/assets/$mesh"
done
echo "Franka model ready."

# --- Universal Robots UR5e ---
echo "Downloading UR5e model..."
UR5_URL="$BASE_URL/universal_robots_ur5e"
UR5_DIR="assets/ur5e"

download_file "$UR5_URL/ur5e.xml"                   "$UR5_DIR/ur5e.xml"

# meshes
UR5_ASSETS=(
    "base.stl" "shoulder.stl" "upperarm.stl"
    "forearm.stl" "wrist1.stl" "wrist2.stl" "wrist3.stl"
)
for mesh in "${UR5_ASSETS[@]}"; do
    download_file "$UR5_URL/assets/$mesh" "$UR5_DIR/assets/$mesh"
done
echo "UR5e model ready."

echo ""
echo "All assets downloaded. You can now run: docker-compose up train"