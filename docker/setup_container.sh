#!/usr/bin/env bash
# Rebuild the tested ApexNav Jazzy environment inside the project container.
set -euo pipefail

project_root="${APEXNAV_ROOT:-/workspace/ApexNav}"
venv="${APEXNAV_VENV:-/opt/apexnav_venv}"
deps="${APEXNAV_DEPS:-/opt/apexnav_deps}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script as root inside the ApexNav container." >&2
    exit 1
fi
if [[ ! -x "$venv/bin/python" ]]; then
    echo "Python virtualenv not found: $venv" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME="$cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-2}"

source "$venv/bin/activate"
mkdir -p "$deps" "$project_root/data"

echo "[1/9] System build and ROS dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git git-lfs ninja-build patch wget \
    freeglut3-dev libarmadillo-dev libbullet-dev libgl1-mesa-dev libomp-dev \
    libompl-dev xorg-dev \
    ros-jazzy-cv-bridge ros-jazzy-message-filters \
    ros-jazzy-pcl-conversions ros-jazzy-tf-transformations \
    ros-jazzy-tf2-geometry-msgs ros-jazzy-tf2-ros

install -D -m 0644 "$project_root/docker/10_nvidia.json" \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo "[2/9] CUDA 12.8 compiler (drivers are supplied by the host)"
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
    keyring=/tmp/cuda-keyring_1.1-1_all.deb
    wget -q -O "$keyring" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i "$keyring"
    apt-get update
    apt-get install -y --no-install-recommends cuda-compiler-12-8
fi

echo "[3/9] Python and CUDA runtime"
python -m pip install --upgrade "pip>=25" "setuptools>=69" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install \
    numpy==1.26.4 numba==0.60.0 scipy==1.13.1 pillow==10.4.0 \
    opencv-python==4.6.0.66 pandas==2.2.2 seaborn==0.13.2 \
    transformers==4.43.2 timm==0.4.12 rospkg==1.5.1 flask==3.0.3 \
    prettytable moviepy==1.0.3 requests thop supervision==0.22.0
python -m pip install \
    open3d==0.19.0 ultralytics==8.3.39 fastapi==0.111.1 \
    uvicorn==0.30.3 modelscope==1.16.1 accelerate==0.32.1

clone_at() {
    local url="$1"
    local directory="$2"
    local revision="$3"
    if [[ ! -d "$directory/.git" ]]; then
        git clone --recursive "$url" "$directory"
    fi
    git -C "$directory" fetch --tags origin
    git -C "$directory" checkout --detach "$revision"
    git -C "$directory" submodule sync --recursive
    git -C "$directory" submodule update --init --recursive
}

apply_once() {
    local directory="$1"
    local strip="$2"
    local patch_file="$3"
    if patch -d "$directory" --dry-run --forward -p"$strip" < "$patch_file" \
        >/dev/null 2>&1; then
        patch -d "$directory" --forward -p"$strip" < "$patch_file"
    elif ! patch -d "$directory" --dry-run --reverse -p"$strip" < "$patch_file" \
        >/dev/null 2>&1; then
        echo "Patch cannot be applied or reversed: $patch_file" >&2
        exit 1
    fi
}

echo "[4/9] Habitat-Lab and Habitat-Sim v0.3.3"
clone_at https://github.com/facebookresearch/habitat-lab.git \
    "$deps/habitat-lab" v0.3.3
python -m pip install --no-deps -e "$deps/habitat-lab/habitat-lab"
apply_once "$deps/habitat-lab" 1 \
    "$project_root/docker/patches/habitat-lab-agent-sensors.patch"
clone_at https://github.com/facebookresearch/habitat-sim.git \
    "$deps/habitat-sim" acbe6f49
if ! python -c 'import habitat_sim' >/dev/null 2>&1; then
    (
        cd "$deps/habitat-sim"
        python setup.py build_ext --parallel 4 install \
            --headless --bullet --no-update-submodules \
            --build-type Release --no-lto
    )
fi

echo "[5/9] GroundingDINO CUDA extension"
clone_at https://github.com/IDEA-Research/GroundingDINO.git \
    "$deps/GroundingDINO" 856dde20aee659246248e20734ef9ba5214f5e44
apply_once "$deps/GroundingDINO" 1 \
    "$project_root/docker/patches/groundingdino-pytorch210.patch"
python -m pip install addict pycocotools yapf
site_packages="$venv/lib/python3.12/site-packages"
export CPATH="$site_packages/nvidia/cusparse/include:$site_packages/nvidia/cublas/include:$site_packages/nvidia/cusolver/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$site_packages/nvidia/cusparse/lib:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cusolver/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
python -m pip install --no-build-isolation --no-deps -e "$deps/GroundingDINO"

echo "[6/9] YOLOv7 and MobileSAM"
clone_at https://github.com/WongKinYiu/yolov7.git \
    "$deps/yolov7" a207844b1ce82d204ab36d87d496728d3d2348e7
apply_once "$deps/yolov7" 1 \
    "$project_root/docker/patches/yolov7-pytorch210.patch"
apply_once "$deps/yolov7" 1 \
    "$project_root/docker/patches/yolov7-absolute-weights.patch"
clone_at https://github.com/ChaoningZhang/MobileSAM.git \
    "$deps/MobileSAM" f706ad9c4eb7f219c00d9050e46328518ffb65d2
python -m pip install --no-deps -e "$deps/MobileSAM"

echo "[7/9] LAVIS inference runtime"
python -m pip install --no-deps salesforce-lavis==1.0.2
python -m pip install \
    iopath contexttimer decord einops==0.8.2 fairscale==0.4.4 ftfy \
    sentencepiece webdataset scikit-image python-magic "spacy>=3.8,<3.9" \
    pycocoevalcap==1.2
lavis_dir="$site_packages/lavis"
apply_once "$site_packages" 1 \
    "$project_root/docker/patches/lavis-low-memory-eva.patch"

echo "[8/9] Model weights"
wget -q --show-progress -c -O "$project_root/data/groundingdino_swint_ogc.pth" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q --show-progress -c -O "$project_root/data/yolov7-e6e.pt" \
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
install -m 0644 "$deps/MobileSAM/weights/mobile_sam.pt" \
    "$project_root/data/mobile_sam.pt"

echo "[9/9] ApexNav Python package and verification"
python -m pip install --no-deps -e "$project_root"
python -m pip install openai==3.37.0 ollama==0.6.3
source /opt/ros/jazzy/setup.bash
colcon --log-base /opt/apexnav_ws/log build \
    --base-paths "$project_root/src" \
    --build-base /opt/apexnav_ws/build \
    --install-base /opt/apexnav_ws/install
source /opt/apexnav_ws/install/setup.bash
python "$project_root/docker/verify_environment.py"

echo "ApexNav container environment is ready."
