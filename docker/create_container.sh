#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
container="${APEXNAV_CONTAINER:-apexnav-jazzy-egl}"
image="${APEXNAV_IMAGE:-apexnav-jazzy:apexnav-v033-cu128}"

if docker inspect "$container" >/dev/null 2>&1; then
    echo "Container already exists: $container" >&2
    echo "Set APEXNAV_CONTAINER to create one under a different name." >&2
    exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Image not found: $image" >&2
    exit 1
fi

docker run -d \
    --name "$container" \
    --gpus all \
    --network host \
    --shm-size 2g \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --mount "type=bind,src=$project_root,dst=/workspace/ApexNav" \
    --workdir /workspace/ApexNav \
    "$image" sleep infinity

echo "Created: $container"
echo "Open it with: docker/apexnav shell"
