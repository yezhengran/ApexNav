# ApexNav Jazzy container workflow

The tested development container is named `apexnav-jazzy-egl`. It uses ROS 2
Jazzy, Python 3.12, Habitat 0.3.3, PyTorch 2.10/CUDA 12.8, NVIDIA EGL, and the
pinned VLM repositories recorded in `setup_container.sh`.

## Daily commands

Run these commands from the repository root on the Ubuntu host:

```bash
docker/apexnav status
docker/apexnav shell
docker/apexnav build
docker/apexnav doctor
docker/apexnav check-data
docker/apexnav smoke
```

The shell command automatically activates `/opt/apexnav_venv` and sources ROS
2 Jazzy. The doctor command performs real CUDA matrix multiplication and a real
headless Habitat EGL render in addition to imports, commit pins, and weight
checks. `smoke` uses a real HM3D validation scene and checks that ROS receives
the rendered observations and returns a planner action. It does not use VLMs
or produce ObjectNav evaluation scores.

## All-local vision services (current deployment)

The laptop runs Habitat and ROS. Each vision client can independently call a
local or remote service without loading its weights in the navigation process.
On 2026-10-27 all four original vision models were deployed locally, retaining
the existing YOLO/GroundingDINO category routing and BLIP2 scores. No remote API
or replacement detector is needed for this configuration. Individual endpoints
can still move remotely later if larger scenes or other workloads exceed RAM or
VRAM. To remeasure the installed checkpoints in isolation:

```bash
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/profile_local_models.py
# JSON reports and logs: outputs/runtime/model-profile/
```

This runs isolated real-image inferences, not concurrent service loads or a
worst-case navigation benchmark. Local service startup is always explicit;
`smoke` and `run-one` never start local VLMs. Start individual chosen models with
`docker/apexnav vlm-start-local --service sam` (or `yolo`, `dino`, `blip2`).

The isolated laptop measurements on 2026-10-27 were:

| Module | Isolated PyTorch peak reserved GPU memory | Current placement |
| --- | --- | --- |
| MobileSAM vit_t | 346 MiB | Local, port 12183 |
| GroundingDINO SwinT | 1,052 MiB | Local, port 12181 |
| BLIP2 image-text matching | 2,834 MiB | Local, port 12182; preserves numeric ITC/ITM scores |
| YOLOv7-e6e | 2,574 MiB | Local, port 12184 |

These figures include allocator caching during load/inference, not just weights;
they exclude driver/context overhead. In the joint test, all four model services
stayed resident while a real Habitat ObjectNav scene turned through six views
and made 24 successful local HTTP inferences. Whole-GPU usage sampled every
200 ms reached 7,456 MiB out of 8,188 MiB, including the desktop. Shorter peaks
may be missed, and other scenes/long navigation episodes are not yet covered.
See `outputs/runtime/dataset-all-local-vlm-validation.json`.

YOLO warm-up now uses inference mode and releases idle startup CUDA cache.
The local env also enables `APEXNAV_VLM_RELEASE_IDLE_CACHE=1`, releasing unused
allocator blocks at server startup and after each inference so independent
processes and Habitat/EGL can share the GPU. Live weights, thresholds, and model
architectures are unchanged. This trades some allocation/latency overhead for
headroom; set it to `0` on a larger GPU to favor cache reuse. See
[PyTorch's cache-release semantics](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html).

The same opt-in also collects unreachable Python objects and calls glibc
`malloc_trim(0)` once after model loading/tracing, before serving requests.
This returns freed CPU buffers to the OS without changing live tensors; it is
skipped on hosts without `malloc_trim`. CPU trimming is not repeated per frame.

The 16 GB host RAM is also tight: about 1.3 GiB remained available after the
first joint test with other desktop apps open. Close unnecessary memory-heavy
applications before a full episode. `run-one` uses cached LLM answers instead
of loading Ollama/Qwen on the same GPU. Joint inference passing is not proof
that a full navigation episode or every scene fits.

```bash
# The current gitignored config/vlm.env already uses four localhost URLs.
# On a fresh checkout only: cp config/vlm.env.example config/vlm.env
docker/apexnav vlm-start-local
docker/apexnav vlm-check --health-only
docker/apexnav vlm-check
docker exec --env-file config/vlm.env -e HABITAT_SIM_LOG=error -e MAGNUM_LOG=quiet \
  apexnav-jazzy-egl /opt/apexnav_venv/bin/python \
  docker/validate_habitat_dataset.py --local-vlm --views 6
```

The helper passes this Docker env file into each command. `config/vlm.env` is
gitignored and may contain an optional `APEXNAV_VLM_API_KEY` bearer token.
Do not commit or paste that token into chat. URL precedence is: explicit client
`url=` argument, per-model environment URL, then `APEXNAV_VLM_HOST` with the
legacy model port. HTTP read timeout defaults to 120 seconds and is configurable
with `APEXNAV_VLM_TIMEOUT`.

| Model | Variable | Default endpoint | POST fields in addition to `image` |
| --- | --- | --- | --- |
| GroundingDINO | `APEXNAV_GDINO_URL` | `http://127.0.0.1:12181/gdino` | `caption`, `box_threshold`, `text_threshold` |
| BLIP2 | `APEXNAV_BLIP2_URL` | `http://127.0.0.1:12182/blip2itm` | `txt` |
| MobileSAM | `APEXNAV_SAM_URL` | `http://127.0.0.1:12183/mobile_sam` | pixel `bbox: [x1,y1,x2,y2]` |
| YOLOv7 | `APEXNAV_YOLO_URL` | `http://127.0.0.1:12184/yolov7` | `agnostic_nms`, `conf_thres`, `iou_thres` |

`image` is a base64 JPEG encoded by the existing ApexNav image transport.
Detection responses contain normalized xyxy `boxes`, `logits`, and `phrases`;
SAM returns `cropped_mask` as base64 uint8/bool bytes of the original image
height and width. BLIP2 returns numeric `response` (ITC cosine) and `itm score`.
Each service has a `GET /healthz` response such as
`{"status":"ready","model":"gdino"}`. HTTP errors propagate to the caller
instead of terminating evaluation with a successful exit code.

On a remote GPU server with this environment and checkout:

```bash
docker/apexnav vlm-start-local
```

By default, model services bind only to loopback. Forward all four ports from
the laptop (replace `user@gpu-server` with your actual SSH destination):

```bash
ssh -N \
  -L 12181:127.0.0.1:12181 \
  -L 12182:127.0.0.1:12182 \
  -L 12183:127.0.0.1:12183 \
  -L 12184:127.0.0.1:12184 \
  user@gpu-server
```

For SSH forwarding, configure all client URLs with `127.0.0.1`. The container
uses host networking, so it reaches these
forwarded ports. For a managed network deployment, the server can instead use
`APEXNAV_VLM_BIND_HOST` and `APEXNAV_VLM_SERVER_TOKEN`; the client sends the same
secret in `APEXNAV_VLM_API_KEY`. Put a production gateway in front if exposing
the service beyond a private network.

Server logs and PID records are stored in `outputs/runtime/`. Startups are
sequential to limit peak RAM. `docker/apexnav vlm-stop-local` stops only processes
owned by the launcher and checks kernel start times before signaling them.

A generic chat/completions vision API is not a drop-in replacement for these
endpoints: pixel masks, detection boxes, and BLIP2 similarity scores must retain
their original meaning for the original navigation method to be reproduced.

### Choosing a remote provider

Reviewed on 2026-10-26. These are options, not services already connected to this
checkout. No paid resources or API calls have been created. A ready-to-call
provider can avoid maintaining a GPU instance, but its protocol needs an adapter;
do not put a generic vision-chat URL into `APEXNAV_GDINO_URL` or `APEXNAV_BLIP2_URL`
and expect the original response fields. Keep a baseline and a method-modified
experiment separate if changing models or score definitions.

| Provider | Fit for this reproduction | Important constraint |
| --- | --- | --- |
| [AutoDL](https://www.autodl.com/docs/ssh/) | Optional GPU host when a required heavy model has no suitable hosted API; selected models can use SSH forwarding. | [Public custom ports require enterprise verification](https://www.autodl.com/docs/port/); personal users can forward ports over SSH instead. |
| [RunPod Pods](https://docs.runpod.io/pods/configuration/expose-ports) | Alternative GPU host; each model can have its own HTTPS proxy URL or SSH tunnel. | Its public HTTP proxy has a 100-second timeout. Add authentication; prefer SSH if inference can exceed that limit. |
| [Modal](https://modal.com/docs/guide/webhooks) | Programmable GPU-backed HTTP service, suitable for a later managed deployment. | Requires packaging the models and handling cold starts. [L4 GPU cost](https://modal.com/pricing) is currently $0.000222/s (about $0.80/GPU-hour); CPU, RAM, storage, and network can add cost. Model loading and keep-alive time are billable too. |
| [Replicate GroundingDINO](https://replicate.com/adirik/grounding-dino) | Ready-to-call detector for a separate API-adaptation experiment. | Not a replacement for all four endpoints. Its payload must be adapted and checkpoint compatibility checked. [The listed BLIP-2 API](https://replicate.com/andreasjansson/blip-2/readme) generates answers/captions, not the numeric ITC/ITM scores required here. |
| [Alibaba Cloud Bailian](https://help.aliyun.com/zh/model-studio/vision-model) | Ready-to-call image understanding / visual question answering API, useful for later method extensions. | A different model and interface: it does not preserve the four original endpoints' score/mask semantics. Treat such experiments separately from baseline reproduction. |
| [SiliconFlow vision APIs](https://docs.siliconflow.cn/docs/userguide/capabilities/vision) | Alternative ready-to-call visual understanding service with image URL/base64 inputs. | Select a currently available vision-capable model; visual tokens are billable. Not a numeric BLIP2 similarity endpoint. |
| [Roboflow managed inference](https://inference.roboflow.com/) | Candidate remote object detector to replace the local YOLO slot. [Its hosted API returns boxes, classes and confidence](https://roboflow.com/blog/object-detection-api). | Choose a specific model covering the HM3D target categories, then adapt pixel center/width/height boxes to normalized xyxy. Latency, model quality and access are not yet tested here. |

These providers are optional future alternatives, not requirements for the
current all-local deployment. Bailian or SiliconFlow would be a separate option
for large-model visual reasoning, not a detection-protocol replacement without
algorithm work. The checkout has not made real requests to any of these
providers; all four current endpoints point to localhost and pass preflight.

For the first remote deployment, a 24 GB RTX 4090 or L4 and at least 32 GB host
RAM are conservative recommendations based on the local model-loading tests,
not a measured minimum. Keep Habitat, ROS, and licensed scene assets on the
laptop; only inference images/text need to travel to the model service. Confirm
dataset terms and the provider's data policy before sending licensed imagery.

The local GroundingDINO extension was built for CUDA compute capability 8.9.
[NVIDIA lists the RTX 4090 and L4 in that same group](https://developer.nvidia.com/cuda/gpus).
Other GPU architectures require rebuilding the extension for the target GPU;
matching the architecture alone does not guarantee driver/runtime compatibility.
Provider instances may not allow Docker-in-Docker: the remote side only needs
the model environment and `python docker/runtime.py start`, not the full ROS or
Habitat stack. Do not upload the entire laptop image or scene dataset by default.

## HM3D data and one-episode reproduction

The recovered PSSD geometry/task data has been copied into the repository's
bind-mounted `data/` tree, so no extra container mount is needed. On 2026-10-26,
official v0.2 validation semantic annotations (2,145,935,360 bytes) and configs
(40,960 bytes) were downloaded and installed, preserving the existing geometry.
HM3Dv2 validation has 1,000 episodes in 36 scenes. All task-referenced geometry,
navmeshes, semantic meshes, semantic labels, and dataset config files are present.
This is the validation split, not a download of HM3D training/test data or MP3D.

```bash
docker/apexnav check-data
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/validate_habitat_dataset.py
docker/apexnav smoke
```

The strict check reports zero missing assets. The actual Habitat ObjectNav reset
also passed: 640x480 RGB, finite depth, 514 semantic objects in scene
`4ok3usBNeis`, and a finite goal geodesic distance after an action. The report is
`outputs/runtime/dataset-validation.json`; it is not a navigation evaluation
score. The official global config references uninstalled training/test scenes,
so Habitat may warn about those unused paths; this does not require downloading
other splits for validation.

To resume/reinstall these same authorized semantic components on this machine:

```bash
# Run on the Ubuntu host with Python 3.11+ and an HTTP/2-enabled curl.
python3 docker/download_hm3d_semantics.py
```

Enter the official Matterport token ID/secret at the hidden prompts. Credentials
are kept in memory and piped to curl, not saved in the repo or process arguments.
Completed/partial chunks are retained under `data/.downloads/` for resuming.
`hm3d-val-semantics-manifest.json` there records archive sizes and locally
computed SHA256 digests (not an independent official checksum comparison).
Existing files with different content are never overwritten. Access is governed
by [the official HM3D dataset terms](https://matterport.com/habitat-matterport-3d-research-dataset).
Expected layout includes:

```text
data/scene_datasets/hm3d_v0.2/
  hm3d_annotated_basis.scene_dataset_config.json
  val/00800-TEEsavR23oF/
    TEEsavR23oF.basis.glb
    TEEsavR23oF.basis.navmesh
    TEEsavR23oF.semantic.glb
    TEEsavR23oF.semantic.txt
```

After data and local model APIs pass preflight:

```bash
docker/apexnav run-one hm3dv2 0
```

This command checks assets and all four API inferences before starting the ROS
planner, runs one actual episode with cached LLM answers, and stops its planner
when finished. Logs/results go to a new `outputs/runs/` directory. The smoke test
and episode runner use ROS domain 79 and localhost discovery to isolate the
simulation; override `APEXNAV_ROS_DOMAIN_ID` when needed.

### Save a 50-episode experiment

On the Ubuntu host, from the repository root:

```bash
docker/apexnav vlm-start-local
docker/apexnav run-batch hm3dv2 --episodes 50 --video \
  --output outputs/runs/hm3dv2-local-50
```

This processes the first 50 validation episodes in the configured deterministic
iterator order, not a random representative sample. `test_epi_num=50` instead
selects a single episode index and must not be used as a count. The batch launcher
uses cached LLM answers, starts/stops its own planner, saves console output, and
only reports completion when 50 result records exist. It does not stop the model
services. Do not run another evaluator in ROS domain 79 at the same time.

Per-completed-episode outputs in the experiment directory:

| File | Contents |
| --- | --- |
| `episodes.csv` | Scene/episode IDs, target, steps, success, SPL, Soft SPL, final distance, failure category, time, and relative video path |
| `summary.json` | Completed/requested counts, success rate, mean metrics, and completion flag; rates and SPL are 0-1, not percentages |
| `console.log`, `planner.log` | Evaluation and ROS diagnostics |
| `config.yaml`, `experiment.json` | Effective configuration and batch settings |
| `success/*.mp4` or failure-category directories | Completed episode videos, 6 fps, maximum width 960 |
| `status.json` | Launcher completion/exit status |

CSV and JSON are updated after each completed episode. Videos are streamed to
FFmpeg as frames arrive rather than retaining up to 500 full frames in RAM.
An interrupted current episode may leave a `.partial.mp4`; it is not a completed
result and will be retried. Existing experiment directories are never silently
reused or overwritten. To continue an interrupted experiment with the same
settings:

```bash
docker/apexnav run-batch hm3dv2 --episodes 50 --video \
  --output outputs/runs/hm3dv2-local-50 --resume
```

Watch `console.log`, or open completed MP4 files in the Ubuntu file manager:

```bash
tail -f outputs/runs/hm3dv2-local-50/console.log
cat outputs/runs/hm3dv2-local-50/summary.json
xdg-open outputs/runs/hm3dv2-local-50
```

Keep the terminal open and the laptop awake/on power during evaluation. Close
unneeded memory-heavy apps first; all-local vision plus the full navigation
client/planner has less RAM/VRAM headroom than the earlier isolated tests. The
bounded counting, resume accounting, and streamed video have unit tests and the
CLI/config have been checked without starting a 50-episode navigation run.

On 2026-10-27, a real one-episode/three-step diagnostic also completed through
ROS, camera actions, model calls, CSV/JSON and MP4 output. Its deliberately low
step cap makes its navigation score meaningless; it is not part of the 50-episode
experiment. `--max-steps 3` exists for such diagnostics only; omit it for normal
evaluation (the current benchmark cap is 500 steps).

The first real look-down action exposed a Habitat-Lab/Sim interface mismatch:
Lab used `Agent.sensors`, but the pinned Sim commit stores its suite in
`Agent._sensors`. `patches/habitat-lab-agent-sensors.patch` preserves Lab's sensor
rotation behavior while supporting both accessors; `setup_container.sh` reapplies
it idempotently. The actual camera regression verifies RGB/depth pitch down/up
30 degrees and restoration without moving the body.

```bash
docker/apexnav run-batch hm3dv2 --episodes 50 --video --dry-run
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/test_batch.py
```

Remote API protocol regression tests (mock responses, no GPU inference):

```bash
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/test_remote_api.py
```

To recreate the environment in a compatible base container:

```bash
docker exec -it apexnav-jazzy-egl bash
cd /workspace/ApexNav
bash docker/setup_container.sh
```

The script is idempotent: repositories are pinned to known commits and patches
are applied only once. The build is intentionally headless because simulation
rendering uses EGL rather than an X display.

`pip check` reports LAVIS 1.0.2's optional demo/development dependencies
(`opendatasets`, `pre-commit`, and `streamlit`) plus its old OpenCV and
Transformers metadata pins. They are intentionally not forced into this runtime:
the older wheels do not match Python 3.12 or GroundingDINO, while the tested
LAVIS import, BLIP2 checkpoint load, and ITC/ITM GPU forwards all pass with the
versions installed here.

To create a new container from the saved image while mounting this repository:

```bash
bash docker/create_container.sh
```

Override `APEXNAV_CONTAINER` or `APEXNAV_IMAGE` if a different name or tag is
needed. The script preserves the tested GPU, host-network, 2 GB shared-memory,
and NVIDIA graphics capability settings.

## Save the configured image

```bash
docker/apexnav commit apexnav-jazzy:apexnav-v033-cu128
```

Model weights under `data/` are in the host bind mount and are therefore not
copied into the image. Keep that directory when moving or recreating the
container.

## Current storage

On 2026-10-26 the Ubuntu ext4 root partition (`nvme0n1p5`) was expanded to
344.8 GiB. Docker remains under `/var/lib/docker` on that expanded filesystem;
there is no separate `/data` mount and no data-root migration is required.

## Docker socket permission

If temporary ACL access was granted with `setfacl`, remove it after maintenance:

```bash
sudo setfacl -x "u:$USER" /var/run/docker.sock
```

Docker socket write access is effectively administrator access to the host.
