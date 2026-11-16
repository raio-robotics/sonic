# sonic

[raio](https://github.com/raio-robotics/raio) port of
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
`gear_sonic_deploy`, driving a Unitree G1 (sim or real) over DDS.

## Setup

Requires [conda](https://docs.conda.io/). For GPU inference: JetPack 7 (aarch64) or x86 + CUDA 12+; otherwise CPU works.

`sonic` is run as scripts from the repo root — it is **not** an installable
package. Dependencies are listed in `environment.yml`.

### 1. Create the conda environment

```bash
# from the repo root
conda env create -f environment.yml
conda activate sonic
```

This installs Python 3.12, [raio](https://github.com/raio-robotics/raio)
(`raio-py` on PyPI) — the deterministic realtime runtime the nodes run on —
numpy, onnxruntime (CPU), PyYAML, huggingface_hub, and the simulator deps
(mujoco, tyro, opencv-python, msgpack, msgpack-numpy, pin, pyzmq, scipy).
See the [raio repository](https://github.com/raio-robotics/raio) for other
ways to install raio, such as building it from source.

### 2. Install the Unitree SDK

`unitree_sdk2_python` is not on PyPI, and it depends on `cyclonedds==0.10.2`,
whose source build needs the matching **CycloneDDS C library v0.10.2**
installed locally (newer versions like 0.11+ removed headers the Python
bindings still reference, e.g. `dds/ddsi/q_radmin.h`).

Build CycloneDDS 0.10.2 first — see the upstream
[installation guide](https://cyclonedds.io/docs/cyclonedds/latest/installation/installation.html),
but make sure you check out the `0.10.2` tag:

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
git checkout tags/0.10.2
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/cyclone_dds_10_2 -DBUILD_EXAMPLES=OFF ..
cmake --build . --parallel
cmake --build . --target install
```

Then install the SDK, pointing the build at that prefix:

```bash
CYCLONEDDS_HOME="$HOME/cyclone_dds_10_2" \
  pip install git+https://github.com/unitreerobotics/unitree_sdk2_python
```

### 3. (Optional) Swap onnxruntime → onnxruntime-gpu

For GPU on JetPack 7 use the cp312 / aarch64 wheel from NVIDIA's Jetson AI
Lab index:

```bash
pip uninstall -y onnxruntime
pip install --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/ \
    onnxruntime-gpu
```

On x86 with CUDA 12+, plain `pip install onnxruntime-gpu` from PyPI works.
Without a supported GPU, stick with the CPU build and run sonic with
`--no-cuda`.

### 4. Verify

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# GPU: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# CPU: ['CPUExecutionProvider']

python -c "import mujoco, tyro, cv2, msgpack, pinocchio, raio, unitree_sdk2py; print('deps OK')"
```

### 5. Download checkpoints + assets

The repo ships node/util code only — the ONNX checkpoints, observation
config, robot model, and reference motion library are not committed.
`download.py` fetches them from the right places:

```bash
python download.py
```

This pulls:

- `model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`,
  `observation_config.yaml` from Hugging Face
  ([`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC))
- the G1 URDF + meshes + scene XML, and the reference motion CSV library,
  from GitHub
  ([`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl))
  via a shallow sparse clone

After it finishes you should have:

```
checkpoints/
├── model_encoder.onnx
├── model_decoder.onnx
└── planner_sonic.onnx
assets/
├── observation_config.yaml
├── reference/example/...        # reference motion CSVs
└── robot/                       # G1 URDF + meshes + scene_43dof.xml
```

## Run

Sim and sonic are two separate programs that talk over DDS (`/rt/lowstate`,
`/rt/secondary_imu`, `/rt/lowcmd`). Start them in two terminals, sim first,
with the environment active in each:

```bash
conda activate sonic
```

### 1. Start the MuJoCo sim

The vendored `gear_sonic` sim publishes `/rt/lowstate` + `/rt/secondary_imu`
and subscribes to `/rt/lowcmd`:

```bash
python sim/run_sim_loop.py
```

Defaults to DDS interface `lo`, domain 0. For a real Unitree G1, skip this
step and point sonic's `--interface` at the robot's network interface
instead. `--no-enable-onscreen` runs the sim without a viewer.

#### macOS note

MuJoCo's passive viewer requires the `mjpython` launcher on macOS (Cocoa
must own the main thread), and macOS's loopback interface is `lo0`, not `lo`
(the default `lo` fails DDS init with `does not match an available
interface`). Use `--interface lo0` for both sim and sonic:

```bash
mjpython sim/run_sim_loop.py --interface lo0
```

conda keeps `libpython3.12.dylib` in `$CONDA_PREFIX/lib`, where `mjpython`
looks for it, so no extra setup is needed.

### 2. Start sonic

In a second terminal:

```bash
python main.py                    # Linux
python main.py --interface lo0    # macOS
```

Each node runs in its own process. Control and Planner load their ONNX
models in their own processes before the app starts, so every node's
logical clock starts once all models are loaded. Each run also writes the
node graph to `diagram.svg`.

Pipeline on startup:

1. `INIT` (3 s) — ramp motor `q` from the first LowState reading toward
   `default_angles`.
2. `WAIT_FOR_CONTROL` — hold `default_angles` via PD; wait for the planner
   to produce its first IDLE/standing trajectory.
3. `CONTROL` — 50 Hz encoder + policy emitting `motor_command`, published
   to `/rt/lowcmd` at 500 Hz.

### Node graph

| Node | Rate | Role |
|---|---|---|
| `DDSReaderNode` | per DDS sample | `/rt/lowstate`, `/rt/secondary_imu` → `low_state`, `imu_torso` |
| `DefaultInputNode` | 100 Hz | default operator / movement commands (IDLE) |
| `PlannerNode` | 10 Hz | kinematic planner → `planner_motion` |
| `ControlNode` | 50 Hz | encoder + policy → `motor_command` |
| `DDSWriterNode` | 500 Hz | latest `motor_command` → `/rt/lowcmd` |
| `Keyboard` | per keypress | terminal keys → `key` |

`DDSReaderNode` and `Keyboard` receive their values from threads through
raio actions. The keyboard connection is physical: each key is stamped with
`ControlNode`'s clock when it arrives, and control never waits for a keypress.
The DDS connections are logical, so every receiver sees a sample at the same
timestamp; they wait only for the first sample. Every other connection is logical too, so a
control tick's motor command is published at that tick, right after
`ControlNode` computes it. `planner_motion` has a delay of one planner
period (100 ms), so a planner inference never stalls the control loop.

### Keyboard controls

With the sonic terminal focused:

- **T** — play the currently-selected reference motion once
- **N / P** — cycle to next / previous reference motion
- **Esc** — shut sonic down cleanly

### Common CLI flags

```
--interface STR     DDS network interface (default: lo)
--domain-id INT     DDS domain ID (default: 0)
--walk-after FLOAT  seconds after start to switch from IDLE to WALK (default: -1, stay IDLE)
--no-cuda           force CPU ONNX execution
```

## Scripts

```bash
python scripts/test_dds_writer.py [--interface lo0]   # DDSWriterNode publishes /rt/lowcmd
python scripts/test_dds_reader.py [--interface lo0]   # DDSReaderNode reads the running sim
python scripts/watch_robot.py [seconds]               # live tilt / joint monitor (lo)
python scripts/inspect_lowcmd.py                      # dump published LowCmd messages (lo)
```
