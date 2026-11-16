"""Entry point — runs the vendored gear_sonic MuJoCo sim loop.

The gear_sonic package lives under ./gear_sonic/ exactly as it does in the
GR00T-WholeBodyControl repo, so its internal imports
(`gear_sonic.utils.mujoco_sim...`, `gear_sonic.data.robot_model...`) resolve
without modification. This wrapper just ensures sim/ is on sys.path before
delegating to the original entry point.

The robot model data (XMLs + meshes) was relocated to sonic/assets/robot/
when vendoring; both the YAML's `ROBOT_SCENE` and `instantiation/g1.py`
resolve paths against that location, so the simulator can be launched from
anywhere:

    python sim/run_sim_loop.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

if __name__ == "__main__":
    import tyro

    from gear_sonic.scripts.run_sim_loop import main, ArgsConfig

    main(tyro.cli(ArgsConfig))
