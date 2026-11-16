"""raio port of GR00T-WholeBodyControl gear_sonic_deploy.

Mirrors deploy.sh's default invocation (sim mode):
    just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \\
        --obs-config policy/release/observation_config.yaml \\
        --encoder-file policy/release/model_encoder.onnx \\
        --planner-file planner/target_vel/V2/planner_sonic.onnx \\
        --input-type manager --output-type all --disable-crc-check

Differences from the C++:
  - Inter-node communication uses raio ports (not ZMQ).
  - Keyboard / gamepad / ROS2 / ZMQ input handlers replaced by a default
    action sequence (IDLE locomotion mode).
  - ZMQ / ROS2 telemetry publishers removed.
  - Audio TTS, Dex3 hands, state CSV logger, error monitor, VR streaming removed.
  - ONNX Runtime replaces TensorRT (same .onnx models).
"""
