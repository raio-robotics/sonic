"""ONNX Runtime wrappers for policy / encoder / planner models.

We use onnxruntime in place of the C++ TensorRT engine — the .onnx files are the
same artefacts that g1_deploy_onnx_ref compiles to a .trt cache, so output is
bit-equivalent on CPU and within FP tolerance on CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def _make_session(model_path: Path, use_cuda: bool = True) -> ort.InferenceSession:
    providers = []
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


class PolicyEngine:
    """model_decoder.onnx: obs_dict[1, 994] -> action[1, 29]."""

    def __init__(self, model_path: Path, use_cuda: bool = True) -> None:
        self.session = _make_session(model_path, use_cuda=use_cuda)
        ins = self.session.get_inputs()
        outs = self.session.get_outputs()
        assert len(ins) == 1, f"policy expects 1 input, got {len(ins)}"
        assert ins[0].name == "obs_dict", ins[0].name
        self.input_name = ins[0].name
        self.output_name = outs[0].name
        self.input_dim = int(ins[0].shape[1])
        self.output_dim = int(outs[0].shape[1])

    def infer(self, obs: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(obs, dtype=np.float32).reshape(1, self.input_dim)
        return self.session.run([self.output_name], {self.input_name: a})[0][0]


class EncoderEngine:
    """model_encoder.onnx: obs_dict[1, 1762] -> encoded_tokens[1, 64]."""

    def __init__(self, model_path: Path, use_cuda: bool = True) -> None:
        self.session = _make_session(model_path, use_cuda=use_cuda)
        ins = self.session.get_inputs()
        outs = self.session.get_outputs()
        self.input_name = ins[0].name
        self.output_name = outs[0].name
        self.input_dim = int(ins[0].shape[1])
        self.output_dim = int(outs[0].shape[1])

    def encode(self, obs: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(obs, dtype=np.float32).reshape(1, self.input_dim)
        return self.session.run([self.output_name], {self.input_name: a})[0][0]


class PlannerEngine:
    """planner_sonic.onnx (V2, 11 inputs).

    Outputs:
        mujoco_qpos       [1, 64, 36]   (only first num_pred_frames valid)
        num_pred_frames   [1] int32
    """

    def __init__(self, model_path: Path, use_cuda: bool = True) -> None:
        self.session = _make_session(model_path, use_cuda=use_cuda)
        self.input_names = [i.name for i in self.session.get_inputs()]
        required = {
            "context_mujoco_qpos", "mode", "target_vel", "movement_direction",
            "facing_direction", "random_seed", "height", "has_specific_target",
            "specific_target_positions", "specific_target_headings",
            "allowed_pred_num_tokens",
        }
        missing = required - set(self.input_names)
        assert not missing, f"planner missing inputs: {missing}"

    def run(
        self,
        context_qpos: np.ndarray,
        mode: int,
        target_vel: float,
        height: float,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        random_seed: int,
        has_specific_target: int = 0,
        specific_target_positions: np.ndarray | None = None,
        specific_target_headings: np.ndarray | None = None,
        allowed_pred_num_tokens: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        if specific_target_positions is None:
            specific_target_positions = np.zeros((4, 3), dtype=np.float32)
        if specific_target_headings is None:
            specific_target_headings = np.zeros(4, dtype=np.float32)
        if allowed_pred_num_tokens is None:
            # default mask from localmotion_kplanner_tensorrt.hpp:96-102
            allowed_pred_num_tokens = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64)
        feed = {
            "context_mujoco_qpos": np.ascontiguousarray(context_qpos, dtype=np.float32).reshape(1, 4, 36),
            "mode": np.array([mode], dtype=np.int64),
            "target_vel": np.array([target_vel], dtype=np.float32),
            "height": np.array([height], dtype=np.float32),
            "movement_direction": np.ascontiguousarray(movement_direction, dtype=np.float32).reshape(1, 3),
            "facing_direction": np.ascontiguousarray(facing_direction, dtype=np.float32).reshape(1, 3),
            "random_seed": np.array([random_seed], dtype=np.int64),
            "has_specific_target": np.array([[has_specific_target]], dtype=np.int64),
            "specific_target_positions": np.ascontiguousarray(specific_target_positions, dtype=np.float32).reshape(1, 4, 3),
            "specific_target_headings": np.ascontiguousarray(specific_target_headings, dtype=np.float32).reshape(1, 4),
            "allowed_pred_num_tokens": np.ascontiguousarray(allowed_pred_num_tokens, dtype=np.int64).reshape(1, 11),
        }
        qpos, n = self.session.run(["mujoco_qpos", "num_pred_frames"], feed)
        return qpos[0], int(n[0])
