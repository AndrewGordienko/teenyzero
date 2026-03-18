from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    input_history_length: int
    piece_planes_per_position: int
    aux_planes: int
    model_version: int
    model_res_blocks: int
    model_channels: int
    policy_head_channels: int
    value_head_hidden: int
    replay_encoder_version: int
    replay_compress: bool
    min_samples_ready: int
    train_increment: int
    replay_window_samples: int
    train_samples_per_cycle: int
    bootstrap_window_samples: int
    max_retained_samples: int
    train_batch_size: int
    train_epochs_per_cycle: int
    train_poll_interval_s: float
    train_optimizer: str
    train_lr: float
    train_weight_decay: float
    train_momentum: float
    train_grad_accum_steps: int
    train_num_workers: int
    train_pin_memory: bool
    train_prefetch_factor: int
    train_compile: bool
    train_precision: str
    max_grad_norm: float
    selfplay_workers: int
    selfplay_simulations: int
    selfplay_leaf_batch_size: int
    arena_simulations: int
    arena_promotion_games: int
    arena_baseline_games: int
    arena_promotion_threshold: float
    inference_single_batch: int
    inference_merged_batch: int
    inference_wait_timeout: float
    inference_compile: bool
    inference_precision: str

    @property
    def input_planes(self):
        return self.input_history_length * self.piece_planes_per_position + self.aux_planes

    @property
    def input_shape(self):
        return (self.input_planes, 8, 8)

    def to_dict(self):
        payload = asdict(self)
        payload["input_planes"] = self.input_planes
        payload["input_shape"] = self.input_shape
        return payload


@dataclass(frozen=True)
class RuntimeSelection:
    device: str
    profile: RuntimeProfile
    requested_device: str
    requested_profile: str

    def to_dict(self):
        return {
            "device": self.device,
            "requested_device": self.requested_device,
            "requested_profile": self.requested_profile,
            "runtime_profile": self.profile.name,
            "runtime_profile_settings": self.profile.to_dict(),
        }


LOCAL_PROFILE = RuntimeProfile(
    name="local",
    input_history_length=4,
    piece_planes_per_position=12,
    aux_planes=8,
    model_version=2,
    model_res_blocks=12,
    model_channels=160,
    policy_head_channels=32,
    value_head_hidden=192,
    replay_encoder_version=2,
    replay_compress=True,
    min_samples_ready=20_000,
    train_increment=20_000,
    replay_window_samples=120_000,
    train_samples_per_cycle=25_000,
    bootstrap_window_samples=80_000,
    max_retained_samples=150_000,
    train_batch_size=64,
    train_epochs_per_cycle=1,
    train_poll_interval_s=10.0,
    train_optimizer="adamw",
    train_lr=1e-3,
    train_weight_decay=1e-4,
    train_momentum=0.9,
    train_grad_accum_steps=1,
    train_num_workers=0,
    train_pin_memory=False,
    train_prefetch_factor=2,
    train_compile=False,
    train_precision="fp32",
    max_grad_norm=0.0,
    selfplay_workers=4,
    selfplay_simulations=96,
    selfplay_leaf_batch_size=16,
    arena_simulations=96,
    arena_promotion_games=12,
    arena_baseline_games=4,
    arena_promotion_threshold=0.55,
    inference_single_batch=32,
    inference_merged_batch=32,
    inference_wait_timeout=0.0001,
    inference_compile=False,
    inference_precision="fp16",
)


MPS_PROFILE = RuntimeProfile(
    name="mps",
    input_history_length=8,
    piece_planes_per_position=12,
    aux_planes=8,
    model_version=4,
    model_res_blocks=16,
    model_channels=192,
    policy_head_channels=48,
    value_head_hidden=192,
    replay_encoder_version=4,
    replay_compress=True,
    min_samples_ready=40_000,
    train_increment=30_000,
    replay_window_samples=250_000,
    train_samples_per_cycle=80_000,
    bootstrap_window_samples=120_000,
    max_retained_samples=300_000,
    train_batch_size=96,
    train_epochs_per_cycle=1,
    train_poll_interval_s=5.0,
    train_optimizer="adamw",
    train_lr=2e-4,
    train_weight_decay=1e-4,
    train_momentum=0.9,
    train_grad_accum_steps=4,
    train_num_workers=4,
    train_pin_memory=False,
    train_prefetch_factor=2,
    train_compile=False,
    train_precision="fp16",
    max_grad_norm=2.0,
    selfplay_workers=6,
    selfplay_simulations=128,
    selfplay_leaf_batch_size=24,
    arena_simulations=192,
    arena_promotion_games=16,
    arena_baseline_games=6,
    arena_promotion_threshold=0.55,
    inference_single_batch=48,
    inference_merged_batch=96,
    inference_wait_timeout=0.0004,
    inference_compile=False,
    inference_precision="fp16",
)


MPS_FAST_PROFILE = RuntimeProfile(
    name="mps_fast",
    input_history_length=4,
    piece_planes_per_position=12,
    aux_planes=8,
    model_version=5,
    model_res_blocks=8,
    model_channels=128,
    policy_head_channels=24,
    value_head_hidden=128,
    replay_encoder_version=5,
    replay_compress=True,
    min_samples_ready=20_000,
    train_increment=20_000,
    replay_window_samples=120_000,
    train_samples_per_cycle=40_000,
    bootstrap_window_samples=80_000,
    max_retained_samples=180_000,
    train_batch_size=128,
    train_epochs_per_cycle=1,
    train_poll_interval_s=5.0,
    train_optimizer="adamw",
    train_lr=3e-4,
    train_weight_decay=1e-4,
    train_momentum=0.9,
    train_grad_accum_steps=2,
    train_num_workers=2,
    train_pin_memory=False,
    train_prefetch_factor=2,
    train_compile=False,
    train_precision="fp16",
    max_grad_norm=2.0,
    selfplay_workers=6,
    selfplay_simulations=48,
    selfplay_leaf_batch_size=32,
    arena_simulations=96,
    arena_promotion_games=12,
    arena_baseline_games=4,
    arena_promotion_threshold=0.55,
    inference_single_batch=64,
    inference_merged_batch=128,
    inference_wait_timeout=0.0002,
    inference_compile=False,
    inference_precision="fp16",
)


H100_PROFILE = RuntimeProfile(
    name="h100",
    input_history_length=8,
    piece_planes_per_position=12,
    aux_planes=8,
    model_version=3,
    model_res_blocks=20,
    model_channels=256,
    policy_head_channels=48,
    value_head_hidden=256,
    replay_encoder_version=3,
    replay_compress=False,
    min_samples_ready=150_000,
    train_increment=100_000,
    replay_window_samples=1_500_000,
    train_samples_per_cycle=250_000,
    bootstrap_window_samples=400_000,
    max_retained_samples=3_000_000,
    train_batch_size=256,
    train_epochs_per_cycle=1,
    train_poll_interval_s=5.0,
    train_optimizer="sgd",
    train_lr=0.2,
    train_weight_decay=1e-4,
    train_momentum=0.9,
    train_grad_accum_steps=8,
    train_num_workers=8,
    train_pin_memory=True,
    train_prefetch_factor=4,
    train_compile=True,
    train_precision="bf16",
    max_grad_norm=2.0,
    selfplay_workers=24,
    selfplay_simulations=256,
    selfplay_leaf_batch_size=48,
    arena_simulations=384,
    arena_promotion_games=24,
    arena_baseline_games=8,
    arena_promotion_threshold=0.55,
    inference_single_batch=96,
    inference_merged_batch=192,
    inference_wait_timeout=0.0004,
    inference_compile=True,
    inference_precision="bf16",
)


H200_PROFILE = RuntimeProfile(
    name="h200",
    input_history_length=8,
    piece_planes_per_position=12,
    aux_planes=8,
    model_version=4,
    model_res_blocks=24,
    model_channels=320,
    policy_head_channels=64,
    value_head_hidden=320,
    replay_encoder_version=4,
    replay_compress=False,
    min_samples_ready=200_000,
    train_increment=150_000,
    replay_window_samples=2_500_000,
    train_samples_per_cycle=400_000,
    bootstrap_window_samples=600_000,
    max_retained_samples=5_000_000,
    train_batch_size=320,
    train_epochs_per_cycle=1,
    train_poll_interval_s=5.0,
    train_optimizer="sgd",
    train_lr=0.2,
    train_weight_decay=1e-4,
    train_momentum=0.9,
    train_grad_accum_steps=8,
    train_num_workers=10,
    train_pin_memory=True,
    train_prefetch_factor=4,
    train_compile=True,
    train_precision="bf16",
    max_grad_norm=2.0,
    selfplay_workers=32,
    selfplay_simulations=320,
    selfplay_leaf_batch_size=64,
    arena_simulations=512,
    arena_promotion_games=28,
    arena_baseline_games=10,
    arena_promotion_threshold=0.55,
    inference_single_batch=128,
    inference_merged_batch=256,
    inference_wait_timeout=0.0006,
    inference_compile=True,
    inference_precision="bf16",
)


PROFILES = {
    LOCAL_PROFILE.name: LOCAL_PROFILE,
    MPS_PROFILE.name: MPS_PROFILE,
    MPS_FAST_PROFILE.name: MPS_FAST_PROFILE,
    H100_PROFILE.name: H100_PROFILE,
    H200_PROFILE.name: H200_PROFILE,
}

DEVICE_ALIASES = {
    "": "auto",
    "auto": "auto",
    "cpu": "cpu",
    "mps": "mps",
    "cuda": "cuda",
    "gpu": "cuda",
    "h100": "cuda",
    "h200": "cuda",
}


def normalize_device_name(device_name):
    return DEVICE_ALIASES.get((device_name or "").strip().lower(), "auto")


def requested_device_name():
    return normalize_device_name(os.environ.get("TEENYZERO_DEVICE", "auto"))


def requested_profile_name():
    return os.environ.get("TEENYZERO_PROFILE", "").strip().lower()


def device_available(device_name):
    if device_name == "cuda":
        return bool(torch.cuda.is_available())
    if device_name == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return device_name == "cpu"


def _auto_device_name():
    if device_available("cuda"):
        return "cuda"
    if device_available("mps"):
        return "mps"
    return "cpu"


def active_device_name():
    requested = requested_device_name()
    if requested != "auto" and device_available(requested):
        return requested
    return _auto_device_name()


def _device_profile_name(device_name):
    if device_name == "mps":
        return MPS_PROFILE.name
    if device_name != "cuda":
        return LOCAL_PROFILE.name

    try:
        device_name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return LOCAL_PROFILE.name

    if "h200" in device_name:
        return H200_PROFILE.name
    if "h100" in device_name:
        return H100_PROFILE.name
    return LOCAL_PROFILE.name


def active_profile_name():
    explicit = requested_profile_name()
    if explicit:
        return explicit
    return _device_profile_name(active_device_name())


def get_runtime_profile():
    return PROFILES.get(active_profile_name(), LOCAL_PROFILE)


def get_runtime_selection():
    device = active_device_name()
    profile = PROFILES.get(active_profile_name(), LOCAL_PROFILE)
    return RuntimeSelection(
        device=device,
        profile=profile,
        requested_device=requested_device_name(),
        requested_profile=requested_profile_name(),
    )


def runtime_profile_payload(profile=None):
    selection = get_runtime_selection()
    active = profile or selection.profile
    payload = active.to_dict()
    payload["device"] = selection.device
    payload["requested_device"] = selection.requested_device
    payload["requested_profile"] = selection.requested_profile
    return payload
