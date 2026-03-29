from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from teenyzero.alphazero.runtime import RuntimeProfile
from teenyzero.autotune.core.common import (
    baseline_actor_mode,
    build_apply_command,
    compile_options,
    cpu_count,
    pin_memory_options,
    precision_options,
    unique_int_candidates,
)
from teenyzero.autotune.phases.phase1 import Phase1Config, phase1_trial_score
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs, save_autotune_run


@dataclass(frozen=True)
class Phase2RoundPlan:
    index: int
    label: str
    searches_per_worker: int
    selfplay_simulations: int
    train_batches: int
    trial_timeout_s: float
    resource_scale: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "searches_per_worker": self.searches_per_worker,
            "selfplay_simulations": self.selfplay_simulations,
            "train_batches": self.train_batches,
            "trial_timeout_s": self.trial_timeout_s,
            "resource_scale": self.resource_scale,
        }


def _top_successful_trials(seed_run: dict | None, limit: int = 6) -> list[dict]:
    if not seed_run:
        return []
    ranked = []
    for trial in seed_run.get("trials", []):
        if trial.get("status") != "ok":
            continue
        ranked.append(trial)
    ranked.sort(key=lambda item: float(item.get("score", -1.0)), reverse=True)
    return ranked[: max(1, int(limit))]


def phase2_seed_run(seed_run: dict | None = None) -> dict | None:
    if seed_run:
        return seed_run
    phase1_run = latest_autotune_run(phase="phase1")
    if phase1_run:
        return phase1_run
    return latest_autotune_run()


def phase2_seed_trial(seed_run: dict | None) -> dict | None:
    if not seed_run:
        return None
    best_trial = seed_run.get("best_trial") or {}
    if best_trial.get("status") == "ok":
        return best_trial
    for trial in seed_run.get("trials", []):
        if trial.get("status") == "ok":
            return trial
    return None


def _seed_config(profile: RuntimeProfile, device: str, seed_trial: dict | None) -> dict:
    if seed_trial and (seed_trial.get("config") or {}):
        return dict(seed_trial["config"])
    baseline = Phase1Config(
        actor_mode=baseline_actor_mode(device),
        selfplay_workers=max(1, int(profile.selfplay_workers)),
        selfplay_leaf_batch_size=max(1, int(profile.selfplay_leaf_batch_size)),
        train_batch_size=max(1, int(profile.train_batch_size)),
        train_num_workers=max(0, int(profile.train_num_workers)),
        train_pin_memory=bool(profile.train_pin_memory if device == "cuda" else False),
        train_precision=str(profile.train_precision),
        train_compile=bool(profile.train_compile if device == "cuda" else False),
    )
    return baseline.to_dict()


def _candidate_caps(profile: RuntimeProfile, seed_config: dict, top_trials: list[dict]) -> dict:
    cpu_limit = cpu_count()
    observed_workers = [int((trial.get("config") or {}).get("selfplay_workers", 0)) for trial in top_trials]
    observed_leaf = [int((trial.get("config") or {}).get("selfplay_leaf_batch_size", 0)) for trial in top_trials]
    observed_batch = [int((trial.get("config") or {}).get("train_batch_size", 0)) for trial in top_trials]
    observed_train_workers = [int((trial.get("config") or {}).get("train_num_workers", 0)) for trial in top_trials]

    return {
        "workers": max(2, min(cpu_limit, max([int(seed_config["selfplay_workers"]), int(profile.selfplay_workers), *observed_workers]) * 2)),
        "leaf": max(8, max([int(seed_config["selfplay_leaf_batch_size"]), int(profile.selfplay_leaf_batch_size), *observed_leaf]) * 2),
        "batch": min(768, max(16, max([int(seed_config["train_batch_size"]), int(profile.train_batch_size), *observed_batch]) * 2)),
        "train_workers": max(0, min(cpu_limit, max([int(seed_config["train_num_workers"]), int(profile.train_num_workers), *observed_train_workers]) + 4)),
    }


def _candidate_value_sets(profile: RuntimeProfile, device: str, seed_config: dict, top_trials: list[dict]) -> dict:
    caps = _candidate_caps(profile, seed_config, top_trials)
    top_configs = [trial.get("config") or {} for trial in top_trials]

    def collect(key: str, values: list[int]) -> list[int]:
        observed = [int(config.get(key, 0)) for config in top_configs if key in config]
        merged = list(values) + observed
        unique = []
        for value in merged:
            value = int(value)
            if value not in unique:
                unique.append(value)
        return sorted(unique)

    actor_modes = []
    for value in [seed_config["actor_mode"], *(config.get("actor_mode") for config in top_configs)]:
        if value and value not in actor_modes:
            actor_modes.append(value)
    if baseline_actor_mode(device) not in actor_modes:
        actor_modes.append(baseline_actor_mode(device))

    precisions = []
    for value in [seed_config["train_precision"], *(config.get("train_precision") for config in top_configs)]:
        if value and value not in precisions:
            precisions.append(value)
    for value in precision_options(profile, device):
        if value not in precisions:
            precisions.append(value)

    compile_values = []
    for value in [seed_config["train_compile"], *(config.get("train_compile") for config in top_configs)]:
        if value not in compile_values:
            compile_values.append(bool(value))
    for value in compile_options(profile, device):
        if value not in compile_values:
            compile_values.append(value)

    pin_memory_values = []
    for value in [seed_config["train_pin_memory"], *(config.get("train_pin_memory") for config in top_configs)]:
        if value not in pin_memory_values:
            pin_memory_values.append(bool(value))
    for value in pin_memory_options(profile, device):
        if value not in pin_memory_values:
            pin_memory_values.append(value)

    return {
        "actor_mode": actor_modes,
        "selfplay_workers": collect(
            "selfplay_workers",
            unique_int_candidates(int(seed_config["selfplay_workers"]), minimum=1, maximum=caps["workers"]),
        ),
        "selfplay_leaf_batch_size": collect(
            "selfplay_leaf_batch_size",
            unique_int_candidates(int(seed_config["selfplay_leaf_batch_size"]), minimum=4, maximum=caps["leaf"]),
        ),
        "train_batch_size": collect(
            "train_batch_size",
            unique_int_candidates(int(seed_config["train_batch_size"]), minimum=8, maximum=caps["batch"]),
        ),
        "train_num_workers": collect(
            "train_num_workers",
            unique_int_candidates(int(seed_config["train_num_workers"]), minimum=0, maximum=caps["train_workers"]),
        ),
        "train_pin_memory": pin_memory_values,
        "train_precision": precisions,
        "train_compile": compile_values,
    }


def _unique_configs(configs: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for config in configs:
        key = tuple(sorted(dict(config).items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(config))
    return unique


def build_phase2_candidates(
    profile: RuntimeProfile,
    device: str,
    trial_count: int,
    seed_run: dict | None = None,
    seed: int = 0,
) -> list[dict]:
    seed_payload = phase2_seed_run(seed_run)
    seed_trial = phase2_seed_trial(seed_payload)
    top_trials = _top_successful_trials(seed_payload, limit=6)
    seed_config = _seed_config(profile, device, seed_trial)
    value_sets = _candidate_value_sets(profile, device, seed_config, top_trials)

    prioritized = [seed_config]
    prioritized.extend(dict(trial.get("config") or {}) for trial in top_trials if trial.get("config"))

    numeric_keys = (
        "selfplay_workers",
        "selfplay_leaf_batch_size",
        "train_batch_size",
        "train_num_workers",
    )
    for key in numeric_keys:
        for value in value_sets[key]:
            if int(value) == int(seed_config[key]):
                continue
            candidate = dict(seed_config)
            candidate[key] = int(value)
            prioritized.append(candidate)

    for key in ("actor_mode", "train_precision", "train_compile", "train_pin_memory"):
        for value in value_sets[key]:
            if value == seed_config[key]:
                continue
            candidate = dict(seed_config)
            candidate[key] = value
            prioritized.append(candidate)

    for workers, leaf, batch, train_workers in itertools.product(
        value_sets["selfplay_workers"],
        value_sets["selfplay_leaf_batch_size"],
        value_sets["train_batch_size"],
        value_sets["train_num_workers"],
    ):
        prioritized.append({
            "actor_mode": seed_config["actor_mode"],
            "selfplay_workers": int(workers),
            "selfplay_leaf_batch_size": int(leaf),
            "train_batch_size": int(batch),
            "train_num_workers": int(train_workers),
            "train_pin_memory": bool(seed_config["train_pin_memory"]),
            "train_precision": seed_config["train_precision"],
            "train_compile": bool(seed_config["train_compile"]),
        })

    combos = list(itertools.product(
        value_sets["actor_mode"],
        value_sets["selfplay_workers"],
        value_sets["selfplay_leaf_batch_size"],
        value_sets["train_batch_size"],
        value_sets["train_num_workers"],
        value_sets["train_pin_memory"],
        value_sets["train_precision"],
        value_sets["train_compile"],
    ))
    random.Random(seed).shuffle(combos)
    for item in combos:
        prioritized.append({
            "actor_mode": item[0],
            "selfplay_workers": int(item[1]),
            "selfplay_leaf_batch_size": int(item[2]),
            "train_batch_size": int(item[3]),
            "train_num_workers": int(item[4]),
            "train_pin_memory": bool(item[5]),
            "train_precision": str(item[6]),
            "train_compile": bool(item[7]),
        })

    selected = _unique_configs(prioritized)[: max(1, int(trial_count))]
    return [
        {
            "index": index,
            "candidate_id": f"C{index + 1:02d}",
            "label": f"C{index + 1:02d}",
            "config": config,
            "is_seed": index == 0,
            "trial_count": len(selected),
        }
        for index, config in enumerate(selected)
    ]


def build_phase2_round_plans(
    searches_per_worker: int,
    selfplay_simulations: int,
    train_batches: int,
    trial_timeout_s: float,
    rounds: int,
) -> list[dict]:
    round_count = max(2, int(rounds))
    plans = []
    for index in range(round_count):
        scale = min(4.0, 0.5 * (2 ** index))
        plans.append(
            Phase2RoundPlan(
                index=index,
                label=f"R{index + 1}",
                searches_per_worker=max(2, int(round(searches_per_worker * scale))),
                selfplay_simulations=max(1, int(selfplay_simulations)),
                train_batches=max(2, int(round(train_batches * scale))),
                trial_timeout_s=max(60.0, float(trial_timeout_s) * max(1.0, scale)),
                resource_scale=float(scale),
            ).to_dict()
        )
    return plans


def phase2_trial_score(trial: dict, reference_trial: dict | None, objective: str) -> float:
    if reference_trial:
        return phase1_trial_score(trial, reference_trial, objective)
    if trial.get("status") != "ok":
        return -1.0
    selfplay = trial.get("selfplay", {})
    train = trial.get("train", {})
    move_ms = float(selfplay.get("move_total_mean_ms") or 0.0)
    latency_score = 1000.0 / move_ms if move_ms > 0 else 0.0
    if objective == "selfplay":
        return (
            (0.65 * float(selfplay.get("positions_per_s") or 0.0)) +
            (0.20 * float(selfplay.get("searches_per_s") or 0.0)) +
            (0.15 * latency_score)
        )
    if objective == "train":
        return float(train.get("samples_per_s") or 0.0)
    return (
        (0.40 * float(selfplay.get("positions_per_s") or 0.0)) +
        (0.15 * float(selfplay.get("searches_per_s") or 0.0)) +
        (0.20 * latency_score) +
        (0.25 * float(train.get("samples_per_s") or 0.0))
    )


def phase2_survivor_count(candidate_count: int, round_index: int, round_count: int, halving_ratio: float) -> int:
    if round_index >= round_count - 1:
        return 1
    ratio = max(1.25, float(halving_ratio))
    return max(1, int(math.ceil(float(candidate_count) / ratio)))


def finalize_phase2_run(run_payload: dict) -> dict:
    objective = str(run_payload.get("objective", "balanced"))
    reference_trial = phase2_seed_trial(run_payload.get("seed_run"))
    rounds = []
    best_overall = None
    last_completed = None

    for round_payload in list(run_payload.get("rounds", [])):
        trials = list(round_payload.get("trials", []))
        for trial in trials:
            trial["score"] = phase2_trial_score(trial, reference_trial, objective)
        ranked = sorted(trials, key=lambda item: float(item.get("score", -1.0)), reverse=True)
        round_payload["trials"] = ranked
        round_payload["best_trial"] = ranked[0] if ranked else None
        round_payload["successful_trials"] = sum(1 for trial in ranked if trial.get("status") == "ok")
        rounds.append(round_payload)
        if ranked:
            last_completed = round_payload
            contender = ranked[0]
            if contender.get("status") == "ok":
                if best_overall is None or float(contender.get("score", -1.0)) > float(best_overall.get("score", -1.0)):
                    best_overall = contender

    run_payload["rounds"] = rounds
    run_payload["trials"] = list((last_completed or {}).get("trials", []))
    run_payload["best_trial"] = (last_completed or {}).get("best_trial") or best_overall
    run_payload["round_count"] = len(rounds)
    if run_payload.get("best_trial"):
        run_payload["apply_command"] = build_apply_command(run_payload["runtime_args"], run_payload["best_trial"]["config"])
    return run_payload


def save_phase2_run(run_payload: dict, archive: bool = False):
    finalized = finalize_phase2_run(dict(run_payload))
    return save_autotune_run(finalized, archive=archive)


def latest_phase2_run() -> dict | None:
    return latest_autotune_run(phase="phase2")


def list_phase2_runs(limit: int = 8) -> list[dict]:
    return list_autotune_runs(limit=limit, phase="phase2")
