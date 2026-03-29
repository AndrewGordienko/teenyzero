from teenyzero.autotune.core.common import build_apply_command
from teenyzero.autotune.phases.phase1 import (
    build_phase1_candidates,
    finalize_phase1_run,
    hardware_fingerprint,
    latest_phase1_run,
    list_phase1_runs,
    phase1_trial_score,
    save_phase1_run,
)
from teenyzero.autotune.phases.phase2 import (
    build_phase2_candidates,
    build_phase2_round_plans,
    finalize_phase2_run,
    latest_phase2_run,
    list_phase2_runs,
    phase2_trial_score,
    save_phase2_run,
)
from teenyzero.autotune.catalog.recommendations import (
    AUTOTUNE_RESULTS_DOC_PATH,
    RECOMMENDATIONS_PATH,
    build_recommendation_entry,
    load_recommendations,
    promote_latest_autotune_run,
    promote_latest_phase1_run,
    recommendation_device_family,
    recommendation_title,
    save_recommendations,
    upsert_recommendation,
    write_recommendations_markdown,
)
from teenyzero.autotune.core.storage import latest_autotune_run, list_autotune_runs, save_autotune_run

try:
    from teenyzero.autotune.phases.phase3 import (
        build_phase3_candidates,
        finalize_phase3_run,
        latest_phase3_run,
        list_phase3_runs,
        phase3_seed_run,
        phase3_trial_score,
        prepare_phase3_datasets,
        prepare_phase3_replay_source,
        run_phase3_trial,
        save_phase3_run,
    )
    _HAS_PHASE3 = True
except Exception:
    _HAS_PHASE3 = False

__all__ = [
    "build_apply_command",
    "build_phase1_candidates",
    "build_phase2_candidates",
    "build_phase2_round_plans",
    "finalize_phase1_run",
    "finalize_phase2_run",
    "hardware_fingerprint",
    "latest_autotune_run",
    "latest_phase1_run",
    "latest_phase2_run",
    "list_autotune_runs",
    "list_phase1_runs",
    "list_phase2_runs",
    "phase1_trial_score",
    "phase2_trial_score",
    "save_autotune_run",
    "save_phase1_run",
    "save_phase2_run",
    "AUTOTUNE_RESULTS_DOC_PATH",
    "RECOMMENDATIONS_PATH",
    "build_recommendation_entry",
    "load_recommendations",
    "promote_latest_autotune_run",
    "promote_latest_phase1_run",
    "recommendation_device_family",
    "recommendation_title",
    "save_recommendations",
    "upsert_recommendation",
    "write_recommendations_markdown",
]

if _HAS_PHASE3:
    __all__.extend(
        [
            "build_phase3_candidates",
            "finalize_phase3_run",
            "latest_phase3_run",
            "list_phase3_runs",
            "phase3_seed_run",
            "phase3_trial_score",
            "prepare_phase3_datasets",
            "prepare_phase3_replay_source",
            "run_phase3_trial",
            "save_phase3_run",
        ]
    )
