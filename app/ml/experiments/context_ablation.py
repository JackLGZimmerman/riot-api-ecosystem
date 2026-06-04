"""Three-encoder sidecar and semantic-context HGNN ablation variants.

This registry keeps the experiment surface explicit: each variant maps to
`HGNNConfig` overrides consumed by `app.ml.train.train(..., model_overrides=...)`.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Any

SIDECAR_VARIANTS: dict[str, dict[str, Any]] = {
    "static_only": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": False,
    },
    "full_game_only": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": False,
    },
    "temporal_only": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": True,
    },
    "static_full_game": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": False,
    },
    "static_temporal": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": True,
    },
    "full_game_temporal": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
    },
    "all_three": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
    },
    "semantic_context_only": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": False,
        "use_identity_semantic_context_head": True,
    },
    "learned_semantic_moe_only": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": False,
        "use_learned_semantic_moe": True,
    },
    "learned_semantic_moe_group_features_only": {
        "use_identity_static_sidecar": False,
        "use_identity_full_game_sidecar": False,
        "use_identity_temporal_sidecar": False,
        "use_learned_semantic_moe": True,
        "use_semantic_group_features": True,
    },
    "all_three_plus_semantic_context": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
        "use_identity_semantic_context_head": True,
    },
    "all_three_plus_learned_semantic_moe": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
        "use_learned_semantic_moe": True,
    },
    "all_three_plus_learned_semantic_moe_group_features": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
        "use_learned_semantic_moe": True,
        "use_semantic_group_features": True,
    },
    "all_three_plus_raw_context": {
        "use_identity_static_sidecar": True,
        "use_identity_full_game_sidecar": True,
        "use_identity_temporal_sidecar": True,
        "use_relationship_integrations": True,
    },
}


def sidecar_variant_overrides(name: str) -> dict[str, Any]:
    try:
        return deepcopy(SIDECAR_VARIANTS[name])
    except KeyError as exc:
        known = ", ".join(sorted(SIDECAR_VARIANTS))
        raise ValueError(f"unknown sidecar variant {name!r}; expected one of: {known}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        nargs="?",
        choices=sorted(SIDECAR_VARIANTS),
        help="Print overrides for one variant. Omit to print the full registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = (
        sidecar_variant_overrides(args.variant)
        if args.variant is not None
        else SIDECAR_VARIANTS
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
