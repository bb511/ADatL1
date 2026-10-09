"""Resolve '<object>.<feature>' references, with wildcard support.

A reference names one flattened L1 feature, e.g. ``FET.Et``. Either part may be
a shell-style pattern, so ``FET.*`` names every feature of the FET object. Object
and feature names are matched case-insensitively, as everywhere else in the
data pipeline.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable, Mapping


def split_feature_ref(feature_ref: str) -> tuple[str, str]:
    """Split ``'<object>.<feature>'`` into its two parts."""
    if "." not in feature_ref:
        raise ValueError(
            "Feature references must have format '<object>.<feature>' "
            f"(wildcards allowed, e.g. 'FET.*'), got {feature_ref!r}."
        )
    object_pattern, feature_pattern = feature_ref.split(".", maxsplit=1)
    return object_pattern, feature_pattern


def feature_ref_matches(object_name: str, feature_name: str, feature_ref: str) -> bool:
    """True if ``<object_name>.<feature_name>`` is named by ``feature_ref``."""
    object_pattern, feature_pattern = split_feature_ref(feature_ref)
    return fnmatchcase(str(object_name).lower(), object_pattern.lower()) and fnmatchcase(
        str(feature_name).lower(), feature_pattern.lower()
    )


def label_matches_any(label: str, feature_refs: Iterable[str]) -> bool:
    """True if the ``'<object>.<feature>'`` label is named by any reference."""
    object_name, feature_name = split_feature_ref(label)
    return any(
        feature_ref_matches(object_name, feature_name, feature_ref)
        for feature_ref in feature_refs
    )


def resolve_feature_refs(
    object_feature_map: Mapping[str, Mapping[str, Iterable[int]]],
    feature_refs: Iterable[str],
    strict: bool = True,
) -> list[tuple[str, str, list[int]]]:
    """Return ``(object_key, feature_key, indices)`` for every matched feature.

    :param strict: Raise ``KeyError`` when a reference matches nothing.
    """
    resolved: list[tuple[str, str, list[int]]] = []
    seen: set[tuple[str, str]] = set()

    for feature_ref in feature_refs:
        matched = False
        for object_key, feature_map in object_feature_map.items():
            for feature_key, indices in feature_map.items():
                if not feature_ref_matches(object_key, feature_key, feature_ref):
                    continue
                matched = True
                if (object_key, feature_key) in seen:
                    continue
                seen.add((object_key, feature_key))
                resolved.append(
                    (object_key, feature_key, [int(idx) for idx in indices])
                )

        if strict and not matched:
            available = sorted(
                f"{obj}.{feat}"
                for obj, feature_map in object_feature_map.items()
                for feat in feature_map
            )
            raise KeyError(
                f"Feature reference {feature_ref!r} matches no feature. "
                f"Available: {available}"
            )

    return resolved
