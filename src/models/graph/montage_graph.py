from __future__ import annotations

import re

import torch


_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "01": "O1", "02": "O2"}


def normalize_electrode(name: str) -> str:
    value = re.sub(r"[\s_.]", "", str(name).upper())
    return _ALIASES.get(value, value)


def bipolar_endpoints(channel: str) -> tuple[str, str]:
    parts = [normalize_electrode(part) for part in re.split(r"-+", str(channel).strip()) if part]
    if len(parts) != 2:
        raise ValueError(f"Expected a bipolar channel 'A-B', got {channel!r}")
    if parts[0] == parts[1]:
        raise ValueError(f"Bipolar channel endpoints must differ: {channel!r}")
    return parts[0], parts[1]


def build_signed_line_graph(channel_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return structural mask and orientation relation for bipolar derivations.

    Relation values are ``+1`` when two derivations use their shared electrode
    with the same coefficient, ``-1`` for opposite coefficients, ``+2`` on the
    diagonal, and ``0`` for non-neighbours.
    """

    endpoints = [bipolar_endpoints(name) for name in channel_names]
    channels = len(endpoints)
    relation = torch.zeros(channels, channels, dtype=torch.long)
    mask = torch.zeros(channels, channels, dtype=torch.bool)
    for i, (source_i, sink_i) in enumerate(endpoints):
        coefficient_i = {source_i: 1, sink_i: -1}
        for j, (source_j, sink_j) in enumerate(endpoints):
            if i == j:
                mask[i, j] = True
                relation[i, j] = 2
                continue
            coefficient_j = {source_j: 1, sink_j: -1}
            shared = set(coefficient_i).intersection(coefficient_j)
            if shared:
                mask[i, j] = True
                signed_sum = sum(coefficient_i[node] * coefficient_j[node] for node in shared)
                relation[i, j] = 1 if signed_sum >= 0 else -1
    return mask, relation
