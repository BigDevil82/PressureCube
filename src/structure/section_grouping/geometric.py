from __future__ import annotations

import numpy as np

from src.structure.enums import BeamType
from src.structure.section_grouping.features import beam_assignment_vector, make_geometry_context
from src.structure.structural_model import StructuralModel


def assign_geometric_beam_groups(
    model: StructuralModel,
    n_groups_by_type: dict[BeamType, int],
) -> dict[int, str]:
    """Assign beams to section groups using geometry-only clustering.

    This is a lightweight fallback for generated training data and simple
    grouped designs. It clusters each BeamType independently using normalized
    midpoint, length, direction, and face-family features.
    """

    context = make_geometry_context(model)
    assignments: dict[int, str] = {}
    for beam_type in sorted(n_groups_by_type, key=lambda item: int(item.value)):
        beam_ids = [
            beam_id
            for beam_id, beam in model.beams.items()
            if not beam.is_virtual and beam.beam_type == beam_type
        ]
        if not beam_ids:
            continue

        n_groups = max(1, min(int(n_groups_by_type.get(beam_type, 1)), len(beam_ids)))
        if n_groups == 1:
            for beam_id in beam_ids:
                assignments[beam_id] = f"{beam_type.name}-SG1"
            continue

        vectors = np.array(
            [beam_assignment_vector(model, model.beams[beam_id], context) for beam_id in beam_ids],
            dtype=float,
        )
        labels = _stable_kmeans(vectors, n_groups)
        labels = _order_labels_by_geometry(vectors, labels)
        for beam_id, label in zip(beam_ids, labels):
            assignments[beam_id] = f"{beam_type.name}-SG{int(label) + 1}"
    return assignments


def _stable_kmeans(vectors: np.ndarray, n_clusters: int, max_iter: int = 30) -> np.ndarray:
    centers = _initial_centers(vectors, n_clusters)
    labels = np.zeros(len(vectors), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels

        for cluster_idx in range(n_clusters):
            members = vectors[labels == cluster_idx]
            if len(members):
                centers[cluster_idx] = members.mean(axis=0)
    return labels


def _initial_centers(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    order_key = vectors[:, 0] + 0.37 * vectors[:, 1] + 0.19 * vectors[:, 2] + 0.11 * vectors[:, 3]
    sorted_vectors = vectors[np.argsort(order_key)]
    positions = np.linspace(0, len(sorted_vectors) - 1, n_clusters).round().astype(int)
    return sorted_vectors[positions].copy()


def _order_labels_by_geometry(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_labels = sorted(set(int(label) for label in labels))
    centroids = {
        label: vectors[labels == label].mean(axis=0)
        for label in unique_labels
    }
    ordered = sorted(
        unique_labels,
        key=lambda label: (
            float(centroids[label][2]),
            float(centroids[label][1]),
            float(centroids[label][0]),
            float(centroids[label][3]),
        ),
    )
    remap = {old_label: new_label for new_label, old_label in enumerate(ordered)}
    return np.array([remap[int(label)] for label in labels], dtype=int)
