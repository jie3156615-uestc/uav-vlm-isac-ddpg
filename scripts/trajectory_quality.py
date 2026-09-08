import math

import numpy as np


def _trajectory_points(output, ground_length, ground_width, height):
    return np.column_stack(
        [
            np.asarray(output["uav_x"], dtype=float) * ground_length,
            np.asarray(output["uav_y"], dtype=float) * ground_width,
            np.asarray(output["uav_z"], dtype=float) * height,
        ]
    )


def _final_gt_centroid(output, ground_length, ground_width):
    if "gt_x" not in output or not output["gt_x"]:
        return None
    return np.array(
        [
            np.mean(np.asarray(output["gt_x"][-1], dtype=float) * ground_length),
            np.mean(np.asarray(output["gt_y"][-1], dtype=float) * ground_width),
        ],
        dtype=float,
    )


def compute_trajectory_quality(output, ground_length, ground_width, height):
    points = _trajectory_points(output, ground_length, ground_width, height)
    if len(points) < 2:
        return {
            "steps": max(len(points) - 1, 0),
            "path_length": 0.0,
            "chord_length": 0.0,
            "straightness_ratio": 0.0,
            "max_deviation": 0.0,
            "mean_deviation": 0.0,
            "end_centroid_distance": float("nan"),
            "boundary_fraction": 1.0,
            "max_vertical_step": 0.0,
            "vertical_range": 0.0,
            "terminal_vertical_drop": 0.0,
        }

    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_length = float(np.sum(segment_lengths))
    chord_vec = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord_vec))
    if chord_length > 1e-9:
        line_deviation = np.linalg.norm(np.cross(points - points[0], chord_vec), axis=1) / chord_length
    else:
        line_deviation = np.zeros(len(points), dtype=float)

    centroid = _final_gt_centroid(output, ground_length, ground_width)
    end_centroid_distance = (
        float(np.linalg.norm(points[-1, :2] - centroid))
        if centroid is not None
        else float("nan")
    )
    boundary_margin = 0.035 * min(float(ground_length), float(ground_width))
    boundary_fraction = float(
        np.mean(
            (points[:, 0] < boundary_margin)
            | (points[:, 1] < boundary_margin)
            | (points[:, 0] > ground_length - boundary_margin)
            | (points[:, 1] > ground_width - boundary_margin)
        )
    )
    z_values = points[:, 2]
    terminal_window = min(10, len(z_values) - 1)
    terminal_vertical_drop = float(z_values[-terminal_window - 1] - z_values[-1]) if terminal_window > 0 else 0.0

    return {
        "steps": int(len(points) - 1),
        "path_length": path_length,
        "chord_length": chord_length,
        "straightness_ratio": path_length / chord_length if chord_length > 1e-9 else 0.0,
        "max_deviation": float(np.max(line_deviation)),
        "mean_deviation": float(np.mean(line_deviation)),
        "end_centroid_distance": end_centroid_distance,
        "boundary_fraction": boundary_fraction,
        "max_vertical_step": float(np.max(np.abs(np.diff(z_values)))) if len(z_values) > 1 else 0.0,
        "vertical_range": float(np.max(z_values) - np.min(z_values)),
        "terminal_vertical_drop": terminal_vertical_drop,
    }


def trajectory_quality_ok(quality, ground_length, ground_width):
    diag = math.sqrt(float(ground_length) ** 2 + float(ground_width) ** 2)
    if quality["steps"] <= 1:
        return False
    if quality["max_deviation"] < 8.0:
        return False
    if quality["boundary_fraction"] > 0.25:
        return False
    if quality["max_vertical_step"] > 6.0:
        return False
    if quality["terminal_vertical_drop"] > 45.0:
        return False
    if math.isfinite(quality["end_centroid_distance"]) and quality["end_centroid_distance"] > 0.45 * diag:
        return False
    return True


def flatten_quality(prefix, quality, ground_length, ground_width):
    return {
        f"{prefix}_quality_ok": trajectory_quality_ok(quality, ground_length, ground_width),
        f"{prefix}_straightness_ratio": quality["straightness_ratio"],
        f"{prefix}_max_deviation": quality["max_deviation"],
        f"{prefix}_end_centroid_distance": quality["end_centroid_distance"],
        f"{prefix}_boundary_fraction": quality["boundary_fraction"],
        f"{prefix}_max_vertical_step": quality["max_vertical_step"],
        f"{prefix}_terminal_vertical_drop": quality["terminal_vertical_drop"],
    }
