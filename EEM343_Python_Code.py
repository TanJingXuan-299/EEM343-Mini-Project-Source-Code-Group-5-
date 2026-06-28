"""
Modern PySide6 GUI for the EEM343 6-DOF robot arm.

This replaces the Tkinter interface with a Qt desktop app:
- true rotatable OpenGL 3D viewport
- modern sidebar/card layout
- live FK, workspace status, PWM display
- trajectory plots for position, velocity, acceleration, and jerk
- optional serial/Bluetooth streaming to the Arduino controller
"""

from __future__ import annotations

import csv
import math
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from dataclasses import dataclass

try:
    import serial  # type: ignore
except ImportError:
    serial = None

try:
    from serial.tools import list_ports  # type: ignore
except Exception:
    list_ports = None


DH_PARAMS = [
    # alpha, a, d, offset, q_min, q_max
    (math.pi / 2, -26.9, 124.1, 0.0, -math.pi / 2, math.pi / 2),
    (0.0, 158.3, 0.0, math.pi / 2, math.radians(-27.0), math.radians(45.0)),
    (-math.pi / 2, 45.8, 0.0, 0.0, math.radians(-20.0), math.radians(50.0)),
    (math.pi / 2, 0.0, 283.0, 0.0, -3 * math.pi / 4, 3 * math.pi / 4),
    (-math.pi / 2, 0.0, 0.0, 0.0, -math.pi / 2, math.pi / 2),
    (0.0, 0.0, 72.2, 0.0, -math.pi / 2, math.pi / 2),
]

Q_MIN = [row[4] for row in DH_PARAMS]
Q_MAX = [row[5] for row in DH_PARAMS]
Q_HOME = [0.0] * 6

SERVO_ANGLE_RANGE_DEG = [180.0, 270.0, 270.0, 270.0, 180.0, 180.0]
HOME_COUNTS = [324.0, 319.0, 131.0, 287.0, 299.0, 307.0]
HOME_US = [count * 20000.0 / 4096.0 for count in HOME_COUNTS]
SERVO_MIN_US = [500.0] * 6
SERVO_MAX_US = [2500.0] * 6
# The J2 servo is mounted opposite to the positive direction in the kinematic
# model.  Inverting it here makes the physical robot follow the animation.
SERVO_DIRECTION = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [[sum(a[r][k] * b[k][c] for k in range(inner)) for c in range(cols)] for r in range(rows)]


def dh_transform(alpha: float, a: float, d: float, theta: float) -> list[list[float]]:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return [
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ]


def identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def kinematic_joint_angles(q: list[float]) -> list[float]:
    q_model = q[:]
    # Joint 3 is driven by a four-bar linkage from the base side. Treat the UI
    # value as Link 3 absolute elevation, then convert to the serial DH elbow
    # angle for FK/IK drawing and solving.
    q_model[2] = q[2] - q[1]
    return q_model


def fkine_all(q: list[float]) -> list[list[list[float]]]:
    transforms = []
    current = identity4()
    for qi, (alpha, a, d, offset, *_rest) in zip(kinematic_joint_angles(q), DH_PARAMS):
        current = matmul(current, dh_transform(alpha, a, d, qi + offset))
        transforms.append(current)
    return transforms


def pose_from_q(q: list[float]) -> tuple[list[float], tuple[float, float, float]]:
    transform = fkine_all(q)[-1]
    xyz = [transform[0][3], transform[1][3], transform[2][3]]
    r11, r12, r13 = transform[0][0], transform[0][1], transform[0][2]
    r21, r22, r23 = transform[1][0], transform[1][1], transform[1][2]
    r31, r32, r33 = transform[2][0], transform[2][1], transform[2][2]
    pitch = math.atan2(-r31, math.sqrt(r11 * r11 + r21 * r21))
    roll = math.atan2(r32, r33)
    yaw = math.atan2(r21, r11)
    return xyz, (roll, pitch, yaw)


def pitch_from_q(q: list[float]) -> float:
    return pose_from_q(q)[1][1]


def residual(q: list[float], target_xyz: list[float], target_pitch: float | None) -> list[float]:
    xyz, rpy = pose_from_q(q)
    err = [target_xyz[i] - xyz[i] for i in range(3)]
    if target_pitch is not None:
        angle_err = target_pitch - rpy[1]
        while angle_err > math.pi:
            angle_err -= 2 * math.pi
        while angle_err < -math.pi:
            angle_err += 2 * math.pi
        err.append(90.0 * angle_err)
    return err


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        aug[col] = [v / divisor for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


def solve_ik(
    target_xyz: list[float],
    seed: list[float] | None = None,
    target_pitch: float | None = None,
    max_iter: int = 120,
    locked_joints: dict[int, float] | None = None,
) -> tuple[bool, list[float], str, float]:
    q = [clamp((seed or Q_HOME)[i], Q_MIN[i], Q_MAX[i]) for i in range(6)]
    locked = {
        joint: clamp(value, Q_MIN[joint], Q_MAX[joint])
        for joint, value in (locked_joints or {}).items()
        if 0 <= joint < 6
    }
    for joint, value in locked.items():
        q[joint] = value
    active_joints = [joint for joint in range(6) if joint not in locked]
    damping = 18.0
    step = 1e-4
    last_norm = float("inf")
    for _ in range(max_iter):
        err = residual(q, target_xyz, target_pitch)
        err_norm = math.sqrt(sum(e * e for e in err))
        if err_norm < 0.20:
            return True, q, "IK converged", err_norm
        if err_norm > last_norm + 80.0:
            damping *= 1.3
        last_norm = err_norm

        jacobian: list[list[float]] = []
        for row in range(len(err)):
            jacobian.append([])
        for joint in active_joints:
            q_plus = q[:]
            q_plus[joint] += step
            err_plus = residual(q_plus, target_xyz, target_pitch)
            for row in range(len(err)):
                jacobian[row].append((err_plus[row] - err[row]) / step)

        # Damped least squares: dq = J.T * inv(J*J.T + lambda^2 I) * err.
        jj_t = []
        for r in range(len(err)):
            row = []
            for c in range(len(err)):
                row.append(sum(
                    jacobian[r][column] * jacobian[c][column]
                    for column in range(len(active_joints))
                ))
            row[r] += damping * damping
            jj_t.append(row)
        y = solve_linear_system(jj_t, err)
        dq = {
            joint: sum(jacobian[r][column] * y[r] for r in range(len(err)))
            for column, joint in enumerate(active_joints)
        }
        for joint in active_joints:
            q[joint] = clamp(q[joint] - clamp(dq[joint], -0.12, 0.12), Q_MIN[joint], Q_MAX[joint])
        for joint, value in locked.items():
            q[joint] = value
    err = residual(q, target_xyz, target_pitch)
    err_norm = math.sqrt(sum(e * e for e in err))
    return False, q, f"IK stopped with {err_norm:.1f} weighted error", err_norm


def hardware_joint_angles(q: list[float]) -> list[float]:
    return q[:]


def joint_angles_to_pwm_us(q: list[float]) -> list[int]:
    q_hw = hardware_joint_angles(q)
    home_hw = hardware_joint_angles(Q_HOME)
    values = []
    for i in range(6):
        delta_deg = math.degrees(q_hw[i] - home_hw[i])
        us_per_deg = (SERVO_MAX_US[i] - SERVO_MIN_US[i]) / SERVO_ANGLE_RANGE_DEG[i]
        pwm = HOME_US[i] + SERVO_DIRECTION[i] * delta_deg * us_per_deg
        values.append(int(round(clamp(pwm, SERVO_MIN_US[i], SERVO_MAX_US[i]))))
    return values


def joint_limit_note(q: list[float], margin_deg: float = 0.2) -> str:
    margin = math.radians(margin_deg)
    hits = []
    for i, value in enumerate(q):
        if value <= Q_MIN[i] + margin:
            hits.append(f"J{i + 1} at min {math.degrees(Q_MIN[i]):.1f} deg")
        elif value >= Q_MAX[i] - margin:
            hits.append(f"J{i + 1} at max {math.degrees(Q_MAX[i]):.1f} deg")
    return "; ".join(hits)


@dataclass
class PoseTarget:
    x: float
    y: float
    z: float
    pitch: float


@dataclass
class MotionTarget:
    pose: PoseTarget
    pour_deg: float = 0.0
    dwell_s: float = 0.0


@dataclass
class Trajectory:
    time_s: list[float]
    q_rad: list[list[float]]
    q_vel: list[list[float]]
    q_acc: list[list[float]]
    q_jerk: list[list[float]]
    pwm_us: list[list[int]]
    ee_xyz: list[list[float]]
    ik_errors: list[float]

    @property
    def dt_ms(self) -> int:
        if len(self.time_s) < 2:
            return 50
        return int(round(1000.0 * (self.time_s[1] - self.time_s[0])))

    def frame_dt_ms(self, index: int) -> int:
        if len(self.time_s) < 2:
            return 50
        if index + 1 < len(self.time_s):
            start_ms = int(round(1000.0 * self.time_s[index]))
            end_ms = int(round(1000.0 * self.time_s[index + 1]))
            return max(1, end_ms - start_ms)
        return 0

    @property
    def duration_s(self) -> float:
        return self.time_s[-1] if self.time_s else 0.0

    @property
    def stream_duration_s(self) -> float:
        return sum(self.frame_dt_ms(i) for i in range(len(self.time_s))) / 1000.0


def home_pose_target() -> PoseTarget:
    xyz, rpy = pose_from_q(Q_HOME)
    return PoseTarget(xyz[0], xyz[1], xyz[2], rpy[1])


TABLE_Z_MM = 0.0
CUP_TOP_Z_MM = 115.0
POUR_HEIGHT_ABOVE_CUP_MM = 55.0
APPROACH_HEIGHT_MM = 270.0
CUP_X_MM = -360.0
CUP_Y_MM = [-180.0, 0.0, 180.0]
TRIANGLE_CUP_XY_MM = [
    (CUP_X_MM, CUP_Y_MM[0]),
    (-470.0, CUP_Y_MM[1]),
    (CUP_X_MM, CUP_Y_MM[2]),
]
STAR_CUP_XY_MM = [
    (-395.0, -104.0),
    (-296.2, -32.5),
    (-333.9, 84.5),
    (-456.1, 84.5),
    (-493.8, -32.5),
]
STAR_EDGE_ORDER = [0, 2, 4, 1, 3, 0]
POUR_ANGLES_DEG = [45.0, 67.5, 90.0]
STAR_POUR_ANGLES_DEG = [45.0, 56.25, 67.5, 78.75, 90.0]
DEFAULT_MAX_JOINT_SPEED_DEG_S = 200.0
DEFAULT_HOME_JOINT_SPEED_DEG_S = 200.0
DEFAULT_MAX_CART_SPEED_MM_S = 480.0
DEFAULT_JOINT_SMOOTHING_PASSES = 5
DEFAULT_MAX_JOINT_JERK_RAD_S3 = 1.0
USB_FRAME_RATE_FPS = 45
BLUETOOTH_FRAME_RATE_FPS = 20
ARDUINO_QUEUE_PRELOAD_FRAMES = 24
ARDUINO_QUEUE_REFILL_BURST = 8
POUR_SECONDS = 0.5
DWELL_SECONDS = 0.15
# The physical fourth servo is J4 in the UI, therefore index 3 here.  The
# triangle routine keeps this wrist-pitch joint at its calibrated home angle.
TRIANGLE_LOCKED_JOINTS = {3: Q_HOME[3]}


def build_waypoints() -> list[PoseTarget]:
    home = home_pose_target()
    y_plane = home.y
    return [
        home,
        PoseTarget(-350.0, y_plane, 100.0, home.pitch),
        PoseTarget(-400.0, y_plane, 100.0, home.pitch),
        PoseTarget(-400.0, y_plane + 50.0, 100.0, -0.6),
        PoseTarget(-400.0, y_plane + 100.0, 100.0, -1.0),
        PoseTarget(-400.0, y_plane + 150.0, 100.0, -1.4),
        home,
    ]


def build_triangle_pour_targets() -> list[MotionTarget]:
    home = home_pose_target()
    hover_z = TABLE_Z_MM + CUP_TOP_Z_MM + POUR_HEIGHT_ABOVE_CUP_MM
    first_x, first_y = TRIANGLE_CUP_XY_MM[0]
    targets = [
        MotionTarget(home, 0.0),
        MotionTarget(PoseTarget(first_x, first_y, APPROACH_HEIGHT_MM, home.pitch), 0.0),
    ]
    for index, (x_value, y_value) in enumerate(TRIANGLE_CUP_XY_MM):
        pour_angle = POUR_ANGLES_DEG[min(index, len(POUR_ANGLES_DEG) - 1)]
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), 0.0))
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), pour_angle, DWELL_SECONDS))
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), 0.0, DWELL_SECONDS))
    targets.append(MotionTarget(PoseTarget(first_x, first_y, hover_z, home.pitch), 0.0))
    targets.append(MotionTarget(PoseTarget(first_x, first_y, APPROACH_HEIGHT_MM, home.pitch), 0.0))
    targets.append(MotionTarget(home, 0.0))
    return targets


def build_star_pour_targets() -> list[MotionTarget]:
    """Visit five cup vertices once and close the pentagram at pour height."""
    home = home_pose_target()
    hover_z = TABLE_Z_MM + CUP_TOP_Z_MM + POUR_HEIGHT_ABOVE_CUP_MM
    first_index = STAR_EDGE_ORDER[0]
    first_x, first_y = STAR_CUP_XY_MM[first_index]
    targets = [
        MotionTarget(home, 0.0),
        MotionTarget(PoseTarget(first_x, first_y, APPROACH_HEIGHT_MM, home.pitch), 0.0),
    ]

    for visit_index, cup_index in enumerate(STAR_EDGE_ORDER[:-1]):
        x_value, y_value = STAR_CUP_XY_MM[cup_index]
        pour_angle = STAR_POUR_ANGLES_DEG[visit_index]
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), 0.0))
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), pour_angle, DWELL_SECONDS))
        targets.append(MotionTarget(PoseTarget(x_value, y_value, hover_z, home.pitch), 0.0, DWELL_SECONDS))

    # Return to the first vertex at the same elevation to draw the final star edge.
    targets.append(MotionTarget(PoseTarget(first_x, first_y, hover_z, home.pitch), 0.0))
    targets.append(MotionTarget(PoseTarget(first_x, first_y, APPROACH_HEIGHT_MM, home.pitch), 0.0))
    targets.append(MotionTarget(home, 0.0))
    return targets


TRAJECTORY_BUILDERS = {
    "Triangle cups": build_triangle_pour_targets,
    "Star cups": build_star_pour_targets,
}


# ---------------------------------------------------------------------------
# Workspace computation: Monte Carlo FK sampling + (r, z) radial envelope
# ---------------------------------------------------------------------------

WORKSPACE_SAMPLE_COUNT = 40000
WORKSPACE_Z_BINS = 48
WORKSPACE_RANDOM_SEED = 42
WORKSPACE_NEAREST_TOL_MM = 90.0


def _workspace_samples() -> np.ndarray:
    """Monte Carlo sample the FK workspace by drawing random joint
    configurations inside the joint limits and recording the TCP xyz.

    Returns an (N, 3) float array. A fixed RNG seed keeps the visualization
    deterministic between runs.
    """
    rng = np.random.default_rng(WORKSPACE_RANDOM_SEED)
    n = WORKSPACE_SAMPLE_COUNT
    cloud = np.empty((n, 3), dtype=float)
    for i in range(n):
        q = [float(rng.uniform(Q_MIN[j], Q_MAX[j])) for j in range(6)]
        xyz, _ = pose_from_q(q)
        cloud[i, 0] = xyz[0]
        cloud[i, 1] = xyz[1]
        cloud[i, 2] = xyz[2]
    return cloud


WORKSPACE_CLOUD = _workspace_samples()
WORKSPACE_R_VALUES = np.hypot(WORKSPACE_CLOUD[:, 0], WORKSPACE_CLOUD[:, 1])
WORKSPACE_Z_VALUES = WORKSPACE_CLOUD[:, 2]

WORKSPACE_R_MIN_MM = float(WORKSPACE_R_VALUES.min())
WORKSPACE_R_MAX_MM = float(WORKSPACE_R_VALUES.max())
WORKSPACE_MIN_Z_MM = float(WORKSPACE_Z_VALUES.min())
WORKSPACE_MAX_Z_MM = float(WORKSPACE_Z_VALUES.max())


def _build_radial_envelope(n_bins: int):
    """Bin samples by Z and record min/max reachable radius at each height.

    Returns (z_centers, r_min_per_z, r_max_per_z, bin_counts). NaNs are
    forward/backward filled from neighbouring bins so the shell mesh is
    continuous.
    """
    z_edges = np.linspace(WORKSPACE_MIN_Z_MM, WORKSPACE_MAX_Z_MM, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    r_min = np.full(n_bins, np.nan)
    r_max = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (WORKSPACE_Z_VALUES >= z_edges[i]) & (WORKSPACE_Z_VALUES <= z_edges[i + 1])
        if mask.sum() > 0:
            r_min[i] = float(WORKSPACE_R_VALUES[mask].min())
            r_max[i] = float(WORKSPACE_R_VALUES[mask].max())
            counts[i] = int(mask.sum())
    valid = ~np.isnan(r_min)
    if valid.any():
        first = int(np.argmax(valid))
        last = n_bins - 1 - int(np.argmax(valid[::-1]))
        r_min[:first] = r_min[first]
        r_max[:first] = r_max[first]
        r_min[last + 1:] = r_min[last]
        r_max[last + 1:] = r_max[last]
    return z_centers, r_min, r_max, counts


WORKSPACE_Z_CENTERS, WORKSPACE_R_MIN_PER_Z, WORKSPACE_R_MAX_PER_Z, WORKSPACE_BIN_COUNTS = \
    _build_radial_envelope(WORKSPACE_Z_BINS)


def workspace_status(xyz: list[float]) -> tuple[bool, str]:
    """Check whether a Cartesian target is inside the sampled FK envelope.

    Uses the (r, z) radial envelope: at the target's height, the radius
    must lie within [r_min(z), r_max(z)]. This captures the unreachable
    donut hole near the base that a simple bounding-box test misses.
    """
    radius = math.hypot(xyz[0], xyz[1])
    z = xyz[2]
    # Nearest cloud point for diagnostics
    diffs = WORKSPACE_CLOUD - np.asarray(xyz, dtype=float)
    nearest = float(np.sqrt(np.min(np.sum(diffs * diffs, axis=1))))
    if z < WORKSPACE_MIN_Z_MM - 5 or z > WORKSPACE_MAX_Z_MM + 5:
        return False, (
            f"workspace: z={z:.0f} mm outside height range "
            f"[{WORKSPACE_MIN_Z_MM:.0f}, {WORKSPACE_MAX_Z_MM:.0f}] mm; "
            f"nearest sample {nearest:.0f} mm"
        )
    idx = int(np.searchsorted(WORKSPACE_Z_CENTERS, z) - 1)
    idx = max(0, min(WORKSPACE_Z_BINS - 1, idx))
    r_min_z = float(WORKSPACE_R_MIN_PER_Z[idx])
    r_max_z = float(WORKSPACE_R_MAX_PER_Z[idx])
    if np.isnan(r_min_z) or np.isnan(r_max_z):
        return False, f"workspace: no samples at z={z:.0f} mm; nearest {nearest:.0f} mm"
    inside = bool(r_min_z - 5 <= radius <= r_max_z + 5)
    if inside:
        if nearest > WORKSPACE_NEAREST_TOL_MM:
            return False, (
                f"workspace: outside sampled FK cloud at z={z:.0f} mm, "
                f"r={radius:.0f} mm; nearest sample {nearest:.0f} mm "
                f"> {WORKSPACE_NEAREST_TOL_MM:.0f} mm"
            )
        return True, (
            f"workspace: inside FK envelope at z={z:.0f} mm, "
            f"r={radius:.0f} mm in [{r_min_z:.0f}, {r_max_z:.0f}] mm; "
            f"nearest sample {nearest:.0f} mm"
        )
    return False, (
        f"workspace: outside FK envelope at z={z:.0f} mm, "
        f"r={radius:.0f} mm not in [{r_min_z:.0f}, {r_max_z:.0f}] mm; "
        f"nearest sample {nearest:.0f} mm"
    )

def workspace_wireframe() -> list[list[list[float]]]:
    """Wireframe rings at multiple heights showing both inner and outer
    radial envelopes, plus vertical meridians on the outer surface."""
    rings: list[list[list[float]]] = []
    # Horizontal rings every 4th Z bin (12 rings total)
    for idx in range(0, WORKSPACE_Z_BINS, 4):
        z = float(WORKSPACE_Z_CENTERS[idx])
        r_out = float(WORKSPACE_R_MAX_PER_Z[idx])
        r_in = float(WORKSPACE_R_MIN_PER_Z[idx])
        if not (np.isnan(r_out) or r_out < 1.0):
            ring = [[r_out * math.cos(2 * math.pi * s / 72),
                     r_out * math.sin(2 * math.pi * s / 72),
                     z] for s in range(73)]
            rings.append(ring)
        if not (np.isnan(r_in) or r_in < 1.0):
            ring = [[r_in * math.cos(2 * math.pi * s / 72),
                     r_in * math.sin(2 * math.pi * s / 72),
                     z] for s in range(73)]
            rings.append(ring)
    # Vertical meridians on the outer surface at 8 azimuths
    for a in range(8):
        angle = 2 * math.pi * a / 8
        line = []
        for idx in range(WORKSPACE_Z_BINS):
            z = float(WORKSPACE_Z_CENTERS[idx])
            r_out = float(WORKSPACE_R_MAX_PER_Z[idx])
            if not (np.isnan(r_out) or r_out < 1.0):
                line.append([r_out * math.cos(angle),
                             r_out * math.sin(angle),
                             z])
        if len(line) > 1:
            rings.append(line)
    return rings


def workspace_shell_mesh() -> gl.MeshData:
    """Translucent shell mesh built by revolving the radial envelope (r_min(z)
    and r_max(z)) around the Z axis. Produces both the outer and inner
    surfaces so the donut hole is visible."""
    side_count = 96
    valid = ~np.isnan(WORKSPACE_R_MAX_PER_Z)
    z_vals = WORKSPACE_Z_CENTERS[valid]
    r_outer = WORKSPACE_R_MAX_PER_Z[valid]
    r_inner = WORKSPACE_R_MIN_PER_Z[valid]
    n_z = len(z_vals)

    vertexes = []
    # Outer surface
    for z, r in zip(z_vals, r_outer):
        for i in range(side_count):
            angle = 2.0 * math.pi * i / side_count
            vertexes.append([float(r) * math.cos(angle),
                             float(r) * math.sin(angle),
                             float(z)])
    # Inner surface
    for z, r in zip(z_vals, r_inner):
        for i in range(side_count):
            angle = 2.0 * math.pi * i / side_count
            vertexes.append([float(r) * math.cos(angle),
                             float(r) * math.sin(angle),
                             float(z)])

    faces = []
    outer_start = 0
    inner_start = n_z * side_count
    for row in range(n_z - 1):
        for i in range(side_count):
            j = (i + 1) % side_count
            # Outer (outward normals)
            faces.append([outer_start + row * side_count + i,
                          outer_start + (row + 1) * side_count + j,
                          outer_start + row * side_count + j])
            faces.append([outer_start + row * side_count + i,
                          outer_start + (row + 1) * side_count + i,
                          outer_start + (row + 1) * side_count + j])
            # Inner (inward normals)
            faces.append([inner_start + row * side_count + i,
                          inner_start + row * side_count + j,
                          inner_start + (row + 1) * side_count + j])
            faces.append([inner_start + row * side_count + i,
                          inner_start + (row + 1) * side_count + j,
                          inner_start + (row + 1) * side_count + i])
    # Top and bottom annular caps
    for i in range(side_count):
        j = (i + 1) % side_count
        # Top cap
        faces.append([outer_start + (n_z - 1) * side_count + i,
                      outer_start + (n_z - 1) * side_count + j,
                      inner_start + (n_z - 1) * side_count + j])
        faces.append([outer_start + (n_z - 1) * side_count + i,
                      inner_start + (n_z - 1) * side_count + j,
                      inner_start + (n_z - 1) * side_count + i])
        # Bottom cap
        faces.append([outer_start + i,
                      inner_start + j,
                      outer_start + j])
        faces.append([outer_start + i,
                      inner_start + i,
                      inner_start + j])

    return gl.MeshData(
        vertexes=np.array(vertexes, dtype=float),
        faces=np.array(faces, dtype=int),
    )


def smoothstep_quintic(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def smootherstep_septic(s: float) -> float:
    s = clamp(s, 0.0, 1.0)
    return 35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7


def interpolate_pose(a: PoseTarget, b: PoseTarget, s: float) -> PoseTarget:
    u = smootherstep_septic(s)
    return PoseTarget(
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.z + (b.z - a.z) * u,
        a.pitch + (b.pitch - a.pitch) * u,
    )


def interpolate_straight_pose(a: PoseTarget, b: PoseTarget, s: float) -> PoseTarget:
    s = clamp(s, 0.0, 1.0)
    return PoseTarget(
        a.x + (b.x - a.x) * s,
        a.y + (b.y - a.y) * s,
        a.z + (b.z - a.z) * s,
        a.pitch + (b.pitch - a.pitch) * s,
    )


def estimate_segment_time(a: MotionTarget, b: MotionTarget, max_cart_speed_mm_s: float) -> float:
    distance = math.sqrt(
        (b.pose.x - a.pose.x) ** 2
        + (b.pose.y - a.pose.y) ** 2
        + (b.pose.z - a.pose.z) ** 2
    )
    pour_delta = abs(math.radians(b.pour_deg - a.pour_deg))
    # Septic interpolation has zero velocity, acceleration, and jerk at each
    # waypoint.  Its peak speed is 35/16 of the average, so scale durations to
    # retain the requested speed limit while eliminating corner impulses.
    septic_peak_speed_factor = 35.0 / 16.0
    move_time = septic_peak_speed_factor * distance / max(1.0, max_cart_speed_mm_s)
    pour_time = max(
        POUR_SECONDS if pour_delta > math.radians(3.0) else 0.0,
        septic_peak_speed_factor * pour_delta / 0.84,
    )
    return max(0.2, move_time, pour_time, b.dwell_s)


def interpolate_motion(a: MotionTarget, b: MotionTarget, s: float) -> MotionTarget:
    u = smootherstep_septic(s)
    return MotionTarget(
        interpolate_pose(a.pose, b.pose, s),
        a.pour_deg + (b.pour_deg - a.pour_deg) * u,
    )


def interpolate_straight_motion(a: MotionTarget, b: MotionTarget, s: float) -> MotionTarget:
    s = clamp(s, 0.0, 1.0)
    return MotionTarget(
        interpolate_straight_pose(a.pose, b.pose, s),
        a.pour_deg + (b.pour_deg - a.pour_deg) * s,
    )


def interpolate_joint_pose(a: list[float], b: list[float], s: float) -> list[float]:
    u = smootherstep_septic(s)
    return [clamp(a[i] + (b[i] - a[i]) * u, Q_MIN[i], Q_MAX[i]) for i in range(6)]


def same_pose(a: PoseTarget, b: PoseTarget, tol: float = 1e-6) -> bool:
    return (
        abs(a.x - b.x) <= tol
        and abs(a.y - b.y) <= tol
        and abs(a.z - b.z) <= tol
        and abs(a.pitch - b.pitch) <= tol
    )


def rescale_time_rows(time_rows: list[float], target_duration_s: float | None) -> list[float]:
    if target_duration_s is None or target_duration_s <= 0.0 or len(time_rows) < 2:
        return time_rows
    current_duration = max(1e-6, time_rows[-1] - time_rows[0])
    scale = target_duration_s / current_duration
    return [(t - time_rows[0]) * scale for t in time_rows]


def trajectory_from_motion_targets(
    targets: list[MotionTarget],
    fps: int,
    start_q: list[float] | None = None,
    max_joint_speed_deg_s: float = DEFAULT_MAX_JOINT_SPEED_DEG_S,
    max_cart_speed_mm_s: float = DEFAULT_MAX_CART_SPEED_MM_S,
    target_duration_s: float | None = None,
    locked_joints: dict[int, float] | None = None,
) -> Trajectory:
    q_rows: list[list[float]] = []
    t_rows: list[float] = []
    errors: list[float] = []
    q_seed = [clamp((start_q or Q_HOME)[i], Q_MIN[i], Q_MAX[i]) for i in range(6)]
    locked = locked_joints or {}
    for joint, value in locked.items():
        if 0 <= joint < 6:
            q_seed[joint] = clamp(value, Q_MIN[joint], Q_MAX[joint])
    waypoint_q: list[list[float]] = []
    waypoint_errors: list[float] = []
    for index, target in enumerate(targets):
        if index == 0:
            q_target = q_seed[:]
            if not same_pose(target.pose, home_pose_target(), tol=1e-3):
                ok, q_target, reason, err = solve_ik(
                    [target.pose.x, target.pose.y, target.pose.z],
                    q_seed,
                    target.pose.pitch,
                    locked_joints=locked,
                )
                if not ok and err > 35.0:
                    raise RuntimeError(f"IK failed at waypoint {index + 1}: {reason}")
            else:
                err = 0.0
        else:
            previous_target = targets[index - 1]
            previous_q = waypoint_q[-1]
            if same_pose(previous_target.pose, target.pose):
                q_target = previous_q[:]
                err = 0.0
            else:
                ok, q_target, reason, err = solve_ik(
                    [target.pose.x, target.pose.y, target.pose.z],
                    previous_q,
                    target.pose.pitch,
                    locked_joints=locked,
                )
                if not ok and err > 35.0:
                    raise RuntimeError(f"IK failed at waypoint {index + 1}: {reason}")
        q_target[5] = clamp(math.radians(target.pour_deg), Q_MIN[5], Q_MAX[5])
        waypoint_q.append(q_target[:])
        waypoint_errors.append(err)

    current_time = 0.0
    for index in range(len(targets) - 1):
        start_target = targets[index]
        end_target = targets[index + 1]
        # Use the configuration actually emitted by the preceding segment.
        # A redundant IK problem can have several valid endpoint solutions;
        # restarting from a separately pre-solved waypoint used to create a
        # one-frame joint jump at some star vertices.
        segment_start_q = q_rows[-1][:] if q_rows else waypoint_q[index]
        segment_end_q = waypoint_q[index + 1]
        if same_pose(start_target.pose, end_target.pose):
            # A pour/dwell does not move the TCP.  Preserve the continuous
            # Cartesian solution and command only the pour servo (J6), rather
            # than blending toward another equivalent IK posture.
            segment_end_q = segment_start_q[:]
            segment_end_q[5] = clamp(math.radians(end_target.pour_deg), Q_MIN[5], Q_MAX[5])
        err = waypoint_errors[index + 1]

        base_time = estimate_segment_time(start_target, end_target, max_cart_speed_mm_s)
        joint_time = (35.0 / 16.0) * max(
            abs(segment_end_q[joint] - segment_start_q[joint]) / max(0.02, math.radians(max_joint_speed_deg_s))
            for joint in range(6)
        )
        segment_time = max(base_time, joint_time, end_target.dwell_s)
        steps = max(8, int(round(segment_time * fps)))
        row_seed = segment_start_q[:]
        for step_index in range(steps + 1):
            if q_rows and step_index == 0:
                continue
            s = step_index / steps
            if same_pose(start_target.pose, end_target.pose):
                q_sol = interpolate_joint_pose(segment_start_q, segment_end_q, s)
                row_err = err * s
            else:
                # Keep every Cartesian segment exactly on its line/edge, but
                # use a C3-continuous septic progress curve to minimise jerk.
                sample_target = interpolate_motion(start_target, end_target, s)
                if step_index == 0:
                    q_sol = segment_start_q[:]
                    row_err = waypoint_errors[index]
                else:
                    ok, q_sol, reason, row_err = solve_ik(
                        [sample_target.pose.x, sample_target.pose.y, sample_target.pose.z],
                        row_seed,
                        sample_target.pose.pitch,
                        max_iter=80,
                        locked_joints=locked,
                    )
                    if not ok and row_err > 35.0:
                        raise RuntimeError(
                            f"IK failed on straight segment {index + 1} at {s * 100.0:.0f}%: {reason}"
                        )
                    q_sol[5] = clamp(math.radians(sample_target.pour_deg), Q_MIN[5], Q_MAX[5])
                    row_seed = q_sol[:]
            q_rows.append(q_sol)
            errors.append(row_err)
            t_rows.append(current_time + s * segment_time)
        current_time += segment_time

    if target_duration_s is None or target_duration_s <= 0.0:
        t_rows = speed_limited_time_rows(q_rows, t_rows, math.radians(max_joint_speed_deg_s))
    else:
        t_rows = rescale_time_rows(t_rows, target_duration_s)
    # Gaussian pre-filter removes high-frequency IK jitter while keeping the
    # trajectory close to the Cartesian straight-line samples (sub-mm TCP
    # deviation with sigma=2.0).  The straight-line path shape is preserved.
    q_rows = smooth_joint_rows_gaussian(q_rows, sigma=2.0, locked_joints=locked)
    xyz_rows = [pose_from_q(q)[0] for q in q_rows]
    # Differentiate against the actual timestamps. The speed limiter can make
    # frame spacing non-uniform, so an average dt would misreport the profiles.
    q_vel, q_acc, q_jerk = spline_derivatives_by_time(q_rows, t_rows)
    pwm_rows = [joint_angles_to_pwm_us(q) for q in q_rows]
    return Trajectory(t_rows, q_rows, q_vel, q_acc, q_jerk, pwm_rows, xyz_rows, errors)


def calculate_trajectory(
    fps: int = 30,
    max_joint_speed_deg_s: float = DEFAULT_MAX_JOINT_SPEED_DEG_S,
    max_cart_speed_mm_s: float = DEFAULT_MAX_CART_SPEED_MM_S,
    target_duration_s: float | None = None,
    path_name: str = "Triangle cups",
) -> Trajectory:
    builder = TRAJECTORY_BUILDERS.get(path_name, build_triangle_pour_targets)
    return trajectory_from_motion_targets(
        builder(),
        fps,
        None,
        max_joint_speed_deg_s,
        max_cart_speed_mm_s,
        target_duration_s,
        TRIANGLE_LOCKED_JOINTS if path_name == "Triangle cups" else None,
    )


def joint_motion_trajectory(
    start_q: list[float],
    end_q: list[float],
    fps: int = 30,
    max_joint_speed_deg_s: float = DEFAULT_MAX_JOINT_SPEED_DEG_S,
) -> Trajectory:
    max_delta = max(abs(end_q[i] - start_q[i]) for i in range(6))
    max_joint_speed_rad_s = math.radians(max_joint_speed_deg_s)
    duration = max(0.8, max_delta / max(0.02, max_joint_speed_rad_s))
    # Clamped cubic spline: globally C2-continuous with analytical derivatives.
    # Produces perfectly smooth velocity/acceleration and piecewise-constant
    # jerk -- no numerical differentiation noise.
    q_rows, t_rows, q_vel, q_acc, q_jerk, xyz_rows = cubic_spline_joint_trajectory(
        start_q, end_q, fps, duration,
    )
    pwm_rows = [joint_angles_to_pwm_us(q) for q in q_rows]
    return Trajectory(t_rows, q_rows, q_vel, q_acc, q_jerk, pwm_rows, xyz_rows, [0.0] * len(q_rows))


def speed_limited_time_rows(q_rows: list[list[float]], time_rows: list[float], max_joint_speed_rad_s: float) -> list[float]:
    if len(q_rows) < 2:
        return time_rows
    max_joint_speed_rad_s = max(0.02, max_joint_speed_rad_s)
    out = [0.0]
    for i in range(1, len(q_rows)):
        planned_dt = max(1e-4, time_rows[i] - time_rows[i - 1])
        required_dt = max(abs(q_rows[i][joint] - q_rows[i - 1][joint]) / max_joint_speed_rad_s for joint in range(6))
        out.append(out[-1] + max(planned_dt, required_dt))
    return out


def jerk_limited_time_rows(
    q_rows: list[list[float]],
    time_rows: list[float],
    max_joint_jerk_rad_s3: float,
) -> list[float]:
    if len(q_rows) < 4 or len(time_rows) != len(q_rows):
        return time_rows
    limit = max(0.05, max_joint_jerk_rad_s3)
    q_vel = gradient_by_time(q_rows, time_rows)
    q_acc = gradient_by_time(q_vel, time_rows)
    q_jerk = gradient_by_time(q_acc, time_rows)
    peak_jerk = max(abs(value) for row in q_jerk for value in row)
    if peak_jerk <= limit:
        return time_rows
    # Jerk scales with 1 / time^3. Stretching the same path is the least
    # invasive way to protect the hardware from abrupt frame-to-frame changes.
    scale = 1.08 * (peak_jerk / limit) ** (1.0 / 3.0)
    return [time_rows[0] + (t - time_rows[0]) * scale for t in time_rows]


def smooth_joint_rows(
    rows: list[list[float]],
    passes: int = 1,
    locked_joints: dict[int, float] | None = None,
) -> list[list[float]]:
    if len(rows) < 5:
        return rows
    locked = {
        joint: clamp(value, Q_MIN[joint], Q_MAX[joint])
        for joint, value in (locked_joints or {}).items()
        if 0 <= joint < 6
    }
    out = [row[:] for row in rows]
    for _ in range(passes):
        next_rows = [out[0][:]]
        for i in range(1, len(out) - 1):
            next_rows.append([
                clamp(
                    0.25 * out[i - 1][joint] + 0.5 * out[i][joint] + 0.25 * out[i + 1][joint],
                    Q_MIN[joint],
                    Q_MAX[joint],
                )
                for joint in range(6)
            ])
        next_rows.append(out[-1][:])
        for row in next_rows:
            for joint, value in locked.items():
                row[joint] = value
        out = next_rows
    return out


def gradient(rows: list[list[float]], dt: float) -> list[list[float]]:
    if len(rows) < 2:
        return [[0.0] * len(rows[0])] if rows else []
    out = []
    for i in range(len(rows)):
        if i == 0:
            base = [(rows[1][j] - rows[0][j]) / dt for j in range(len(rows[0]))]
        elif i == len(rows) - 1:
            base = [(rows[i][j] - rows[i - 1][j]) / dt for j in range(len(rows[0]))]
        else:
            base = [(rows[i + 1][j] - rows[i - 1][j]) / (2 * dt) for j in range(len(rows[0]))]
        out.append(base)
    return out


def gradient_by_time(rows: list[list[float]], time_s: list[float]) -> list[list[float]]:
    if len(rows) < 2:
        return [[0.0] * len(rows[0])] if rows else []
    out = []
    for i in range(len(rows)):
        if i == 0:
            dt = max(1e-6, time_s[1] - time_s[0])
            base = [(rows[1][j] - rows[0][j]) / dt for j in range(len(rows[0]))]
        elif i == len(rows) - 1:
            dt = max(1e-6, time_s[i] - time_s[i - 1])
            base = [(rows[i][j] - rows[i - 1][j]) / dt for j in range(len(rows[0]))]
        else:
            dt = max(1e-6, time_s[i + 1] - time_s[i - 1])
            base = [(rows[i + 1][j] - rows[i - 1][j]) / dt for j in range(len(rows[0]))]
        out.append(base)
    return out


def smooth_joint_rows_gaussian(
    rows: list[list[float]],
    sigma: float = 2.0,
    locked_joints: dict[int, float] | None = None,
) -> list[list[float]]:
    """Lightweight Gaussian filter on joint-angle rows to remove IK noise.

    Preserves the first and last rows exactly (endpoint pinning) so the
    trajectory still starts and ends at the solved waypoints.  The sigma
    parameter controls how many samples are averaged -- 2.0 gives sub-mm
    TCP deviation on typical trajectories.
    """
    if len(rows) < 5:
        return rows
    locked = {
        joint: clamp(value, Q_MIN[joint], Q_MAX[joint])
        for joint, value in (locked_joints or {}).items()
        if 0 <= joint < 6
    }
    arr = np.array(rows, dtype=float)  # (N, 6)
    first_row = arr[0].copy()
    last_row = arr[-1].copy()
    for j in range(arr.shape[1]):
        if j in locked:
            arr[:, j] = locked[j]
        else:
            arr[:, j] = gaussian_filter1d(arr[:, j], sigma=sigma, mode='nearest')
    # Pin endpoints so the trajectory starts/ends exactly at the waypoints.
    arr[0] = first_row
    arr[-1] = last_row
    # Clamp to joint limits.
    for j in range(6):
        arr[:, j] = np.clip(arr[:, j], Q_MIN[j], Q_MAX[j])
    return [arr[i].tolist() for i in range(arr.shape[0])]


def spline_derivatives_by_time(
    q_rows: list[list[float]],
    time_s: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Compute velocity, acceleration, and jerk from the real sample times.

    The Cartesian planner may stretch individual intervals to enforce joint
    speed limits. A clamped cubic spline keeps the derivative estimates tied
    to those non-uniform timestamps and enforces zero endpoint velocity.
    """
    n = len(q_rows)
    n_joints = len(q_rows[0]) if q_rows else 6
    if n < 4 or len(time_s) != n:
        zeros = [[0.0] * n_joints] * max(n, 1)
        return zeros[:], zeros[:], zeros[:]

    arr = np.array(q_rows, dtype=float)
    t_arr = np.array(time_s, dtype=float)
    if np.any(np.diff(t_arr) <= 1e-9):
        q_vel = gradient_by_time(q_rows, time_s)
        q_acc = gradient_by_time(q_vel, time_s)
        q_jerk = gradient_by_time(q_acc, time_s)
        return q_vel, q_acc, q_jerk

    q_vel_arr = np.zeros_like(arr)
    q_acc_arr = np.zeros_like(arr)
    q_jerk_arr = np.zeros_like(arr)

    for j in range(n_joints):
        cs = CubicSpline(t_arr, arr[:, j], bc_type="clamped")
        q_vel_arr[:, j] = cs(t_arr, 1)
        q_acc_arr[:, j] = cs(t_arr, 2)
        q_jerk_arr[:, j] = cs(t_arr, 3)

    q_vel = [q_vel_arr[i].tolist() for i in range(n)]
    q_acc = [q_acc_arr[i].tolist() for i in range(n)]
    q_jerk = [q_jerk_arr[i].tolist() for i in range(n)]
    return q_vel, q_acc, q_jerk


def cubic_spline_joint_trajectory(
    start_q: list[float],
    end_q: list[float],
    fps: int,
    duration: float,
) -> tuple[list[list[float]], list[float], list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """Compute a joint-space trajectory using a clamped cubic spline.

    The spline passes through start_q at t=0 and end_q at t=duration
    with zero velocity at both endpoints (clamped boundary conditions).
    Velocity, acceleration, and jerk are computed analytically from the
    spline, giving perfectly smooth profiles with no numerical noise.

    Returns (q_rows, t_rows, q_vel, q_acc, q_jerk, xyz_rows).
    """
    steps = max(12, int(round(duration * fps)))
    # Knot times: just the two endpoints for a single-segment clamped spline.
    # Adding a midpoint knot gives the spline more freedom for a natural shape.
    t_knots = np.array([0.0, duration / 2.0, duration])
    # Midpoint via linear interpolation (the spline shape handles the
    # acceleration profile; no need for septic shaping here).
    mid_q = [0.5 * (start_q[i] + end_q[i]) for i in range(6)]
    q_knots = np.array([start_q, mid_q, end_q])  # (3, 6)

    # Build one clamped cubic spline per joint (zero velocity at endpoints).
    t_samples = np.linspace(0.0, duration, steps + 1)
    q_arr = np.zeros((steps + 1, 6))
    vel_arr = np.zeros((steps + 1, 6))
    acc_arr = np.zeros((steps + 1, 6))
    jerk_arr = np.zeros((steps + 1, 6))

    for j in range(6):
        cs = CubicSpline(t_knots, q_knots[:, j], bc_type='clamped')
        q_arr[:, j] = np.clip(cs(t_samples), Q_MIN[j], Q_MAX[j])
        vel_arr[:, j] = cs(t_samples, 1)   # 1st derivative
        acc_arr[:, j] = cs(t_samples, 2)   # 2nd derivative
        jerk_arr[:, j] = cs(t_samples, 3)  # 3rd derivative

    q_rows = [q_arr[i].tolist() for i in range(steps + 1)]
    t_rows = t_samples.tolist()
    q_vel = [vel_arr[i].tolist() for i in range(steps + 1)]
    q_acc = [acc_arr[i].tolist() for i in range(steps + 1)]
    q_jerk = [jerk_arr[i].tolist() for i in range(steps + 1)]
    xyz_rows = [pose_from_q(q)[0] for q in q_rows]
    return q_rows, t_rows, q_vel, q_acc, q_jerk, xyz_rows


# ---------------------------------------------------------------------------
# Serial hardware interface
# ---------------------------------------------------------------------------


class SerialLink:
    def __init__(self) -> None:
        self.port = None
        self.is_open = False

    def connect(self, port_name: str, baud: int) -> str:
        if serial is None:
            raise RuntimeError("PySerial is not installed. Run: pip uninstall serial && pip install pyserial")
        if not hasattr(serial, "Serial"):
            module_path = getattr(serial, "__file__", "unknown location")
            raise RuntimeError(
                "The wrong package named 'serial' is installed. "
                "Remove it and install PySerial:\n\n"
                "pip uninstall serial\n"
                "pip install pyserial\n\n"
                f"Currently loaded serial module: {module_path}"
            )
        if not port_name:
            raise RuntimeError("Choose a COM port before connecting.")
        self.close()
        self.port = serial.Serial(port_name, baudrate=baud, timeout=1, write_timeout=1)
        time.sleep(2.0)
        self.port.reset_input_buffer()
        self.is_open = True
        line = self.port.readline().decode(errors="ignore").strip()
        return line or "CONNECTED"

    def close(self) -> None:
        if self.port is not None:
            try:
                self.port.close()
            except Exception:
                pass
        self.port = None
        self.is_open = False

    def command(self, command: str) -> str:
        if self.port is None or not self.is_open:
            raise RuntimeError("Serial port is not connected")
        self.port.write((command.strip() + "\n").encode())
        return self.port.readline().decode(errors="ignore").strip()

    def send_frame(self, pwm_us: list[int], dt_ms: int) -> str:
        started = time.perf_counter()
        reply = self.command("F," + ",".join(str(v) for v in pwm_us) + f",{dt_ms}")
        remaining_s = (max(0, dt_ms) / 1000.0) - (time.perf_counter() - started)
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        return reply

    def clear_queue(self) -> str:
        return self.command("C")

    def start_queue(self) -> str:
        return self.command("G")

    def enqueue_frame(self, pwm_us: list[int], dt_ms: int) -> str:
        return self.command("Q," + ",".join(str(v) for v in pwm_us) + f",{dt_ms}")


APP_STYLE = """
* {
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow {
    background: #0b1120;
}
QFrame#Sidebar, QFrame#Card, QFrame#ViewportCard, QFrame#StatusCard {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 14px;
}
QLabel#Title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
}
QLabel#Subtitle {
    color: #94a3b8;
    font-size: 10px;
}
QLabel#Section {
    color: #f8fafc;
    font-weight: 700;
    font-size: 12px;
}
QLabel#BadgeGood {
    background: #064e3b;
    color: #d1fae5;
    border-radius: 9px;
    padding: 8px 10px;
    font-weight: 700;
}
QLabel#BadgeBad {
    background: #7f1d1d;
    color: #fee2e2;
    border-radius: 9px;
    padding: 8px 10px;
    font-weight: 700;
}
QLabel {
    color: #dbeafe;
}
QPushButton {
    background: #3b82f6;
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 9px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #2563eb;
}
QPushButton:pressed {
    background: #1e40af;
}
QPushButton#Secondary {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #334155;
}
QPushButton#Danger {
    background: #dc2626;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #334155;
    border-bottom: 1px solid #334155;
    background: #1f2937;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #334155;
    background: #1f2937;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 7px solid #f8fafc;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid #f8fafc;
}
QComboBox QAbstractItemView {
    background: #111827;
    color: #f8fafc;
    selection-background-color: #2563eb;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QTabWidget::pane {
    border: 0;
}
QTabBar::tab {
    background: #1f2937;
    color: #cbd5e1;
    padding: 9px 14px;
    border-radius: 9px;
    margin-right: 6px;
}
QTabBar::tab:selected {
    background: #3b82f6;
    color: white;
}
QTextEdit {
    background: #020617;
    color: #dbeafe;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 8px;
    font-family: Consolas;
    font-size: 9pt;
}
QDialog {
    background: #0b1120;
}
"""


def cylinder_mesh_between(start: np.ndarray, end: np.ndarray, radius: float | tuple[float, float], sides: int = 24) -> gl.MeshData:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        length = 1.0
    axis = axis / length
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    if isinstance(radius, (int, float)):
        r_start = r_end = float(radius)
    else:
        r_start, r_end = float(radius[0]), float(radius[1])

    vertexes = []
    for ci, center in enumerate((start, end)):
        r = r_start if ci == 0 else r_end
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            vertexes.append(center + r * (math.cos(angle) * u + math.sin(angle) * v))
    vertexes.append(start)
    start_center = len(vertexes) - 1
    vertexes.append(end)
    end_center = len(vertexes) - 1

    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
        faces.append([start_center, j, i])
        faces.append([end_center, sides + i, sides + j])
    return gl.MeshData(vertexes=np.array(vertexes, dtype=float), faces=np.array(faces, dtype=int))

def box_mesh(
    center: np.ndarray,
    sx: float,
    sy: float,
    sz: float,
    axis_x: np.ndarray | None = None,
    axis_y: np.ndarray | None = None,
    axis_z: np.ndarray | None = None,
) -> gl.MeshData:
    """Create a rectangular box mesh centered at *center* with full sizes
    sx, sy, sz along the provided local axes (defaults to world axes)."""
    if axis_x is None:
        axis_x = np.array([1.0, 0.0, 0.0], dtype=float)
    if axis_y is None:
        axis_y = np.array([0.0, 1.0, 0.0], dtype=float)
    if axis_z is None:
        axis_z = np.array([0.0, 0.0, 1.0], dtype=float)

    h = 0.5
    corners = []
    for dx in (-h, h):
        for dy in (-h, h):
            for dz in (-h, h):
                corners.append(
                    center + dx * sx * axis_x + dy * sy * axis_y + dz * sz * axis_z
                )

    faces = [
        [0, 1, 2], [1, 3, 2],   # z-minus face
        [4, 6, 5], [5, 6, 7],   # z-plus  face
        [0, 4, 1], [1, 4, 5],   # x-minus face
        [2, 3, 6], [3, 7, 6],   # x-plus  face
        [0, 2, 4], [2, 6, 4],   # y-minus face
        [1, 5, 3], [3, 5, 7],   # y-plus  face
    ]
    return gl.MeshData(
        vertexes=np.array(corners, dtype=float),
        faces=np.array(faces, dtype=int),
    )

class Robot3DView(gl.GLViewWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundColor("#0b1120")
        self.opts["distance"] = 900
        self.opts["azimuth"] = -58
        self.opts["elevation"] = 24
        self.opts["center"] = QtGui.QVector3D(80.0, 0.0, 190.0)
        self.q = Q_HOME[:]
        self.path: list[list[float]] = []
        self.target_xyz: list[float] | None = None
        self._items: list[object] = []
        self._static_items: list[object] = []
        self._workspace_cloud_item: gl.GLScatterPlotItem | None = None
        self._build_scene()
        self.update_robot(self.q)

    def _build_scene(self) -> None:
        grid = gl.GLGridItem()
        grid.setSize(1000, 1000)
        grid.setSpacing(50, 50)
        grid.setColor((0.20, 0.27, 0.38, 0.34))
        self.addItem(grid)
        self._static_items.append(grid)

        axis_specs = [
            ((0, 0, 0), (180, 0, 0), (1.0, 0.20, 0.20, 1.0), "X"),
            ((0, 0, 0), (0, 180, 0), (0.18, 0.85, 0.38, 1.0), "Y"),
            ((0, 0, 0), (0, 0, 180), (0.25, 0.55, 1.0, 1.0), "Z"),
        ]
        for start, end, color, label in axis_specs:
            item = gl.GLLinePlotItem(pos=np.array([start, end], dtype=float), color=color, width=2, antialias=True)
            self.addItem(item)
            self._static_items.append(item)
            text = gl.GLTextItem(pos=np.array(end, dtype=float), text=label, color=color)
            self.addItem(text)
            self._static_items.append(text)
        base_text = gl.GLTextItem(pos=np.array([18.0, -45.0, 12.0]), text="Base", color=(0.92, 0.96, 1.0, 1.0))
        self.addItem(base_text)
        self._static_items.append(base_text)

        self._add_workspace_wireframe()

    def _add_workspace_wireframe(self) -> None:
        shell = gl.GLMeshItem(
            meshdata=workspace_shell_mesh(),
            color=(0.08, 0.34, 0.64, 0.026),
            smooth=True,
            shader="shaded",
            glOptions="additive",
        )
        self.addItem(shell)
        self._static_items.append(shell)

        for index, pts in enumerate(workspace_wireframe()):
            is_outer = index % 3 == 2 or len(pts) <= 4
            item = gl.GLLinePlotItem(
                pos=np.array(pts),
                color=(0.26, 0.66, 1.0, 0.26 if is_outer else 0.09),
                width=1,
                antialias=True,
            )
            item.setGLOptions("additive")
            self.addItem(item)
            self._static_items.append(item)

        for z, label in (
            (WORKSPACE_MIN_Z_MM, "workspace min"),
            (WORKSPACE_MAX_Z_MM, "workspace max"),
        ):
            text = gl.GLTextItem(
                pos=np.array([WORKSPACE_R_MAX_MM + 18.0, 0.0, z], dtype=float),
                text=label,
                color=(0.72, 0.88, 1.0, 0.95),
            )
            self.addItem(text)
            self._static_items.append(text)

        sample_step = max(1, len(WORKSPACE_CLOUD) // 4000)
        sample_item = gl.GLScatterPlotItem(
            pos=WORKSPACE_CLOUD[::sample_step],
            color=(0.36, 0.76, 1.0, 0.20),
            size=3.2,
            pxMode=True,
        )
        sample_item.setGLOptions("additive")
        self.addItem(sample_item)
        self._static_items.append(sample_item)
        self._workspace_cloud_item = sample_item

        ws_info = gl.GLTextItem(
            pos=np.array([0.0, 0.0, WORKSPACE_MAX_Z_MM + 40.0]),
            text=(f"workspace: r=[{WORKSPACE_R_MIN_MM:.0f}, {WORKSPACE_R_MAX_MM:.0f}] mm, "
                f"z=[{WORKSPACE_MIN_Z_MM:.0f}, {WORKSPACE_MAX_Z_MM:.0f}] mm  "
                f"({WORKSPACE_SAMPLE_COUNT} MC samples, {WORKSPACE_Z_BINS} Z-bins)"),
            color=(0.72, 0.88, 1.0, 0.95),
        )
        self.addItem(ws_info)
        self._static_items.append(ws_info)

    def clear_dynamic(self) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items = []

    def set_workspace_cloud_visible(self, visible: bool) -> None:
        if self._workspace_cloud_item is not None:
            self._workspace_cloud_item.setVisible(visible)

    def update_robot(
        self,
        q: list[float],
        path: list[list[float]] | None = None,
        target_xyz: list[float] | None = None,
    ) -> None:
        self.q = q[:]
        if path is not None:
            self.path = path
        self.target_xyz = target_xyz[:] if target_xyz is not None else None
        self.clear_dynamic()

        transforms = fkine_all(q)
        pts = np.array([[0.0, 0.0, 0.0]] + [[t[0][3], t[1][3], t[2][3]] for t in transforms], dtype=float)

        if self.path:
            path_item = gl.GLLinePlotItem(
                pos=np.array(self.path, dtype=float),
                color=(0.18, 0.46, 0.90, 0.62),
                width=2,
                antialias=True,
            )
            path_item.setGLOptions("translucent")
            self.addItem(path_item)
            self._items.append(path_item)

        if self.target_xyz is not None:
            target = np.array(self.target_xyz, dtype=float)
            marker = gl.GLScatterPlotItem(
                pos=np.array([target], dtype=float),
                color=(1.0, 0.18, 0.18, 1.0),
                size=13,
                pxMode=True,
            )
            self.addItem(marker)
            self._items.append(marker)
            cross_color = (1.0, 0.18, 0.18, 0.95)
            for axis in range(3):
                start = target.copy()
                end = target.copy()
                start[axis] -= 28.0
                end[axis] += 28.0
                line = gl.GLLinePlotItem(pos=np.array([start, end]), color=cross_color, width=3, antialias=True)
                self.addItem(line)
                self._items.append(line)
            text = gl.GLTextItem(pos=target + np.array([18.0, 18.0, 22.0]), text="XYZ target", color=cross_color)
            self.addItem(text)
            self._items.append(text)

        base_specs = [
            (np.array([0.0, 0.0, -34.0]), np.array([0.0, 0.0, -16.0]), 82.0, (0.06, 0.08, 0.12, 1.0)),
            (np.array([0.0, 0.0, -16.0]), np.array([0.0, 0.0, -4.0]), 60.0, (0.10, 0.13, 0.17, 1.0)),
            (np.array([0.0, 0.0, -4.0]), np.array([0.0, 0.0, 10.0]), 44.0, (0.72, 0.76, 0.80, 1.0)),
            (np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 24.0]), 34.0, (0.80, 0.84, 0.88, 1.0)),
        ]
        for bs, be, br, bcol in base_specs:
            base_item = gl.GLMeshItem(
                meshdata=cylinder_mesh_between(bs, be, br, sides=48),
                color=bcol,
                smooth=True,
                shader="shaded",
                glOptions="opaque",
            )
            self.addItem(base_item)
            self._items.append(base_item)

        # ---- Tapered links (wider at proximal joint, narrower at distal) ----
        link_colors = [
            (0.86, 0.68, 0.30, 1.0),   # warm bronze
            (0.90, 0.76, 0.38, 1.0),   # gold
            (0.80, 0.58, 0.28, 1.0),   # dark bronze
            (0.52, 0.62, 0.70, 1.0),   # steel blue
            (0.48, 0.60, 0.70, 1.0),   # cool steel
            (0.70, 0.78, 0.84, 1.0),   # light steel
        ]
        link_specs = [
            ((28.0, 17.0), 36),
            ((17.0, 13.0), 36),
            ((13.0, 10.0), 32),
            ((10.0, 8.0), 32),
            ((8.0, 6.5), 28),
            ((6.5, 5.0), 28),
        ]
        for i in range(len(pts) - 1):
            r_start, r_end = link_specs[i][0]
            sides = link_specs[i][1]
            link_item = gl.GLMeshItem(
                meshdata=cylinder_mesh_between(pts[i], pts[i + 1], (r_start, r_end), sides=sides),
                color=link_colors[i],
                smooth=True,
                shader="shaded",
                glOptions="opaque",
            )
            self.addItem(link_item)
            self._items.append(link_item)

        # ---- Motor housings + golden accent bands at each intermediate joint ----
        for i in range(1, len(pts) - 1):
            # Rotation axis = Z-column of the *previous* frame's transform
            z_axis = np.array(
                [transforms[i - 1][0][2], transforms[i - 1][1][2], transforms[i - 1][2][2]],
                dtype=float,
            )
            housing_r = max(15.0, 23.0 - i * 2.0)
            housing_half = 12.0            
            hs = pts[i] - z_axis * housing_half
            he = pts[i] + z_axis * housing_half
            housing = gl.GLMeshItem(
                meshdata=cylinder_mesh_between(hs, he, (housing_r, housing_r * 0.88), sides=32),
                color=(0.14, 0.17, 0.22, 1.0),
                smooth=True,
                shader="shaded",
                glOptions="opaque",
            )
            self.addItem(housing)
            self._items.append(housing)

            # Thin golden accent band around the housing
            band_half = 2.0
            band_r = housing_r + 2.5
            bs2 = pts[i] - z_axis * band_half
            be2 = pts[i] + z_axis * band_half
            band = gl.GLMeshItem(
                meshdata=cylinder_mesh_between(bs2, be2, band_r, sides=32),
                color=(0.92, 0.72, 0.18, 1.0),
                smooth=True,
                shader="shaded",
                glOptions="opaque",
            )
            self.addItem(band)
            self._items.append(band)

        # ---- Subtle silhouette trace (keeps arm visible at edge-on angles) ----
        self._add_link_silhouette(pts)

        # ---- Gripper assembly ----
        tcp = pts[-1]
        wrist_x = np.array([transforms[-1][0][0], transforms[-1][1][0], transforms[-1][2][0]], dtype=float)
        wrist_y = np.array([transforms[-1][0][1], transforms[-1][1][1], transforms[-1][2][1]], dtype=float)
        wrist_z = np.array([transforms[-1][0][2], transforms[-1][1][2], transforms[-1][2][2]], dtype=float)

        # Wrist body block sits behind the TCP and extends forward to fingers.
        wrist_block = gl.GLMeshItem(
            meshdata=box_mesh(tcp - wrist_x * 8.0, 16.0, 28.0, 16.0, wrist_x, wrist_y, wrist_z),
            color=(0.62, 0.70, 0.78, 1.0),
            smooth=True,
            shader="shaded",
            glOptions="opaque",
        )
        self.addItem(wrist_block)
        self._items.append(wrist_block)

        # Two flat finger plates
        for side in (-1.0, 1.0):
            finger_center = tcp + wrist_x * 22.0 + side * wrist_y * 9.0
            finger = gl.GLMeshItem(
                meshdata=box_mesh(finger_center, 28.0, 4.0, 12.0, wrist_x, wrist_y, wrist_z),
                color=(0.84, 0.90, 0.96, 1.0),
                smooth=True,
                shader="shaded",
                glOptions="opaque",
            )
            self.addItem(finger)
            self._items.append(finger)

        # Small TCP indicator sphere
        tcp_sphere = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=16, cols=24, radius=5.0),
            color=(0.88, 0.94, 1.0, 1.0),
            smooth=True,
            shader="shaded",
            glOptions="opaque",
        )
        tcp_sphere.translate(float(tcp[0]), float(tcp[1]), float(tcp[2]))
        self.addItem(tcp_sphere)
        self._items.append(tcp_sphere)

        tcp_label = gl.GLTextItem(pos=tcp + np.array([14.0, 10.0, 14.0]), text="TCP", color=(0.82, 0.92, 1.0, 0.92))
        self.addItem(tcp_label)
        self._items.append(tcp_label)

        tcp_rot = np.array([[transforms[-1][r][c] for c in range(3)] for r in range(3)], dtype=float)
        for axis, color in ((0, (1.0, 0.24, 0.24, 0.86)), (1, (0.22, 0.86, 0.36, 0.86)), (2, (0.26, 0.56, 1.0, 0.86))):
            line = np.array([tcp, tcp + tcp_rot[:, axis] * 34.0])
            item = gl.GLLinePlotItem(pos=line, color=color, width=2, antialias=True)
            item.setGLOptions("translucent")
            self.addItem(item)
            self._items.append(item)

    def _add_link_silhouette(self, pts: np.ndarray) -> None:
        # A subtle always-readable center trace keeps the arm visible when OpenGL lighting
        # makes a shaded cylinder blend into the dark viewport at edge-on camera angles.
        trace = gl.GLLinePlotItem(
            pos=pts,
            color=(1.0, 0.84, 0.52, 0.58),
            width=3,
            antialias=True,
        )
        trace.setGLOptions("additive")
        self.addItem(trace)
        self._items.append(trace)

        joint_dots = gl.GLScatterPlotItem(
            pos=pts,
            color=(0.96, 0.98, 1.0, 0.62),
            size=5,
            pxMode=True,
        )
        joint_dots.setGLOptions("additive")
        self.addItem(joint_dots)
        self._items.append(joint_dots)

    def _fit_camera_to_scene(self, pts: np.ndarray) -> None:
        scene_pts = [pts]
        if self.target_xyz is not None:
            scene_pts.append(np.array([self.target_xyz], dtype=float))
        cloud = np.vstack(scene_pts)
        mins = cloud.min(axis=0)
        maxs = cloud.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(np.linalg.norm(maxs - mins))
        self.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
        if span > 1.0:
            self.opts["distance"] = max(520.0, min(float(self.opts.get("distance", 900)), span * 2.15))

    def set_camera(self, azimuth: int, elevation: int, distance: int) -> None:
        self.setCameraPosition(azimuth=azimuth, elevation=elevation, distance=distance)


class ProfileWindow(QtWidgets.QDialog):
    def __init__(self, trajectory: Trajectory, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trajectory Profiles")
        self.resize(1360, 900)
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        profiles = [
            ("Position", trajectory.q_rad, "rad"),
            ("Velocity", trajectory.q_vel, "rad/s"),
            ("Acceleration", trajectory.q_acc, "rad/s^2"),
            ("Jerk", trajectory.q_jerk, "rad/s^3"),
        ]
        for title, rows, unit in profiles:
            plot = pg.PlotWidget(background="#0f172a")
            plot.getPlotItem().setClipToView(False)
            plot.getPlotItem().setDownsampling(auto=False)
            plot.getPlotItem().showAxis("bottom", True)
            plot.getPlotItem().showAxis("left", True)
            plot.showGrid(x=True, y=True, alpha=0.35)
            plot.setTitle(f"{title} vs Time", color="#f8fafc", size="12pt")
            plot.setLabel("bottom", "Time", units="s")
            plot.setLabel("left", title, units=unit)
            bottom_axis = plot.getAxis("bottom")
            bottom_axis.setTextPen("#f8fafc")
            bottom_axis.setPen("#94a3b8")
            bottom_axis.enableAutoSIPrefix(False)
            bottom_axis.setStyle(showValues=True, tickTextOffset=8)
            bottom_axis.setTickFont(QtGui.QFont("Segoe UI", 9))
            left_axis = plot.getAxis("left")
            left_axis.setTextPen("#f8fafc")
            left_axis.setPen("#94a3b8")
            left_axis.setStyle(showValues=True, tickTextOffset=8)
            left_axis.setTickFont(QtGui.QFont("Segoe UI", 9))
            plot.setXRange(0.0, max(0.001, trajectory.duration_s), padding=0.02)
            plot.addLegend(offset=(10, 10))
            for i in range(6):
                values = [row[i] for row in rows]
                plot.plot(trajectory.time_s, values, pen=pg.mkPen(pg.intColor(i, hues=6), width=1.6), name=f"J{i + 1}")
            tabs.addTab(plot, title)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEM343 Robot Arm Studio")
        self.resize(1420, 860)
        self.q = Q_HOME[:]
        self.trajectory: Trajectory | None = None
        self.serial_link = SerialLink()
        self.animation_index = 0
        self.hardware_mode = False
        self.hardware_q = Q_HOME[:]
        self.target_xyz: list[float] | None = None
        self._syncing_ui = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._animation_tick)
        self._build_ui()
        self.set_q(Q_HOME)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_layout = QtWidgets.QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        sidebar = QtWidgets.QFrame(objectName="Sidebar")
        sidebar.setMinimumWidth(430)
        sidebar.setMaximumWidth(540)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)
        title = QtWidgets.QLabel("EEM343 Robot Arm", objectName="Title")
        subtitle = QtWidgets.QLabel("Modern Qt simulation and hardware control", objectName="Subtitle")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)

        tabs = QtWidgets.QTabWidget()
        side_layout.addWidget(tabs, 1)
        manual_tab = QtWidgets.QWidget()
        sequence_tab = QtWidgets.QWidget()
        tools_tab = QtWidgets.QWidget()
        tabs.addTab(manual_tab, "Manual")
        tabs.addTab(sequence_tab, "Sequence")
        tabs.addTab(tools_tab, "Tools")
        self._build_manual_tab(manual_tab)
        self._build_sequence_tab(sequence_tab)
        self._build_tools_tab(tools_tab)
        main_layout.addWidget(sidebar)

        content = QtWidgets.QVBoxLayout()
        content.setSpacing(12)
        view_card = QtWidgets.QFrame(objectName="ViewportCard")
        view_layout = QtWidgets.QVBoxLayout(view_card)
        view_layout.setContentsMargins(12, 12, 12, 12)
        view_layout.setSpacing(8)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel("3D View", objectName="Section"))
        self.workspace_cloud_check = QtWidgets.QCheckBox("Workspace cloud")
        self.workspace_cloud_check.setChecked(True)
        self.workspace_cloud_check.toggled.connect(self._workspace_cloud_toggled)
        toolbar.addWidget(self.workspace_cloud_check)
        toolbar.addStretch(1)
        self.az_slider = self._small_slider(-180, 180, -58, self._camera_changed)
        self.el_slider = self._small_slider(-70, 70, 24, self._camera_changed)
        self.dist_slider = self._small_slider(450, 1400, 900, self._camera_changed)
        for label, slider in (("Azimuth", self.az_slider), ("Elevation", self.el_slider), ("Distance", self.dist_slider)):
            toolbar.addWidget(QtWidgets.QLabel(label))
            toolbar.addWidget(slider)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setObjectName("Secondary")
        reset_btn.clicked.connect(self._reset_camera)
        toolbar.addWidget(reset_btn)
        view_layout.addLayout(toolbar)
        self.view = Robot3DView()
        view_layout.addWidget(self.view, 1)
        content.addWidget(view_card, 1)

        self.reach_label = QtWidgets.QLabel(objectName="BadgeGood")
        self.reach_label.setWordWrap(True)
        content.addWidget(self.reach_label)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("StatusCard")
        self.status_label.setMinimumHeight(86)
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        content.addWidget(self.status_label)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(130)
        content.addWidget(self.log_box)
        main_layout.addLayout(content, 1)

    def _card_layout(self, parent: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(10)
        return layout

    def _build_manual_tab(self, parent: QtWidgets.QWidget) -> None:
        layout = self._card_layout(parent)
        layout.addWidget(QtWidgets.QLabel("Joint Control", objectName="Section"))
        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_values: list[QtWidgets.QLabel] = []
        for i in range(6):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"J{i + 1}"))
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            if i == 1:
                slider.setRange(int(-math.degrees(Q_MAX[i]) * 10), int(-math.degrees(Q_MIN[i]) * 10))
            else:
                slider.setRange(int(math.degrees(Q_MIN[i]) * 10), int(math.degrees(Q_MAX[i]) * 10))
            slider.valueChanged.connect(self._joint_slider_changed)
            value = QtWidgets.QLabel("0.0")
            value.setFixedWidth(48)
            row.addWidget(slider, 1)
            row.addWidget(value)
            layout.addLayout(row)
            self.joint_sliders.append(slider)
            self.joint_values.append(value)
        layout.addWidget(QtWidgets.QLabel("Cartesian IK", objectName="Section"))
        self.cart_inputs = {}
        form = QtWidgets.QFormLayout()
        for name, value in (("X mm", 0.0), ("Y mm", 0.0), ("Z mm", 0.0), ("Pitch deg", -90.0)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-900, 900)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.valueChanged.connect(self._cartesian_input_changed)
            self.cart_inputs[name] = spin
            form.addRow(name, spin)
        layout.addLayout(form)
        move_btn = QtWidgets.QPushButton("Move to XYZ / Pitch")
        move_btn.clicked.connect(self.apply_cartesian_target)
        layout.addWidget(move_btn)
        home_btn = QtWidgets.QPushButton("Home")
        home_btn.setObjectName("Secondary")
        home_btn.clicked.connect(self.smooth_home)
        layout.addWidget(home_btn)
        send_btn = QtWidgets.QPushButton("Send Current Pose")
        send_btn.setObjectName("Secondary")
        send_btn.clicked.connect(self.send_current_pose)
        layout.addWidget(send_btn)
        layout.addStretch(1)

    def _build_sequence_tab(self, parent: QtWidgets.QWidget) -> None:
        layout = self._card_layout(parent)
        layout.addWidget(QtWidgets.QLabel("Execution Mode", objectName="Section"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Simulate only", "Simulate + physical robot"])
        layout.addWidget(self.mode_combo)
        self.path_combo = QtWidgets.QComboBox()
        self.path_combo.addItems(list(TRAJECTORY_BUILDERS.keys()))
        self.link_type_combo = QtWidgets.QComboBox()
        self.link_type_combo.addItems(["USB cable to Arduino Nano", "PC Bluetooth serial adapter"])
        self.link_type_combo.currentIndexChanged.connect(self._set_link_defaults)
        self.port_input = QtWidgets.QComboBox()
        self.port_input.setEditable(True)
        self.port_input.setMinimumWidth(150)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setObjectName("Secondary")
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(self.port_input, 1)
        port_row.addWidget(refresh_btn)
        self.baud_input = QtWidgets.QComboBox()
        self.baud_input.addItems(["1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"])
        self.fps_input = QtWidgets.QSpinBox()
        self.fps_input.setRange(10, 60)
        self.fps_input.setValue(USB_FRAME_RATE_FPS)
        self.max_joint_speed_input = QtWidgets.QDoubleSpinBox()
        self.max_joint_speed_input.setRange(3.0, 220.0)
        self.max_joint_speed_input.setDecimals(1)
        self.max_joint_speed_input.setValue(DEFAULT_MAX_JOINT_SPEED_DEG_S)
        self.max_joint_speed_input.setSuffix(" deg/s")
        self.max_cart_speed_input = QtWidgets.QDoubleSpinBox()
        self.max_cart_speed_input.setRange(5.0, 160.0)
        self.max_cart_speed_input.setDecimals(1)
        self.max_cart_speed_input.setValue(DEFAULT_MAX_CART_SPEED_MM_S)
        self.max_cart_speed_input.setSuffix(" mm/s")
        self.total_time_input = QtWidgets.QDoubleSpinBox()
        self.total_time_input.setRange(0.0, 300.0)
        self.total_time_input.setDecimals(1)
        self.total_time_input.setSingleStep(1.0)
        self.total_time_input.setSpecialValueText("Auto")
        self.total_time_input.setSuffix(" s")
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.addRow("Path", self.path_combo)
        form.addRow("Connection", self.link_type_combo)
        form.addRow("Port", port_row)
        form.addRow("Baud", self.baud_input)
        form.addRow("Frame rate", self.fps_input)
        form.addRow("Max joint speed", self.max_joint_speed_input)
        form.addRow("Max XYZ speed", self.max_cart_speed_input)
        form.addRow("Total trajectory time", self.total_time_input)
        layout.addLayout(form)
        note = QtWidgets.QLabel("USB uses 115200. HC-05 Bluetooth uses 9600. Both send the same H/R/S/?/F commands.")
        note.setWordWrap(True)
        note.setObjectName("Subtitle")
        layout.addWidget(note)
        connect_btn = QtWidgets.QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_serial)
        home_btn = QtWidgets.QPushButton("Smooth Home")
        home_btn.setObjectName("Secondary")
        home_btn.clicked.connect(self.smooth_home)
        reset_btn = QtWidgets.QPushButton("Reset Stop")
        reset_btn.setObjectName("Secondary")
        reset_btn.clicked.connect(self.reset_stop)
        gen_btn = QtWidgets.QPushButton("Generate Trajectory")
        gen_btn.clicked.connect(self.generate_trajectory)
        run_btn = QtWidgets.QPushButton("Run Selected Mode")
        run_btn.clicked.connect(self.run_sequence)
        plots_btn = QtWidgets.QPushButton("Open Profiles")
        plots_btn.setObjectName("Secondary")
        plots_btn.clicked.connect(self.show_plots)

        button_grid = QtWidgets.QGridLayout()
        button_grid.setHorizontalSpacing(8)
        button_grid.setVerticalSpacing(8)
        grid_buttons = [
            connect_btn,
            home_btn,
            reset_btn,
            gen_btn,
            run_btn,
            plots_btn,
        ]
        for index, button in enumerate(grid_buttons):
            button.setMinimumHeight(44)
            button_grid.addWidget(button, index // 3, index % 3)
        layout.addLayout(button_grid)

        self._refresh_ports()
        self._set_link_defaults()
        layout.addStretch(1)
        stop_btn = QtWidgets.QPushButton("Stop")
        stop_btn.setObjectName("Danger")
        stop_btn.setMinimumHeight(46)
        stop_btn.clicked.connect(self.stop_robot)
        layout.addWidget(stop_btn)

    def _build_tools_tab(self, parent: QtWidgets.QWidget) -> None:
        layout = self._card_layout(parent)
        export_btn = QtWidgets.QPushButton("Export Trajectory CSV")
        export_btn.clicked.connect(self.export_csv)
        layout.addWidget(export_btn)
        status_btn = QtWidgets.QPushButton("Query Robot Status")
        status_btn.setObjectName("Secondary")
        status_btn.clicked.connect(self.query_status)
        layout.addWidget(status_btn)
        disconnect_btn = QtWidgets.QPushButton("Disconnect Serial")
        disconnect_btn.setObjectName("Secondary")
        disconnect_btn.clicked.connect(self.disconnect_serial)
        layout.addWidget(disconnect_btn)
        layout.addStretch(1)

    def _selected_port(self) -> str:
        return self.port_input.currentText().strip()

    def _refresh_ports(self) -> None:
        current = self._selected_port() if hasattr(self, "port_input") else ""
        self.port_input.blockSignals(True)
        self.port_input.clear()
        ports: list[str] = []
        if list_ports is not None:
            ports = [port.device for port in list_ports.comports()]
        if not ports:
            ports = [current or "COM5"]
        self.port_input.addItems(ports)
        if current:
            index = self.port_input.findText(current)
            if index >= 0:
                self.port_input.setCurrentIndex(index)
            else:
                self.port_input.setEditText(current)
        self.port_input.blockSignals(False)

    def _set_link_defaults(self) -> None:
        baud = "115200" if self.link_type_combo.currentIndex() == 0 else "9600"
        index = self.baud_input.findText(baud)
        if index >= 0:
            self.baud_input.setCurrentIndex(index)
        if hasattr(self, "fps_input"):
            self.fps_input.setValue(USB_FRAME_RATE_FPS if self.link_type_combo.currentIndex() == 0 else BLUETOOTH_FRAME_RATE_FPS)

    def _change_baud(self, direction: int) -> None:
        next_index = clamp(self.baud_input.currentIndex() + direction, 0, self.baud_input.count() - 1)
        self.baud_input.setCurrentIndex(int(next_index))

    def _small_slider(self, low: int, high: int, value: int, callback) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setFixedWidth(120)
        slider.valueChanged.connect(callback)
        return slider

    def _camera_changed(self) -> None:
        self.view.set_camera(self.az_slider.value(), self.el_slider.value(), self.dist_slider.value())

    def _workspace_cloud_toggled(self, checked: bool) -> None:
        self.view.set_workspace_cloud_visible(checked)

    def _reset_camera(self) -> None:
        self.az_slider.setValue(-58)
        self.el_slider.setValue(24)
        self.dist_slider.setValue(900)
        self._camera_changed()

    def log(self, message: str) -> None:
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def set_q(self, q: list[float]) -> None:
        self.q = [max(Q_MIN[i], min(Q_MAX[i], q[i])) for i in range(6)]
        self._syncing_ui = True
        for i, slider in enumerate(self.joint_sliders):
            slider.blockSignals(True)
            slider_degrees = -math.degrees(self.q[i]) if i == 1 else math.degrees(self.q[i])
            slider.setValue(int(slider_degrees * 10))
            slider.blockSignals(False)
            self.joint_values[i].setText(f"{math.degrees(self.q[i]):.1f}")
        xyz, rpy = pose_from_q(self.q)
        values = {"X mm": xyz[0], "Y mm": xyz[1], "Z mm": xyz[2], "Pitch deg": math.degrees(rpy[1])}
        for key, spin in self.cart_inputs.items():
            spin.blockSignals(True)
            spin.setValue(values[key])
            spin.blockSignals(False)
        self._syncing_ui = False
        self.update_display()

    def update_display(self, path: list[list[float]] | None = None) -> None:
        display_path = [] if path is None and self.target_xyz is None else path
        self.view.update_robot(self.q, display_path, self.target_xyz)
        xyz, rpy = pose_from_q(self.q)
        pwm = joint_angles_to_pwm_us(self.q)
        reachable, reach_text = workspace_status(xyz)
        self.reach_label.setObjectName("BadgeGood" if reachable else "BadgeBad")
        self.reach_label.setText(reach_text)
        self.reach_label.style().unpolish(self.reach_label)
        self.reach_label.style().polish(self.reach_label)
        target_text = ""
        if self.target_xyz is not None:
            target_error = math.sqrt(sum((self.target_xyz[i] - xyz[i]) ** 2 for i in range(3)))
            target_text = (
                f"\nXYZ target: {self.target_xyz[0]:.1f}, {self.target_xyz[1]:.1f}, {self.target_xyz[2]:.1f} mm"
                f"     final error: {target_error:.1f} mm"
            )
        self.status_label.setText(
            f"FK XYZ: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f} mm     "
            f"RPY: {math.degrees(rpy[0]):.1f}, {math.degrees(rpy[1]):.1f}, {math.degrees(rpy[2]):.1f} deg\n"
            f"PWM us: {', '.join(str(v) for v in pwm)}     "
            f"Serial: {'connected' if self.serial_link.is_open else 'not connected'}"
            f"{target_text}"
        )

    def _joint_slider_changed(self) -> None:
        if self._syncing_ui:
            return
        self.q = [
            math.radians((-slider.value() if i == 1 else slider.value()) / 10.0)
            for i, slider in enumerate(self.joint_sliders)
        ]
        self.target_xyz = None
        for i, value in enumerate(self.joint_values):
            value.setText(f"{math.degrees(self.q[i]):.1f}")
        self.update_display()

    def _cartesian_target_values(self) -> tuple[list[float], float]:
        return (
            [self.cart_inputs["X mm"].value(), self.cart_inputs["Y mm"].value(), self.cart_inputs["Z mm"].value()],
            math.radians(self.cart_inputs["Pitch deg"].value()),
        )

    def _target_workspace_status(self, target: list[float]) -> tuple[bool, str]:
        return workspace_status(target)

    def _show_target_workspace_warning(self, target: list[float], reach_text: str) -> None:
        message = (
            f"XYZ target {target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f} mm is outside the robot workspace.\n\n"
            f"{reach_text}\n\n"
            "Choose a coordinate inside the workspace cloud/envelope before moving."
        )
        QtWidgets.QMessageBox.warning(self, "Workspace limit", message)
        self.log(f"Workspace warning: {reach_text}")

    def _cartesian_preview_path(self, target: list[float], pitch: float) -> list[list[float]]:
        try:
            start_xyz, start_rpy = pose_from_q(self.q)
            start = PoseTarget(start_xyz[0], start_xyz[1], start_xyz[2], start_rpy[1])
            end = PoseTarget(target[0], target[1], target[2], pitch)
            rows = []
            seed = self.q[:]
            for step_index in range(25):
                pose = interpolate_pose(start, end, step_index / 24.0)
                ok, q_sol, _reason, err = solve_ik([pose.x, pose.y, pose.z], seed, pose.pitch, max_iter=60)
                if not ok and err > 35.0:
                    break
                rows.append(pose_from_q(q_sol)[0])
                seed = q_sol
            return rows or [start_xyz, target]
        except Exception:
            return [pose_from_q(self.q)[0], target]

    def _cartesian_input_changed(self) -> None:
        if self._syncing_ui:
            return
        target, pitch = self._cartesian_target_values()
        reachable, reach_text = self._target_workspace_status(target)
        if not reachable:
            self.target_xyz = target[:]
            self.update_display([pose_from_q(self.q)[0], target])
            self.reach_label.setObjectName("BadgeBad")
            self.reach_label.setText(f"{reach_text} | target outside workspace")
            self.reach_label.style().unpolish(self.reach_label)
            self.reach_label.style().polish(self.reach_label)
            return
        ok, q_sol, _reason, _err = solve_ik(target, self.q, pitch)
        self.target_xyz = target[:]
        self.update_display(self._cartesian_preview_path(target, pitch))
        note = joint_limit_note(q_sol)
        if note:
            self.reach_label.setText(f"{self.reach_label.text()} | IK limit: {note}")
        if not ok:
            self.reach_label.setText(f"{self.reach_label.text()} | nearest IK preview")

    def apply_cartesian_target(self) -> None:
        target, pitch = self._cartesian_target_values()
        reachable, reach_text = self._target_workspace_status(target)
        if not reachable:
            self.target_xyz = target[:]
            self.update_display([pose_from_q(self.q)[0], target])
            self.reach_label.setObjectName("BadgeBad")
            self.reach_label.setText(f"{reach_text} | move blocked")
            self.reach_label.style().unpolish(self.reach_label)
            self.reach_label.style().polish(self.reach_label)
            self._show_target_workspace_warning(target, reach_text)
            return
        ok, q_sol, reason, err = solve_ik(target, self.q, pitch)
        self.target_xyz = target[:]
        preview_path = self._cartesian_preview_path(target, pitch)
        self.set_q(q_sol)
        self.update_display(preview_path)
        xyz, _rpy = pose_from_q(q_sol)
        final_error = math.sqrt(sum((target[i] - xyz[i]) ** 2 for i in range(3)))
        note = joint_limit_note(q_sol)
        limit_text = f"; {note}" if note else ""
        self.log(f"{reason}; target error={final_error:.2f} mm; weighted error={err:.2f}{limit_text}")
        if not ok:
            QtWidgets.QMessageBox.warning(self, "IK result", "The nearest reachable pose is displayed.")

    def connect_serial(self) -> None:
        try:
            line = self.serial_link.connect(self._selected_port(), int(self.baud_input.currentText()))
            self.log(f"Serial connected: {line}")
            self.update_display()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Serial", str(exc))
            self.log(f"Serial error: {exc}")

    def reset_stop(self) -> None:
        if not self.serial_link.is_open:
            QtWidgets.QMessageBox.information(self, "Serial", "Connect to the robot before resetting the stop latch.")
            return
        try:
            self.log(f"Arduino: {self.serial_link.command('R')}")
        except Exception as exc:
            self.log(f"Reset failed: {exc}")

    def stream_trajectory(self, trajectory: Trajectory, label: str) -> None:
        self.trajectory = trajectory
        self.update_display(trajectory.ee_xyz)
        if not self.serial_link.is_open:
            self.set_q(trajectory.q_rad[-1])
            self.update_display(trajectory.ee_xyz)
            self.log(f"{label}: simulation only")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            clear_reply = self.serial_link.clear_queue()
            if clear_reply.startswith("ESTOP") or clear_reply.startswith("STOPPED"):
                self.log(f"Arduino: {clear_reply}")
                return

            next_upload_index = 0
            initial_frames = min(ARDUINO_QUEUE_PRELOAD_FRAMES, len(trajectory.q_rad))
            while next_upload_index < initial_frames:
                reply = self.serial_link.enqueue_frame(
                    trajectory.pwm_us[next_upload_index],
                    trajectory.frame_dt_ms(next_upload_index),
                )
                if reply.startswith("OKQ"):
                    next_upload_index += 1
                    continue
                self.log(f"Arduino: {reply}")
                return

            play_reply = self.serial_link.start_queue()
            if not play_reply.startswith("OK,PLAY"):
                self.log(f"Arduino: {play_reply}")
                return

            start_s = time.perf_counter()
            display_index = 0
            stopped_by_robot = False
            while display_index < len(trajectory.q_rad) - 1 or next_upload_index < len(trajectory.q_rad):
                for _ in range(ARDUINO_QUEUE_REFILL_BURST):
                    if next_upload_index >= len(trajectory.q_rad):
                        break
                    reply = self.serial_link.enqueue_frame(
                        trajectory.pwm_us[next_upload_index],
                        trajectory.frame_dt_ms(next_upload_index),
                    )
                    if reply.startswith("OKQ"):
                        next_upload_index += 1
                        continue
                    if "Q_FULL" in reply:
                        break
                    if reply.startswith("ESTOP") or reply.startswith("STOPPED"):
                        self.log(f"Arduino: {reply}")
                        stopped_by_robot = True
                        break
                    self.log(f"Arduino: {reply}")
                    stopped_by_robot = True
                    break

                if stopped_by_robot:
                    break

                elapsed_s = time.perf_counter() - start_s
                while (
                    display_index + 1 < len(trajectory.time_s)
                    and trajectory.time_s[display_index + 1] <= elapsed_s
                ):
                    display_index += 1
                self.set_q(trajectory.q_rad[display_index])
                self.view.update_robot(self.q, trajectory.ee_xyz)
                QtWidgets.QApplication.processEvents()
                time.sleep(0.01)

            if not stopped_by_robot:
                while time.perf_counter() - start_s < trajectory.stream_duration_s:
                    elapsed_s = time.perf_counter() - start_s
                    while (
                        display_index + 1 < len(trajectory.time_s)
                        and trajectory.time_s[display_index + 1] <= elapsed_s
                    ):
                        display_index += 1
                    self.set_q(trajectory.q_rad[display_index])
                    self.view.update_robot(self.q, trajectory.ee_xyz)
                    QtWidgets.QApplication.processEvents()
                    time.sleep(0.01)

            self.hardware_q = trajectory.q_rad[min(len(trajectory.q_rad) - 1, display_index)][:]
            self.log(
                f"{label}: queued {next_upload_index}/{len(trajectory.q_rad)} frames "
                f"with Arduino-side playback"
            )
        except Exception as exc:
            self.log(f"{label} failed: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def smooth_home(self) -> None:
        self.target_xyz = None
        trajectory = joint_motion_trajectory(
            self.hardware_q if self.serial_link.is_open else self.q,
            Q_HOME,
            self.fps_input.value(),
            self.max_joint_speed_input.value(),
        )
        self.stream_trajectory(trajectory, "Smooth home")

    def send_current_pose(self) -> None:
        if not self.serial_link.is_open:
            QtWidgets.QMessageBox.information(self, "Serial", "Connect to the robot before sending a pose.")
            return
        try:
            trajectory = joint_motion_trajectory(
                self.hardware_q,
                self.q,
                self.fps_input.value(),
                self.max_joint_speed_input.value(),
            )
            self.stream_trajectory(trajectory, "Manual pose")
        except Exception as exc:
            self.log(f"Send failed: {exc}")

    def generate_trajectory(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.trajectory = calculate_trajectory(
                fps=self.fps_input.value(),
                max_joint_speed_deg_s=self.max_joint_speed_input.value(),
                max_cart_speed_mm_s=self.max_cart_speed_input.value(),
                target_duration_s=self.total_time_input.value() if self.total_time_input.value() > 0.0 else None,
                path_name=self.path_combo.currentText(),
            )
            self.update_display(self.trajectory.ee_xyz)
            time_note = "auto" if self.total_time_input.value() <= 0.0 else f"requested {self.total_time_input.value():.1f} s"
            self.log(
                f"Trajectory ready: {len(self.trajectory.q_rad)} frames, "
                f"{self.trajectory.duration_s:.3f} s planned, "
                f"{self.trajectory.stream_duration_s:.3f} s streamed ({time_note}), "
                f"path {self.path_combo.currentText()}, "
                f"max joint speed {self.max_joint_speed_input.value():.1f} deg/s"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Trajectory", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def run_sequence(self) -> None:
        if self.trajectory is None:
            self.generate_trajectory()
        if self.trajectory is None:
            return
        self.hardware_mode = self.mode_combo.currentIndex() == 1
        if self.hardware_mode and not self.serial_link.is_open:
            QtWidgets.QMessageBox.warning(self, "Hardware", "Connect serial before simulation + physical robot.")
            return
        if self.hardware_mode:
            self.stream_trajectory(self.trajectory, "Run selected mode")
            return
        self.animation_index = 0
        self.timer.start(1)
        self.log("Running " + self.mode_combo.currentText())

    def _animation_tick(self) -> None:
        if self.trajectory is None:
            self.timer.stop()
            return
        if self.animation_index >= len(self.trajectory.q_rad):
            self.timer.stop()
            self.log("Sequence complete")
            return
        self.set_q(self.trajectory.q_rad[self.animation_index])
        self.view.update_robot(self.q, self.trajectory.ee_xyz)
        if self.hardware_mode:
            try:
                reply = self.serial_link.send_frame(
                    self.trajectory.pwm_us[self.animation_index],
                    self.trajectory.frame_dt_ms(self.animation_index),
                )
                self.hardware_q = self.q[:]
                if reply.startswith("ESTOP") or reply.startswith("STOPPED"):
                    self.timer.stop()
                    self.log(f"Arduino: {reply}")
            except Exception as exc:
                self.timer.stop()
                self.log(f"Hardware stream failed: {exc}")
        self.animation_index += 1
        if self.animation_index < len(self.trajectory.q_rad):
            self.timer.start(1 if self.hardware_mode else self.trajectory.frame_dt_ms(self.animation_index - 1))

    def stop_robot(self) -> None:
        self.timer.stop()
        if self.serial_link.is_open:
            try:
                self.log(f"Arduino: {self.serial_link.command('S')}")
            except Exception as exc:
                self.log(f"Stop failed: {exc}")
        else:
            self.log("Simulation stopped")

    def show_plots(self) -> None:
        if self.trajectory is None:
            QtWidgets.QMessageBox.information(self, "Profiles", "Generate a trajectory first.")
            return
        dialog = ProfileWindow(self.trajectory, self)
        dialog.exec()

    def export_csv(self) -> None:
        if self.trajectory is None:
            self.generate_trajectory()
        if self.trajectory is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save trajectory CSV", "eem343_planned_trajectory.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s"]
                + [f"q{i + 1}_deg" for i in range(6)]
                + [f"dq{i + 1}_rad_s" for i in range(6)]
                + [f"ddq{i + 1}_rad_s2" for i in range(6)]
                + [f"dddq{i + 1}_rad_s3" for i in range(6)]
                + [f"servo{i + 1}_us" for i in range(6)]
                + ["ik_error"]
            )
            for t, q, vel, acc, jerk, pwm, err in zip(
                self.trajectory.time_s,
                self.trajectory.q_rad,
                self.trajectory.q_vel,
                self.trajectory.q_acc,
                self.trajectory.q_jerk,
                self.trajectory.pwm_us,
                self.trajectory.ik_errors,
            ):
                writer.writerow(
                    [f"{t:.4f}"]
                    + [f"{math.degrees(v):.3f}" for v in q]
                    + [f"{v:.5f}" for v in vel]
                    + [f"{v:.5f}" for v in acc]
                    + [f"{v:.5f}" for v in jerk]
                    + pwm
                    + [f"{err:.3f}"]
                )
        self.log(f"Wrote CSV: {path}")

    def query_status(self) -> None:
        if not self.serial_link.is_open:
            QtWidgets.QMessageBox.information(self, "Serial", "Serial is not connected.")
            return
        self.log(f"Arduino: {self.serial_link.command('?')}")

    def disconnect_serial(self) -> None:
        self.serial_link.close()
        self.update_display()
        self.log("Serial disconnected")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.serial_link.close()
        event.accept()


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
