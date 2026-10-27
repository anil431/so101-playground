"""Draw a shape on the table with the SO-101 follower arm (no leader needed).

Three steps, run from the repo folder inside the lerobot conda environment:

  1. python scripts/draw_shape.py preview
       No robot needed. Plans the path, checks it is reachable and saves preview.png.

  2. python scripts/draw_shape.py locate
       The arm goes limp, you put the chalk tip on the table where the CENTRE of
       the shape should be and press ENTER. The spot is saved to surface.json.

  3. python scripts/draw_shape.py draw
       The arm draws the shape around that spot and returns to where it started.

Defaults: a 10 cm square. Other shapes: --shape rectangle|triangle|pentagon
(or their ID number), --size in cm. Press Ctrl+C at any time to stop: the arm
lifts the chalk and goes back to its start pose.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_kinematics as kin  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
SURFACE_FILE = REPO_DIR / "surface.json"

# Fine tuning: added to the angles reported by the robot before they go into the
# arm model. Leave at 0 unless `locate` shows clearly wrong numbers in the
# calibration pose (upper arm vertical, forearm horizontal should read ~0 everywhere).
JOINT_OFFSETS_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}

RATE_HZ = 30
LIMIT_MARGIN_DEG = 3.0
# A neutral pose with the gripper well above the table, used as a stop-over
# when a direct move from the start pose would pass too close to the table.
READY_POSE_DEG = np.array([0.0, -30.0, 30.0, 60.0, 0.0])
IK_SEED_DEG = np.array([0.0, 0.0, 20.0, 60.0, 0.0])
SIDE_SEED_DEG = np.array([0.0, 45.0, 30.0, -75.0, 0.0])


# --------------------------------------------------------------------------- pen / chalk mounting


def make_tool(args, joints_deg=None) -> dict:
    """Describe how the pen sits on the gripper.

    inline: held between the jaws, pointing where the gripper points; drawn
            leaning `--tilt` degrees from vertical.
    side:   strapped across the gripper (gripper roughly horizontal, pen
            pointing down). Its direction is learned in `locate`: whatever
            points straight down when you press ENTER is taken as the pen, and
            the gripper keeps that same attitude while drawing.
    """
    stickout_m = args.stickout / 100.0
    if args.mount == "inline":
        return {"mount": "inline", "stickout_m": stickout_m, "tilt_deg": args.tilt, "dir": [0.0, 0.0, 1.0],
                "roll_deg": 0.0, "seed_deg": IK_SEED_DEG.tolist()}
    q = SIDE_SEED_DEG if joints_deg is None else np.asarray(joints_deg, dtype=float)
    down_in_gripper = kin.gripper_pose(q)[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    return {"mount": "side", "stickout_m": stickout_m, "tilt_deg": 0.0, "dir": down_in_gripper.tolist(),
            "roll_deg": float(q[4]), "seed_deg": q.tolist()}


def tool_fk(joints_deg, tool):
    return kin.forward(joints_deg, tool["stickout_m"], tool["dir"])


def tool_ik(target, seed_deg, tool):
    return kin.inverse(target, seed_deg, tilt_deg=tool["tilt_deg"], stickout=tool["stickout_m"],
                       roll_deg=tool["roll_deg"], tool_dir=tool["dir"])


def describe_tool(tool) -> str:
    if tool["mount"] == "side":
        return f"pen strapped across the gripper, tip {tool['stickout_m'] * 100:.1f} cm below it"
    return f"pen/chalk between the jaws, {tool['stickout_m'] * 100:.1f} cm sticking out, leaning {tool['tilt_deg']:.0f} deg"


# --------------------------------------------------------------------------- shapes


def _regular_polygon(n_sides: int, side: float) -> np.ndarray:
    radius = side / (2 * np.sin(np.pi / n_sides))
    # First corner chosen so the bottom edge (nearest the robot) is parallel to y.
    angles = np.pi + np.pi / n_sides + 2 * np.pi * np.arange(n_sides) / n_sides
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _rectangle(size_x: float, size_y: float) -> np.ndarray:
    hx, hy = size_x / 2, size_y / 2
    return np.array([[-hx, -hy], [-hx, hy], [hx, hy], [hx, -hy]])


# ID -> (name, function(size_m, height_m) -> corner points relative to the centre, in metres)
SHAPES = {
    1: ("square", lambda size, height: _rectangle(size, size)),
    2: ("rectangle", lambda size, height: _rectangle(height, size)),
    3: ("triangle", lambda size, height: _regular_polygon(3, size)),
    4: ("pentagon", lambda size, height: _regular_polygon(5, size)),
}


def shape_corners(shape: str, size_m: float, height_m: float) -> tuple[str, np.ndarray]:
    for shape_id, (name, make) in SHAPES.items():
        if shape.lower() in (name, str(shape_id)):
            return name, make(size_m, height_m)
    known = ", ".join(f"{i}={n}" for i, (n, _) in SHAPES.items())
    raise SystemExit(f"Unknown shape '{shape}'. Known shapes: {known}")


def load_design(path: str, size_m: float) -> tuple[str, list]:
    """Read a design file: {"name": ..., "strokes": [[[x, y], ...], ...]}.

    Coordinates are fractions of the design's full width (-0.5 .. 0.5), x to the
    right and y up as you look at the picture. The picture is drawn as seen
    standing behind the robot: up = away from the base, right = the robot's right.
    """
    f = Path(path)
    if not f.is_file():
        f = REPO_DIR / "designs" / (path if path.endswith(".json") else path + ".json")
    if not f.is_file():
        raise SystemExit(f"Design file '{path}' not found (looked in the designs folder too).")
    data = json.loads(f.read_text())
    strokes = []
    for stroke in data["strokes"]:
        pts = np.asarray(stroke, dtype=float) * size_m
        strokes.append(np.column_stack([pts[:, 1], -pts[:, 0]]))  # picture up -> robot x, picture right -> robot -y
    return data.get("name", f.stem), strokes


def shape_strokes(args) -> tuple[str, list]:
    """The pen strokes to draw, relative to the centre, in robot x/y metres."""
    if getattr(args, "design", None):
        return load_design(args.design, args.size / 100.0)
    name, corners = shape_corners(args.shape, args.size / 100.0, args.height / 100.0)
    return name, [np.vstack([corners, corners[:1]])]


def outline_points(strokes, n_sectors: int = 16) -> np.ndarray:
    """A few extreme points of a drawing, for a quick reach test."""
    pts = np.vstack(strokes)
    angle = np.arctan2(pts[:, 1], pts[:, 0])
    sector = ((angle + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    keep = []
    for k in range(n_sectors):
        idx = np.nonzero(sector == k)[0]
        if len(idx):
            keep.append(pts[idx[np.argmax(np.linalg.norm(pts[idx], axis=1))]])
    return np.array(keep)


# --------------------------------------------------------------------------- planning


class PlanError(Exception):
    pass


def plan_drawing(strokes_xy, centre_xy, table_z, args, limits_deg, tool):
    """Turn pen strokes into a list of joint targets, one per control tick.

    Between strokes the pen lifts, moves across at hover height and comes down
    again. Returns a dict with the Cartesian path, the joint path and a mask of
    the ticks during which the pen is on the paper.
    """
    step = args.speed / 100.0 / RATE_HZ  # metres per tick while drawing
    fast = 2.0 * step  # pen up
    draw_z = table_z - args.press / 1000.0
    hover_z = table_z + args.lift / 100.0
    dwell = max(1, int(round(0.4 * RATE_HZ)))

    def line(a, b, st):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / st)) + 1)
        return [a + (b - a) * t for t in np.linspace(0.0, 1.0, n)]

    points, pen = [], []

    def add(pts, down):
        points.extend(pts)
        pen.extend([down] * len(pts))

    for k, stroke in enumerate(strokes_xy):
        xy = np.asarray(stroke) + np.asarray(centre_xy)
        top = np.array([*xy[0], hover_z])
        if k == 0:
            add([top], False)
        else:
            add(line(points[-1], top, fast)[1:], False)  # travel with the pen up
        add(line(top, np.array([*xy[0], draw_z]), step)[1:], False)
        pen[-1] = True
        for i, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
            add(line(np.array([*a, draw_z]), np.array([*b, draw_z]), step)[1:], True)
            nxt = xy[i + 2] if i + 2 < len(xy) else (xy[1] if np.allclose(xy[0], xy[-1]) else None)
            if nxt is not None:
                u, v = b - a, nxt - b
                if np.linalg.norm(u) > 1e-9 and np.linalg.norm(v) > 1e-9:
                    turn = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
                    if turn > 40.0:
                        add([points[-1]] * dwell, True)  # let the motors settle so corners stay sharp
        add(line(points[-1], np.array([*xy[-1], hover_z]), step)[1:], False)
    points = np.array(points)
    pen = np.array(pen)

    joints = np.empty((len(points), 5))
    seed = np.array(tool["seed_deg"], dtype=float)
    seed[0] = -np.degrees(np.arctan2(points[0][1], points[0][0] - 0.0388))
    for i, p in enumerate(points):
        q, err = tool_ik(p, seed, tool)
        if err > 5e-4:
            raise PlanError(
                f"Point x={p[0] * 100:.1f} cm, y={p[1] * 100:.1f} cm, z={p[2] * 100:.1f} cm cannot be reached "
                f"(off by {err * 1000:.0f} mm). Move the shape closer to the middle of the workspace "
                "(about 20-30 cm in front of the base) or make it smaller."
            )
        joints[i] = seed = q

    for j, name in enumerate(kin.ARM_JOINTS):
        lo, hi = limits_deg[name]
        if joints[:, j].min() < lo + LIMIT_MARGIN_DEG or joints[:, j].max() > hi - LIMIT_MARGIN_DEG:
            raise PlanError(
                f"Joint '{name}' would need {joints[:, j].min():.0f} to {joints[:, j].max():.0f} deg but may only "
                f"use {lo + LIMIT_MARGIN_DEG:.0f} to {hi - LIMIT_MARGIN_DEG:.0f} deg. Move the shape, make it "
                "smaller, or hold the pen at a different angle."
            )
    down = np.nonzero(pen)[0]
    return {"points": points, "joints": joints, "pen_mask": pen, "pen": (int(down[0]), int(down[-1]) + 1)}


def quick_fit(corners_xy, centre_xy, table_z, args, limits_deg, tool) -> bool:
    """Fast check (corners only) whether the shape can be drawn around centre_xy."""
    seed = np.array(tool["seed_deg"], dtype=float)
    seed[0] = -np.degrees(np.arctan2(centre_xy[1], centre_xy[0] - 0.0388))
    lo = np.array([limits_deg[j][0] for j in kin.ARM_JOINTS]) + LIMIT_MARGIN_DEG
    hi = np.array([limits_deg[j][1] for j in kin.ARM_JOINTS]) - LIMIT_MARGIN_DEG
    for z in (table_z + args.lift / 100.0, table_z - args.press / 1000.0):
        pts = np.asarray(corners_xy) + np.asarray(centre_xy)
        pts = np.vstack([(pts + np.roll(pts, -1, axis=0)) / 2, pts])  # edge midpoints, then corners
        for corner in pts[np.argsort(np.arctan2(*(pts - centre_xy).T))]:
            q, err = tool_ik([*corner, z], seed, tool)
            if err > 5e-4 or np.any(q < lo) or np.any(q > hi):
                return False
            seed = q
    return True


def print_plan(name, plan, centre_xy, table_z):
    pts, joints = plan["points"], plan["joints"]
    print(f"\nShape: {name}, centre x={centre_xy[0] * 100:.1f} cm  y={centre_xy[1] * 100:.1f} cm, table z={table_z * 100:.1f} cm")
    print(f"Path: {len(pts)} steps, about {len(pts) / RATE_HZ:.0f} s")
    print("Joint ranges used (deg):")
    for j, joint in enumerate(kin.ARM_JOINTS):
        print(f"  {joint:14s} {joints[:, j].min():7.1f} .. {joints[:, j].max():7.1f}")


def save_plot(path: Path, name, plan, actual_xy=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib is not installed, skipping the picture: pip install matplotlib)")
        return
    pts, joints = plan["points"], plan["joints"]
    mask = plan["pen_mask"]
    drawn = np.where(mask[:, None], pts[:, :2], np.nan)  # NaN breaks the line where the pen is up
    a = plan["pen"][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # Top view as seen standing behind the robot: forward is up, the robot's left is left.
    ax1.plot(-drawn[:, 1] * 100, drawn[:, 0] * 100, color="#2a78d6", lw=2, label="planned pen line")
    if actual_xy is not None and len(actual_xy):
        ax1.plot(-actual_xy[:, 1] * 100, actual_xy[:, 0] * 100, color="#d6602a", lw=1.5, label="measured by the motors")
    ax1.plot(-pts[a, 1] * 100, pts[a, 0] * 100, "o", color="#2a78d6", label="start corner")
    ax1.plot(0, 0, "s", color="black", ms=12)
    ax1.annotate("robot base", (0, 0), textcoords="offset points", xytext=(10, -4))
    ax1.set_xlabel("to the robot's right (cm)")
    ax1.set_ylabel("forward from the base (cm)")
    ax1.set_title(f"{name}: top view")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    t = np.arange(len(joints)) / RATE_HZ
    for j, joint in enumerate(kin.ARM_JOINTS[:4]):
        ax2.plot(t, joints[:, j], label=joint)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("joint angle (deg)")
    ax2.set_title("joint angles")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Picture saved to {path}")


# --------------------------------------------------------------------------- robot helpers


def connect_robot(args):
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # older LeRobot releases
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    # The Feetech serial bus occasionally garbles a reply ("Incorrect status packet").
    # LeRobot gives up on the first one while connecting, so retry the whole connect.
    last_error = None
    for attempt in range(1, 6):
        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
        try:
            robot.connect()
            return robot
        except ConnectionError as e:
            last_error = e
            print(f"Connection attempt {attempt} failed ({str(e).splitlines()[-1][:90]}), retrying...")
            try:
                robot.bus.disconnect(False)  # close the port, leave the motors as they are
            except Exception:
                pass
            time.sleep(0.5)
    raise SystemExit(
        f"\nCould not talk to the arm on {args.port} after 5 attempts.\nLast error: {last_error}\n"
        "Check that the arm is powered, and that the 3-wire cables are pushed fully into the motor named in the error."
    )


def read_arm(robot) -> tuple[np.ndarray, float]:
    """Return (5 arm joint angles in model degrees, gripper position 0-100)."""
    for attempt in range(4):
        try:
            obs = robot.get_observation()
            break
        except ConnectionError:  # one garbled reply must not abort a drawing
            if attempt == 3:
                raise
            time.sleep(0.01)
    q = np.array([obs[f"{j}.pos"] + JOINT_OFFSETS_DEG[j] for j in kin.ARM_JOINTS], dtype=float)
    live_sample(q, obs["gripper.pos"])
    return q, float(obs["gripper.pos"])


def send_arm(robot, q_model_deg):
    LIVE["cmd"] = np.asarray(q_model_deg, dtype=float)
    robot.send_action({f"{j}.pos": float(q_model_deg[i] - JOINT_OFFSETS_DEG[j]) for i, j in enumerate(kin.ARM_JOINTS)})


def calibrated_limits(robot) -> dict[str, tuple[float, float]]:
    """Joint limits in model degrees: the tighter of the URDF and your calibration file."""
    limits = {}
    for joint in kin.ARM_JOINTS:
        lo, hi = kin.URDF_LIMITS_DEG[joint]
        cal = robot.bus.calibration.get(joint) if robot.bus.calibration else None
        if cal is not None and joint != "wrist_roll":
            half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
            off = JOINT_OFFSETS_DEG[joint]
            lo, hi = max(lo, -half + off), min(hi, half + off)
        limits[joint] = (lo, hi)
    return limits


def clearance_ok(q_from, q_to, table_z, tool) -> bool:
    """True if a straight joint-space move keeps the chalk tip above the table."""
    for t in np.linspace(0.0, 1.0, 25)[1:]:
        pos, _ = tool_fk(q_from + (q_to - q_from) * t, tool)
        if pos[2] < table_z + 0.015:
            return False
    return True


RUN_LOG = {"events": []}
LOG_FILE = REPO_DIR / "draw_log.json"

# ---- live feed for viewer.py: one JSON object per line, appended while the arm moves
LIVE_FILE = REPO_DIR / "live_run.jsonl"
LIVE = {"fh": None, "last": 0.0, "t0": 0.0, "cmd": None, "pen": False, "progress": None, "tool": None}


def live_plan(name, plan, centre, table_z):
    """Send the planned pen lines (with breaks where the pen is up) to the viewer."""
    pts, mask = plan["points"], plan["pen_mask"]
    path = []
    for i in range(0, len(pts), 2):
        path.append([round(float(pts[i, 0]), 4), round(float(pts[i, 1]), 4)] if mask[i] else None)
    live_event("plan", name=name, centre=[float(c) for c in centre], table_z=float(table_z), path=path,
               seconds=round(len(pts) / RATE_HZ, 1))


def live_start():
    try:
        LIVE["fh"] = open(LIVE_FILE, "w", encoding="utf-8")
    except OSError:
        LIVE["fh"] = None
    LIVE.update(last=0.0, t0=time.time(), cmd=None, pen=False, progress=None)


def live_event(kind, **data):
    fh = LIVE["fh"]
    if fh is None:
        return
    try:
        fh.write(json.dumps({"type": kind, "t": round(time.time() - LIVE["t0"], 3), **data}) + "\n")
        fh.flush()
    except (OSError, ValueError):
        LIVE["fh"] = None


def live_phase(text):
    live_event("phase", phase=text)


def live_sample(q, grip, force=False):
    now = time.time()
    if LIVE["fh"] is None or (not force and now - LIVE["last"] < 0.08):
        return
    LIVE["last"] = now
    tip = tool_fk(q, LIVE["tool"])[0] if LIVE["tool"] else kin.forward(q)[0]
    live_event(
        "s",
        q=[round(float(v), 2) for v in q],
        cmd=None if LIVE["cmd"] is None else [round(float(v), 2) for v in LIVE["cmd"]],
        tip=[round(float(v), 4) for v in tip],
        pen=bool(LIVE["pen"]),
        grip=round(float(grip), 1),
        progress=LIVE["progress"],
    )


# ---- remote control: the drawing app (scripts/drawing_app.py) answers prompts through stdin
REMOTE = {"on": False, "answers": None, "stop": threading.Event()}


def start_remote():
    import queue

    REMOTE["on"] = True
    REMOTE["answers"] = queue.Queue()

    def reader():
        for line in sys.stdin:
            word = line.strip().lower()
            if word == "stop":
                REMOTE["stop"].set()
            REMOTE["answers"].put(word)
        REMOTE["answers"].put("stop")  # app closed

    threading.Thread(target=reader, daemon=True).start()


def _for_app(text):
    return (text.replace("press ENTER", "press Continue").replace("Press ENTER", "Press Continue")
            .replace("Ctrl+C stops it", "STOP stops it").replace("press Ctrl+C", "press STOP"))


def ask(text, terminal_text=None):
    """Wait for the person: ENTER in the terminal, or the Continue button in the app."""
    if not REMOTE["on"]:
        return input(text if terminal_text is None else terminal_text)
    main = threading.current_thread() is threading.main_thread()
    if main:
        check_stop()
    while not REMOTE["answers"].empty():  # ignore Continue clicks made before this question
        if REMOTE["answers"].get_nowait() == "stop" and main:
            check_stop()
    print(_for_app(text), flush=True)
    live_event("prompt", text=_for_app(text).strip())
    answer = REMOTE["answers"].get()
    live_event("prompt_done")
    if answer == "stop":
        if main:  # in a helper thread the main loop's check_stop() handles it
            REMOTE["stop"].clear()
            raise KeyboardInterrupt
        return "stop"
    return ""


def check_stop():
    """Raise KeyboardInterrupt (like Ctrl+C) when the app's Stop button was pressed."""
    if REMOTE["on"] and REMOTE["stop"].is_set():
        REMOTE["stop"].clear()
        raise KeyboardInterrupt


def launch_viewer(args):
    if getattr(args, "gui", False):
        import subprocess

        subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "viewer.py"), "--file", str(LIVE_FILE)])


def log_pose(robot, label, commanded=None):
    """Record where the motors say they are (and where they were told to go) for troubleshooting."""
    q, grip = read_arm(robot)
    RUN_LOG["events"].append(
        {
            "label": label,
            "commanded_deg": None if commanded is None else [round(float(v), 2) for v in commanded],
            "measured_deg": [round(float(v), 2) for v in q],
            "gripper": round(grip, 1),
        }
    )
    return q, grip


RAW_REGISTERS = ["Torque_Enable", "Goal_Position", "Present_Position", "Present_Load", "Present_Current",
                 "Present_Voltage", "Present_Temperature", "Status", "Torque_Limit", "Max_Torque_Limit",
                 "Min_Position_Limit", "Max_Position_Limit", "Homing_Offset", "Operating_Mode"]


def log_raw(robot, label):
    """Dump raw motor registers (torque on/off, goal vs present position, load, error status)."""
    dump = {}
    for motor in list(kin.ARM_JOINTS) + ["gripper"]:
        dump[motor] = {}
        for reg in RAW_REGISTERS:
            try:
                dump[motor][reg] = int(robot.bus.read(reg, motor, normalize=False))
            except Exception as e:  # keep going: this is only for troubleshooting
                dump[motor][reg] = f"error: {e}"
    RUN_LOG.setdefault("raw", {})[label] = dump
    return dump


def save_log():
    try:
        LOG_FILE.write_text(json.dumps(RUN_LOG, indent=1))
    except Exception:
        pass


def print_check(robot, commanded, tool, table_z):
    """Compare told vs actual joint angles while hovering, to spot a motor that is not following."""
    q, grip = log_pose(robot, "hover_check", commanded)
    raw = log_raw(robot, "hover_check")
    off = [m for m in raw if raw[m].get("Torque_Enable") == 0]
    if off:
        print(f"\n  WARNING: these motors are switched OFF (no torque): {', '.join(off)}")
    print("\n  joint            told   actual   difference (deg)")
    for j, name in enumerate(kin.ARM_JOINTS):
        diff = q[j] - commanded[j]
        flag = "   <-- NOT FOLLOWING" if abs(diff) > 6.0 else ""
        print(f"  {name:14s} {commanded[j]:7.1f}  {q[j]:7.1f}  {diff:7.1f}{flag}")
    saved = tool.get("gripper_pos")
    if saved is not None:
        flag = "   <-- JAW MOVED since locate" if abs(grip - saved) > 6.0 else ""
        print(f"  gripper (jaw)  {saved:7.1f}  {grip:7.1f}  {grip - saved:7.1f}{flag}   (told = value in locate)")
    pos, direction = tool_fk(q, tool)
    off_vertical = np.degrees(np.arccos(np.clip(-direction[2], -1.0, 1.0)))
    print(
        f"  Going by the motors, the pen tip is {(pos[2] - table_z) * 100:.1f} cm above the paper "
        f"and {off_vertical:.0f} deg away from pointing straight down."
    )


class MotorNotFollowing(Exception):
    pass


def motor_fault(robot, motor) -> str:
    """Return the motor's own error report ('' if it answers normally)."""
    try:
        robot.bus.read("Status", motor, normalize=False)
        return ""
    except Exception as e:
        text = str(e)
        return text.split("]")[-1].strip() or text


def preflight(robot):
    """Refuse to move if a motor is hot, reports a fault, or sits jammed against an end stop."""
    q, _ = read_arm(robot)
    problems = []
    for j, motor in enumerate(kin.ARM_JOINTS):
        fault = motor_fault(robot, motor)
        if fault:
            problems.append(f"{motor}: the motor reports a fault ({fault})")
            continue
        try:
            temp = int(robot.bus.read("Present_Temperature", motor, normalize=False))
        except Exception:
            temp = None
        if temp is not None and temp >= 50:
            problems.append(f"{motor}: motor is hot ({temp} C; normal is 30-40). It has been straining")
        cal = robot.bus.calibration.get(motor) if robot.bus.calibration else None
        if cal is not None and motor != "wrist_roll":
            half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
            reading = q[j] - JOINT_OFFSETS_DEG[motor]
            if abs(reading) > half + 3.0:
                problems.append(
                    f"{motor}: at {reading:.0f} deg it is pressed past the end of its calibrated range (+/-{half:.0f} deg)"
                )
    if problems:
        print("\nNOT STARTING, the arm is not ready:")
        for line in problems:
            print("  - " + line)
        raise SystemExit(
            "\nUnplug the arm's power supply for 10 minutes (cools the motors and clears faults).\n"
            "Then start again from an OPEN pose: hold the arm with the pen a few cm above the paper,\n"
            "run the command, and let go once it has connected (the motors then hold it)."
        )


def move_joints(robot, q_to, speed_deg_s=35.0):
    """Smooth joint-space move from the present pose to q_to. Stops if a motor does not follow."""
    q_from, _ = read_arm(robot)
    duration = max(0.6, float(np.abs(q_to - q_from).max()) / speed_deg_s * 1.5)
    n = int(duration * RATE_HZ)
    strikes = 0
    for i in range(1, n + 1):
        blend = 0.5 - 0.5 * np.cos(np.pi * i / n)
        target = q_from + (q_to - q_from) * blend
        send_arm(robot, target)
        time.sleep(1.0 / RATE_HZ)
        check_stop()
        if i % 5 == 0:
            q_now, _ = read_arm(robot)
            lag = np.abs(q_now - target)
            strikes = strikes + 1 if lag.max() > 20.0 else 0
            if strikes >= 3:  # about half a second of a joint being far behind
                send_arm(robot, q_now)  # stop pushing
                log_pose(robot, "not_following", target)
                motor = kin.ARM_JOINTS[int(lag.argmax())]
                raise MotorNotFollowing(
                    f"'{motor}' is {lag.max():.0f} deg away from where it was told to go. "
                    f"{motor_fault(robot, motor) or 'It reports no fault, so something is probably blocking it.'}"
                )
    time.sleep(0.5)
    log_pose(robot, "after_move", q_to)


def travel(robot, q_to, table_z, tool):
    """Move to q_to, going through the ready pose if the direct way is too low."""
    q_now, _ = read_arm(robot)
    if clearance_ok(q_now, q_to, table_z, tool):
        move_joints(robot, q_to)
        return
    # Side-mounted pen: keep the gripper level so the pen keeps pointing down.
    ready = np.array([0.0, 0.0, 45.0, -45.0, 0.0]) if tool["mount"] == "side" else READY_POSE_DEG.copy()
    move_joints(robot, np.array([q_now[0], *ready[1:]]))  # raise first, then turn
    move_joints(robot, np.array([q_to[0], *ready[1:4], q_to[4]]))
    move_joints(robot, q_to)


def close_gripper(robot) -> float:
    """Close the jaws gently until they stop, then hold them there. Returns the held position (0-100)."""
    _, grip = read_arm(robot)
    goal = grip
    while goal > 0:
        goal = max(0.0, goal - 2.0)
        robot.send_action({"gripper.pos": float(goal)})
        time.sleep(0.06)
        _, grip = read_arm(robot)
        if grip - goal > 8.0:  # jaws stopped following: they are on something
            break
    time.sleep(0.6)
    _, grip = read_arm(robot)
    hold = max(0.0, grip - 6.0) if grip > 3.0 else 0.0
    robot.send_action({"gripper.pos": float(hold)})
    return grip


def grip_chalk(robot, args, tool):
    if args.keep_grip:
        return
    if tool["mount"] == "side":
        # The pen is strapped to a jaw: if the jaw moves, the pen swings with it. Closing the
        # jaws the same way in `locate` and in `draw` puts the pen at the same angle both times.
        ask("\nThe gripper will close gently and stay closed, so the pen cannot swing. Press ENTER...")
        grip = close_gripper(robot)
        print(f"Gripper closed and held (at {grip:.0f} of 100).")
        return
    _, grip = read_arm(robot)
    ask("\nThe gripper will open. Press ENTER...")
    for g in np.linspace(grip, args.grip_open, 20):
        robot.send_action({"gripper.pos": float(g)})
        time.sleep(0.03)
    ask(
        f"Hold the chalk between the jaws, pointing straight out of the gripper,\n"
        f"with about {args.stickout:.0f} cm sticking out past the fingertips. Press ENTER to close..."
    )
    grip = close_gripper(robot)
    if grip > 3.0:
        print(f"Chalk gripped (gripper at {grip:.0f} of 100).")
    else:
        print("WARNING: the gripper closed completely, it does not seem to hold anything.")


def load_surface(args):
    """Return (centre, table_z, tool). The tool saved by `locate` wins over command-line options."""
    centre, table_z, tool = None, None, None
    if SURFACE_FILE.is_file():
        data = json.loads(SURFACE_FILE.read_text())
        centre, table_z = np.array(data["centre_xy_m"]), float(data["table_z_m"])
        tool = data.get("tool")
    if tool is None:
        tool = make_tool(args)
    if args.centre is not None:
        centre = np.array(args.centre) / 100.0
    if args.table_z is not None:
        table_z = args.table_z / 100.0
    return centre, table_z, tool


# --------------------------------------------------------------------------- commands


def cmd_preview(args):
    name, strokes = shape_strokes(args)
    centre, table_z, tool = load_surface(args)
    if centre is None:
        centre = np.array([0.25, 0.0])
        print("No surface.json yet: using a default centre 25 cm in front of the base.")
    if table_z is None:
        table_z = 0.0
    print(f"Tool: {describe_tool(tool)}")
    try:
        plan = plan_drawing(strokes, centre, table_z, args, kin.URDF_LIMITS_DEG, tool)
    except PlanError as e:
        raise SystemExit(f"\nNOT POSSIBLE: {e}")
    print_plan(name, plan, centre, table_z)
    save_plot(REPO_DIR / "preview.png", name, plan)
    print("\nThe shape is reachable. Next: 'locate' (once per chalk / table setup), then 'draw'.")


def cmd_locate(args):
    live_start()
    live_phase("connecting")
    launch_viewer(args)
    robot = connect_robot(args)
    try:
        side = args.mount == "side"
        grip_chalk(robot, args, make_tool(args))
        ask(
            "\nNow HOLD THE ARM with one hand: its motors (except the gripper) will switch off\n"
            "and it would otherwise fall. Press ENTER when you are holding it..."
        )
        robot.bus.disable_torque(kin.ARM_JOINTS, num_retry=5)
        live_phase("locate: move the arm by hand")
        print(
            "\nMove the arm by hand. Check the numbers make sense: x grows when the chalk moves\n"
            "away from the base, y grows to the robot's left, z grows upwards.\n"
            "Put the pen tip on the table where the CENTRE of the shape should be.\n"
            + (
                "Hold the PEN EXACTLY UPRIGHT (vertical): the arm will keep the gripper at\n"
                "this same attitude while drawing.\n"
                if side
                else "Pen nearly upright, leaning slightly away from the base.\n"
            )
            + "When the line below says FITS HERE, press ENTER.\n"
        )
        name, strokes = shape_strokes(args)
        limits = calibrated_limits(robot)
        outline = outline_points(strokes)
        done = threading.Event()
        threading.Thread(
            target=lambda: (ask("When the line says FITS HERE, press ENTER to save this spot.", terminal_text=""), done.set()),
            daemon=True,
        ).start()
        while not done.is_set():
            q, _ = read_arm(robot)
            tool = make_tool(args, q)
            LIVE["tool"] = tool
            pos, _ = tool_fk(q, tool)
            angles = " ".join(f"{a:6.1f}" for a in q)
            fit = "FITS HERE   " if quick_fit(outline, pos[:2], pos[2], args, limits, tool) else "out of reach"
            check_stop()
            if time.time() - LIVE.get("info_t", 0.0) > 0.25:
                LIVE["info_t"] = time.time()
                live_event("info", text=f"pen tip {pos[0] * 100:.1f} cm out, {pos[1] * 100:+.1f} cm left, "
                                        f"{pos[2] * 100:.1f} cm high  |  {name}: {fit.strip()}")
            if not REMOTE["on"]:  # the app shows this line itself
                print(
                    f"\r tip x={pos[0] * 100:6.1f} y={pos[1] * 100:6.1f} z={pos[2] * 100:6.1f} cm | joints {angles} | {name}: {fit} ",
                    end="",
                    flush=True,
                )
            time.sleep(0.05)
        check_stop()  # Stop pressed instead of Continue: do not save this spot
        q, grip = read_arm(robot)
        tool = make_tool(args, q)
        tool["gripper_pos"] = round(grip, 1)
        pos, _ = tool_fk(q, tool)
        SURFACE_FILE.write_text(
            json.dumps({"centre_xy_m": [pos[0], pos[1]], "table_z_m": pos[2], "tool": tool}, indent=2)
        )
        print(f"\nTool: {describe_tool(tool)}")
        print(f"Saved to {SURFACE_FILE.name}: centre x={pos[0] * 100:.1f} cm, y={pos[1] * 100:.1f} cm, table z={pos[2] * 100:.1f} cm")

        try:
            plan_drawing(strokes, pos[:2], pos[2], args, limits, tool)
            print(f"The {args.size:.0f} cm {name} fits here.")
        except PlanError as e:
            print(f"WARNING, the {args.size:.0f} cm {name} does not fit here: {e}")
        ask("\nPut the arm back in its rest position and press ENTER to finish...")
    finally:
        robot.disconnect()


def cmd_draw(args):
    name, strokes = shape_strokes(args)
    centre, table_z, tool = load_surface(args)
    if centre is None or table_z is None:
        raise SystemExit("I don't know where the table is yet. Run 'locate' first (or pass --centre and --table-z).")
    print(f"Tool (as saved by locate): {describe_tool(tool)}")

    live_start()
    LIVE["tool"] = tool
    live_phase("connecting")
    launch_viewer(args)
    robot = connect_robot(args)
    measured = {}
    try:
        try:
            plan = plan_drawing(strokes, centre, table_z, args, calibrated_limits(robot), tool)
        except PlanError as e:
            raise SystemExit(f"\nNOT POSSIBLE: {e}")
        print_plan(name, plan, centre, table_z)
        live_plan(name, plan, centre, table_z)
        RUN_LOG.update({"tool": tool, "centre_xy_m": list(map(float, centre)), "table_z_m": float(table_z),
                        "hover_deg": plan["joints"][0].tolist(), "pen_down_deg": plan["joints"][plan["pen"][0]].tolist()})
        home, _ = log_pose(robot, "start")
        log_raw(robot, "start")
        try:
            import lerobot

            RUN_LOG["lerobot_version"] = getattr(lerobot, "__version__", "?")
            RUN_LOG["calibration"] = {m: vars(c) for m, c in (robot.bus.calibration or {}).items()}
        except Exception:
            pass
        preflight(robot)
        grip_chalk(robot, args, tool)
        log_pose(robot, "after_gripper")
        ask("\nKeep clear of the arm. Ctrl+C stops it. Press ENTER to start drawing...")

        blocked = None
        try:
            print("Moving above the first corner (the arm may first rise to a stop-over pose)...")
            live_phase("moving to the start")
            travel(robot, plan["joints"][0], table_z, tool)
            print_check(robot, plan["joints"][0], tool, table_z)
            live_phase("CHECK: waiting for ENTER")
            save_log()
            ask(
                f"\nCHECK: the pen should now point down at the paper, about {args.lift:.0f} cm above it.\n"
                "If it does, press ENTER to draw. If not, press Ctrl+C: the arm goes back to its start pose..."
            )
            mask = plan["pen_mask"]
            live_phase("drawing")
            strikes = 0
            next_tick = time.perf_counter()
            for i, q in enumerate(plan["joints"]):
                send_arm(robot, q)
                LIVE["pen"], LIVE["progress"] = bool(mask[i]), round(i / len(mask), 4)
                if mask[i] or i % 5 == 0:
                    q_now, _ = read_arm(robot)
                    if mask[i]:
                        measured[i] = tool_fk(q_now, tool)[0]
                    lag = np.abs(q_now - q)
                    strikes = strikes + 1 if lag.max() > 20.0 else 0
                    if strikes >= 15:
                        send_arm(robot, q_now)
                        log_pose(robot, "not_following", q)
                        motor = kin.ARM_JOINTS[int(lag.argmax())]
                        raise MotorNotFollowing(
                            f"'{motor}' is {lag.max():.0f} deg away from where it was told to go. "
                            f"{motor_fault(robot, motor) or 'It reports no fault, so something is probably blocking it.'}"
                        )
                next_tick += 1.0 / RATE_HZ
                time.sleep(max(0.0, next_tick - time.perf_counter()))
                check_stop()
            LIVE["pen"] = False
            print("Done drawing.")
            live_phase("done: returning to start")
        except (KeyboardInterrupt, EOFError):
            LIVE["pen"] = False
            print("\nStopped. Lifting the chalk and going back to the start pose...")
            live_phase("stopped: returning to start")
            q_now, _ = read_arm(robot)
            pos, _ = tool_fk(q_now, tool)
            q_up, err = tool_ik(pos + [0, 0, args.lift / 100.0], q_now, tool)
            if err < 2e-3:
                try:
                    move_joints(robot, q_up)
                except MotorNotFollowing as e:
                    blocked = str(e)
        except MotorNotFollowing as e:
            blocked = str(e)
        if blocked is None:
            try:
                travel(robot, home, table_z, tool)
            except MotorNotFollowing as e:
                blocked = str(e)
        if blocked is not None:
            live_phase("STOPPED: " + blocked)
            print(f"\nSTOPPED: {blocked}")
            print("The arm is holding where it stopped. Look for the pen, tape or a cable catching on the arm.")

        if measured:
            idx = np.array(sorted(measured))
            got = np.array([measured[i] for i in idx])
            err_mm = np.linalg.norm(got[:, :2] - plan["points"][idx, :2], axis=1) * 1000
            print(f"Motors followed the path within {np.percentile(err_mm, 95):.1f} mm (95% of the time).")
            trace = np.full((len(plan["points"]), 2), np.nan)
            trace[idx] = got[:, :2]
            save_plot(REPO_DIR / "last_run.png", name, plan, trace)
        ask("\nThe motors will now switch off. Support the arm if it is not resting, then press ENTER...")
    finally:
        save_log()
        live_phase("finished")
        robot.disconnect()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gui", action="store_true", help="open the live viewer window (scripts/viewer.py) alongside")
    p.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)  # used by drawing_app.py
    p.add_argument("command", choices=["preview", "locate", "draw"])
    p.add_argument("--shape", default="square", help="square, rectangle, triangle, pentagon or the ID 1-4")
    p.add_argument("--size", type=float, default=10.0,
                   help="side length in cm (rectangle: width, left-right; design: full width)")
    p.add_argument("--design", help="draw a design file instead of a shape, e.g. lotus (from the designs folder)")
    p.add_argument("--height", type=float, default=6.0, help="rectangle only: depth in cm (front-back)")
    p.add_argument("--port", default="COM6", help="serial port of the follower arm")
    p.add_argument("--id", default="follower", help="robot id used when calibrating")
    p.add_argument("--mount", choices=["inline", "side"], default="inline",
                   help="how the pen is fixed: 'inline' between the jaws, 'side' strapped across the gripper pointing down. "
                        "Used by locate, which saves it for draw")
    p.add_argument("--tilt", type=float, default=20.0, help="inline only: pen lean from vertical in deg (0 = straight down)")
    p.add_argument("--stickout", type=float, default=3.0,
                   help="cm from the fingertips to the pen tip (inline: along the pen; side: straight down to the paper)")
    p.add_argument("--speed", type=float, default=2.0, help="drawing speed in cm/s")
    p.add_argument("--press", type=float, default=2.0, help="mm to push below the table surface for chalk pressure")
    p.add_argument("--lift", type=float, default=3.0, help="cm to lift the chalk when not drawing")
    p.add_argument("--centre", type=float, nargs=2, metavar=("X", "Y"), help="shape centre in cm, overrides surface.json")
    p.add_argument("--table-z", type=float, help="table height in cm, overrides surface.json")
    p.add_argument("--grip-open", type=float, default=60.0, help="gripper opening (0-100) used to insert the chalk")
    p.add_argument("--keep-grip", action="store_true", help="do not move the gripper (chalk is already fixed)")
    return p


def main():
    args = build_parser().parse_args()
    if args.remote:
        start_remote()
    {"preview": cmd_preview, "locate": cmd_locate, "draw": cmd_draw}[args.command](args)


if __name__ == "__main__":
    main()
