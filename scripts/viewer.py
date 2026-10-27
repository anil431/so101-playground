"""Live viewer for the SO-101 drawing arm.

Shows, while draw_shape.py runs:
  - the arm in 3D, moving as the motors report it
  - a top view of the paper: the planned lines (grey) and what the pen has
    actually drawn so far (blue), with the pen position
  - every joint's angle: where it was told to go vs where it actually is

Usage (from the repo folder, in the lerobot environment):

  python scripts/draw_shape.py draw --design lotus --size 15 --press 1 --gui
      opens this window automatically next to the drawing

  python scripts/viewer.py
      opens it by hand; it follows whatever draw_shape.py is doing

  python scripts/viewer.py --demo --design lotus --size 15
      no robot needed: plays the planned drawing as a simulation
      (add --speed-up 4 to watch it four times faster)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_kinematics as kin  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_DIR / "live_run.jsonl"

# Colours (reference palette): surface, ink, muted, series blue, series orange, critical red
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#c3c2b7"
GRID = "#e6e5e0"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
LAG_WARN_DEG = 6.0
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "shoulder", "elbow_flex": "elbow",
         "wrist_flex": "wrist bend", "wrist_roll": "wrist roll"}


class LiveState:
    """Everything the window needs, rebuilt from the live file."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.name = None
        self.plan_path = None  # Nx2 with NaN rows where the pen is up
        self.centre = None
        self.table_z = None
        self.total_s = None
        self.phase = "waiting for draw_shape.py ..."
        self.q = None
        self.cmd = None
        self.tip = None
        self.pen = False
        self.progress = None
        self.t = 0.0
        self.trace = []  # list of [x, y] or None (break)
        self.start_wall = None
        self.prompt = None  # question the drawing script is waiting on
        self.info = ""  # extra status line (e.g. the locate readout)

    def feed(self, msg):
        kind = msg.get("type")
        if kind == "plan":
            self.name = msg["name"]
            self.centre = msg["centre"]
            self.table_z = msg["table_z"]
            self.total_s = msg.get("seconds")
            self.plan_path = np.array([p if p is not None else [np.nan, np.nan] for p in msg["path"]], dtype=float)
            self.trace = []
        elif kind == "prompt":
            self.prompt = msg.get("text", "")
        elif kind == "prompt_done":
            self.prompt = None
        elif kind == "info":
            self.info = msg.get("text", "")
        elif kind == "phase":
            self.info = ""
            self.phase = msg["phase"]
            if msg["phase"] == "drawing":
                self.start_wall = msg.get("t")
        elif kind == "s":
            self.t = msg.get("t", self.t)
            self.q = np.array(msg["q"], dtype=float)
            self.cmd = None if msg.get("cmd") is None else np.array(msg["cmd"], dtype=float)
            self.tip = np.array(msg["tip"], dtype=float)
            self.progress = msg.get("progress")
            if msg.get("pen"):
                self.trace.append(self.tip[:2].tolist())
            elif self.pen:
                self.trace.append(None)
            self.pen = bool(msg.get("pen"))


class FileTail:
    """Reads new lines appended to the live file; starts over when a new run truncates it."""

    def __init__(self, path: Path, state: LiveState):
        self.path, self.state = path, state
        self.fh, self.pos, self.inode = None, 0, None

    def poll(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return
        ident = st.st_ino if os.name != "nt" else None  # a new run truncates the file: caught by the size check
        if self.fh is None or st.st_size < self.pos or ident != self.inode:
            if self.fh is not None:
                self.fh.close()
            self.fh = open(self.path, "r", encoding="utf-8")
            self.pos, self.inode = 0, ident
            self.state.reset()
        self.fh.seek(self.pos)
        while True:
            line = self.fh.readline()
            if not line or not line.endswith("\n"):
                break
            self.pos = self.fh.tell()
            try:
                self.state.feed(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass


# --------------------------------------------------------------------------- demo (no robot)


def run_demo(argv, out_file: Path, speed_up: float, stop_event=None):
    """Plan a drawing with draw_shape.py and play it into out_file as if the arm were moving."""
    import draw_shape as ds

    args = ds.build_parser().parse_args(["preview", *argv])
    name, strokes = ds.shape_strokes(args)
    centre, table_z, tool = ds.load_surface(args)
    if centre is None:
        centre = np.array([0.36, 0.0])
    if table_z is None:
        table_z = -0.05
    plan = ds.plan_drawing(strokes, centre, table_z, args, kin.URDF_LIMITS_DEG, tool)
    ds.LIVE_FILE = out_file
    ds.live_start()
    ds.LIVE["tool"] = tool
    ds.live_plan(name, plan, centre, table_z)
    ds.live_phase("drawing (simulation)")
    mask, joints = plan["pen_mask"], plan["joints"]
    lagged = joints[0].copy()
    t_next = time.perf_counter()
    for i, q in enumerate(joints):
        if stop_event is not None and stop_event.is_set():
            break
        lagged += (q - lagged) * 0.5  # motors trail the command a little, as on the real arm
        ds.LIVE["cmd"], ds.LIVE["pen"], ds.LIVE["progress"] = q, bool(mask[i]), round(i / len(mask), 4)
        ds.live_sample(lagged, 0.0, force=(i % 3 == 0))
        t_next += 1.0 / ds.RATE_HZ / speed_up
        time.sleep(max(0.0, t_next - time.perf_counter()))
    ds.LIVE["pen"] = False
    ds.live_phase("finished (simulation)")


# --------------------------------------------------------------------------- window


def build_window(state: LiveState, tail: FileTail, left: float = 0.02, figsize=(13, 7.2), animate: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    try:
        fig.canvas.manager.set_window_title("SO-101 live viewer")
    except Exception:
        pass
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.25, 1], left=left, right=0.97,
                          top=0.88, bottom=0.08, wspace=0.12, hspace=0.32)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_j = fig.add_subplot(gs[1, 1])
    title = fig.text(left, 0.955, "", fontsize=14, weight="bold", color=INK)
    status = fig.text(left, 0.918, "", fontsize=10, color=INK_2)

    # ---- 3D arm
    ax3.set_facecolor(SURFACE)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((0.99, 0.99, 0.98, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax3.set_xlim(-0.05, 0.5)
    ax3.set_ylim(-0.25, 0.25)
    ax3.set_zlim(-0.1, 0.35)
    ax3.set_box_aspect((0.55, 0.5, 0.45))
    ax3.view_init(elev=24, azim=-128)
    ax3.set_xlabel("forward (m)")
    ax3.set_ylabel("left (m)")
    ax3.set_zlabel("up (m)")
    fig.text(left + 0.02, 0.035, "Arm as the motors report it.  Orange = pen on the paper,  x = where it was told to be.",
             fontsize=8.5, color=INK_2)
    (arm_line,) = ax3.plot([], [], [], "-o", color=INK_2, lw=4, ms=6, mfc=SURFACE, mec=INK_2, mew=2)
    (pen_line,) = ax3.plot([], [], [], "-", color=ORANGE, lw=3)
    (plan3,) = ax3.plot([], [], [], "-", color=MUTED, lw=1)
    (trace3,) = ax3.plot([], [], [], "-", color=BLUE, lw=2)
    (cmd_tip3,) = ax3.plot([], [], [], "x", color=INK, ms=7)
    base_sq = np.array([[-0.04, -0.05], [0.06, -0.05], [0.06, 0.05], [-0.04, 0.05], [-0.04, -0.05]])
    ax3.plot(base_sq[:, 0], base_sq[:, 1], np.zeros(5), color=INK_2, lw=1.5)
    paper3 = ax3.plot([], [], [], color=GRID, lw=1)[0]

    # ---- top view of the paper (as seen standing behind the robot)
    ax_top.set_facecolor(SURFACE)
    ax_top.set_aspect("equal")
    ax_top.grid(color=GRID, lw=0.6)
    ax_top.set_xlabel("to the robot's right (cm)")
    ax_top.set_ylabel("forward from the base (cm)")
    ax_top.set_title("paper, top view", color=INK_2, fontsize=10)
    ax_top.set_xlim(-25, 25)
    ax_top.set_ylim(0, 50)
    (plan2,) = ax_top.plot([], [], "-", color=MUTED, lw=1.2, label="planned")
    (trace2,) = ax_top.plot([], [], "-", color=BLUE, lw=2, label="drawn (measured)")
    (pen2,) = ax_top.plot([], [], "o", ms=10, mew=2, mfc="none", mec=INK_2, label="pen (filled = down)")
    ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)

    # ---- joints: actual (bar) vs told (black tick)
    names = kin.ARM_JOINTS
    ypos = np.arange(len(names))[::-1]
    ax_j.set_facecolor(SURFACE)
    bars = ax_j.barh(ypos, np.zeros(len(names)), height=0.5, color=BLUE)
    (told,) = ax_j.plot([], [], "|", color=INK, ms=18, mew=2.5, label="told")
    ax_j.axvline(0, color=INK_2, lw=0.8)
    ax_j.set_yticks(ypos, [SHORT[n] for n in names])
    ax_j.set_xlim(-120, 120)
    ax_j.set_xlabel("joint angle (deg); bar = actual, black tick = told")
    ax_j.grid(axis="x", color=GRID, lw=0.6)
    ax_j.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_j.spines[spine].set_visible(False)
    vals = [ax_j.text(118, y, "", ha="right", va="center", fontsize=8, color=INK_2) for y in ypos]

    view = {"plan_drawn": None}

    def update(_frame):
        tail.poll()
        s = state
        title.set_text(f"Drawing: {s.name}" if s.name else "SO-101 live viewer")
        bits = [s.phase]
        if s.progress is not None:
            bits.append(f"{s.progress * 100:.0f}% done")
            if s.total_s:
                bits.append(f"about {max(0, s.total_s * (1 - s.progress)):.0f} s left")
        bits.append("pen DOWN" if s.pen else "pen up")
        if s.info:
            bits = [s.info]
        status.set_text("   |   ".join(bits))

        # plan (only redrawn when a new plan arrives)
        if s.plan_path is not None and view["plan_drawn"] is not s.plan_path:
            p = s.plan_path
            plan2.set_data(-p[:, 1] * 100, p[:, 0] * 100)
            plan3.set_data_3d(p[:, 0], p[:, 1], np.full(len(p), s.table_z))
            ok = p[~np.isnan(p[:, 0])]
            lo, hi = ok.min(0), ok.max(0)
            pad = 0.02
            ax_top.set_xlim(-(hi[1] + pad) * 100, -(lo[1] - pad) * 100)
            ax_top.set_ylim((lo[0] - pad) * 100, (hi[0] + pad) * 100)
            px = [lo[0] - pad, hi[0] + pad, hi[0] + pad, lo[0] - pad, lo[0] - pad]
            py = [lo[1] - pad, lo[1] - pad, hi[1] + pad, hi[1] + pad, lo[1] - pad]
            paper3.set_data_3d(px, py, [s.table_z] * 5)
            ax3.set_zlim(min(-0.1, s.table_z - 0.02), 0.35)
            view["plan_drawn"] = s.plan_path
        elif s.plan_path is None and view["plan_drawn"] is not None:
            plan2.set_data([], [])
            plan3.set_data_3d([], [], [])
            view["plan_drawn"] = None

        if s.trace:
            tr = np.array([p if p is not None else [np.nan, np.nan] for p in s.trace], dtype=float)
            trace2.set_data(-tr[:, 1] * 100, tr[:, 0] * 100)
            z = s.table_z if s.table_z is not None else 0.0
            trace3.set_data_3d(tr[:, 0], tr[:, 1], np.full(len(tr), z))
        else:
            trace2.set_data([], [])
            trace3.set_data_3d([], [], [])

        if s.q is not None:
            pts = kin.link_points(s.q)
            arm_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            finger = pts[-1]
            pen_line.set_data_3d([finger[0], s.tip[0]], [finger[1], s.tip[1]], [finger[2], s.tip[2]])
            pen_line.set_color(ORANGE if s.pen else INK_2)
            pen2.set_data([-s.tip[1] * 100], [s.tip[0] * 100])
            pen2.set_markerfacecolor(ORANGE if s.pen else "none")
            pen2.set_markeredgecolor(ORANGE if s.pen else INK_2)
            if s.cmd is not None:
                ct = kin.link_points(s.cmd)[-1]
                cmd_tip3.set_data_3d([ct[0]], [ct[1]], [ct[2]])
            for k, (bar, y) in enumerate(zip(bars, ypos)):
                bar.set_width(s.q[k])
                lag = None if s.cmd is None else s.q[k] - s.cmd[k]
                bad = lag is not None and abs(lag) > LAG_WARN_DEG
                bar.set_color(CRITICAL if bad else BLUE)
                text = f"{s.q[k]:6.1f}"
                if lag is not None:
                    text += f"  (off {lag:+.1f})" + ("  ! not following" if bad else "")
                vals[k].set_text(text)
                vals[k].set_color(CRITICAL if bad else INK_2)
            if s.cmd is not None:
                told.set_data(s.cmd, ypos)
        return []

    anim = FuncAnimation(fig, update, interval=120, cache_frame_data=False) if animate else None
    fig._viewer_update = update  # lets drawing_app.py chain its own refresh
    return fig, anim


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=str(DEFAULT_FILE), help="live file written by draw_shape.py")
    p.add_argument("--demo", action="store_true", help="simulate the drawing (no robot needed)")
    p.add_argument("--speed-up", type=float, default=1.0, help="demo only: play faster")
    args, rest = p.parse_known_args()

    path = Path(args.file)
    if args.demo:
        path = REPO_DIR / "live_demo.jsonl"
        path.write_text("")
        threading.Thread(target=run_demo, args=(rest, path, args.speed_up), daemon=True).start()
    elif rest:
        p.error(f"unknown options: {' '.join(rest)} (shape options only work with --demo)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "The viewer needs matplotlib, which is not installed in this environment.\n"
            "Install it once with:   pip install matplotlib\n"
            "(run that in the same window, with the lerobot environment active)"
        )

    state = LiveState()
    fig, anim = build_window(state, FileTail(path, state))
    plt.show()


if __name__ == "__main__":
    main()
