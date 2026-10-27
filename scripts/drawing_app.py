"""Drawing app for the SO-101: pick a pattern, then run everything with buttons.

Start it from the repo folder, in the lerobot environment:

    python scripts/drawing_app.py

Left: the controls.
  Pattern       one of the designs in the designs folder
  Size, pen     width of the drawing in cm, pen pressure in mm
  1 Locate      mark where the centre of the drawing goes (hold the arm when asked)
  2 Draw        draw the selected pattern around that spot
  Simulate      play the drawing on screen only, no robot needed
  Continue      answers the question shown above it (instead of ENTER in a terminal)
  STOP          stops the arm (like Ctrl+C): it lifts the pen and returns to its start pose

Right: the live view of the arm, the paper and the joints.

The robot side runs draw_shape.py exactly as before, in the background; this window
only presses its buttons for you.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer as vw  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent / "draw_shape.py"
DESIGNS = REPO_DIR / "designs"
LIVE_FILE = REPO_DIR / "live_run.jsonl"
DEMO_FILE = REPO_DIR / "live_demo.jsonl"

# How the pen is fixed (used by "Locate"). Change these if you re-mount the pen.
PEN_MOUNT = "side"
PEN_STICKOUT_CM = "7"

PANEL_LEFT, PANEL_W = 0.012, 0.2


def list_designs():
    found = []
    for f in sorted(DESIGNS.glob("*.json")):
        try:
            name = json.loads(f.read_text()).get("name", f.stem)
        except (OSError, json.JSONDecodeError):
            continue
        found.append((name, f.stem))
    return found or [("square", None)]


class Runner:
    """Runs draw_shape.py in the background and talks to it through its stdin."""

    def __init__(self):
        self.proc = None
        self.log = collections.deque(maxlen=200)
        self.demo_thread = None
        self.demo_stop = threading.Event()

    @property
    def busy(self):
        robot = self.proc is not None and self.proc.poll() is None
        demo = self.demo_thread is not None and self.demo_thread.is_alive()
        return robot or demo

    def start(self, args):
        if self.busy:
            return
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        cmd = [sys.executable, "-u", str(SCRIPT), *args, "--remote"]
        self.log.append("> " + " ".join(args))
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            line = line.rstrip("\n").split("\r")[-1].rstrip()
            if line.strip():
                self.log.append(line)
        code = proc.wait()
        self.log.append("-- finished --" if code == 0 else f"-- ended with an error (code {code}), see above --")

    def send(self, word):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(word + "\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    def start_demo(self, argv):
        if self.busy:
            return
        self.demo_stop.clear()
        DEMO_FILE.write_text("")
        self.log.append("> simulate " + " ".join(argv))

        def run():
            try:
                vw.run_demo(argv, DEMO_FILE, 2.0, self.demo_stop)
                self.log.append("-- simulation finished --")
            except SystemExit as e:
                self.log.append(str(e))
            except Exception as e:  # show planning errors (e.g. out of reach) in the log
                self.log.append(f"Simulation failed: {e}")

        self.demo_thread = threading.Thread(target=run, daemon=True)
        self.demo_thread.start()

    def stop(self):
        self.demo_stop.set()
        self.send("stop")

    def shutdown(self):
        self.demo_stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.send("stop")
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # the motors keep holding their last position


def build_app():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, RadioButtons, TextBox
    except ImportError:
        raise SystemExit("The app needs matplotlib:  pip install matplotlib")

    designs = list_designs()
    runner = Runner()
    state = vw.LiveState()
    tail = vw.FileTail(LIVE_FILE, state)
    fig, _ = vw.build_window(state, tail, left=0.25, figsize=(15.5, 8), animate=False)
    try:
        fig.canvas.manager.set_window_title("SO-101 drawing app")
    except Exception:
        pass

    def panel_ax(y, h, x=PANEL_LEFT, w=PANEL_W):
        return fig.add_axes([x, y, w, h])

    # panel background and headings
    bg = panel_ax(0.0, 1.0, x=0.0, w=PANEL_LEFT * 2 + PANEL_W)
    bg.set_facecolor("#f1f0ec")
    bg.set_xticks([])
    bg.set_yticks([])
    for spine in bg.spines.values():
        spine.set_visible(False)
    bg.set_zorder(-10)
    fig.text(PANEL_LEFT, 0.955, "Controls", fontsize=14, weight="bold", color=vw.INK)
    fig.text(PANEL_LEFT, 0.905, "Pattern", fontsize=10, weight="bold", color=vw.INK_2)

    # pattern choice
    ax_radio = panel_ax(0.73, 0.17)
    ax_radio.set_facecolor("#f1f0ec")
    for spine in ax_radio.spines.values():
        spine.set_visible(False)
    radio = RadioButtons(ax_radio, [n for n, _ in designs], activecolor=vw.BLUE)
    for lbl in radio.labels:
        lbl.set_fontsize(11)

    fig.text(PANEL_LEFT, 0.69, "Size (cm, full width)", fontsize=9, color=vw.INK_2)
    tb_size = TextBox(panel_ax(0.645, 0.04, w=0.08), "", initial="15")
    fig.text(PANEL_LEFT + 0.11, 0.69, "Pen pressure (mm)", fontsize=9, color=vw.INK_2)
    tb_press = TextBox(panel_ax(0.645, 0.04, x=PANEL_LEFT + 0.11, w=0.08), "", initial="1")

    def make_button(y, h, label, color, x=PANEL_LEFT, w=PANEL_W, fontsize=11):
        b = Button(panel_ax(y, h, x=x, w=w), label, color=color, hovercolor=color)
        b.label.set_fontsize(fontsize)
        b._base = color
        return b

    b_locate = make_button(0.565, 0.055, "1   Locate centre", "#ffffff")
    b_draw = make_button(0.5, 0.055, "2   Draw", "#ffffff")
    b_sim = make_button(0.435, 0.055, "Simulate (no robot)", "#ffffff", fontsize=10)

    fig.text(PANEL_LEFT, 0.39, "The arm is waiting for:", fontsize=9, color=vw.INK_2)
    prompt_txt = fig.text(PANEL_LEFT, 0.375, "", fontsize=9.5, weight="bold", color=vw.INK, va="top")
    b_cont = make_button(0.2, 0.065, "Continue  >", "#ffffff", w=0.12, fontsize=12)
    b_stop = make_button(0.2, 0.065, "STOP", vw.CRITICAL, x=PANEL_LEFT + 0.13, w=0.07, fontsize=12)
    b_stop.label.set_color("white")
    b_stop.label.set_weight("bold")

    fig.text(PANEL_LEFT, 0.17, "Messages", fontsize=9, weight="bold", color=vw.INK_2)
    log_txt = fig.text(PANEL_LEFT, 0.155, "", fontsize=7.5, color=vw.INK_2, va="top", family="monospace")

    def selected_design():
        return dict(designs)[radio.value_selected]

    def shape_args():
        key = selected_design()
        args = ["--design", key] if key else ["--shape", "square"]
        return [*args, "--size", tb_size.text.strip() or "15"]

    def use_file(path):
        if tail.path != path:
            tail.path, tail.fh, tail.pos = path, None, 0
            state.reset()

    def on_locate(_):
        if runner.busy:
            return
        use_file(LIVE_FILE)
        runner.start(["locate", *shape_args(), "--mount", PEN_MOUNT, "--stickout", PEN_STICKOUT_CM])

    def on_draw(_):
        if runner.busy:
            return
        use_file(LIVE_FILE)
        runner.start(["draw", *shape_args(), "--press", tb_press.text.strip() or "1"])

    def on_sim(_):
        if runner.busy:
            return
        use_file(DEMO_FILE)
        runner.start_demo([*shape_args(), "--press", tb_press.text.strip() or "1"])

    def on_continue(_):
        if state.prompt is not None:
            state.prompt = None  # hide it at once; the script confirms with the next message
            runner.send("ok")

    b_locate.on_clicked(on_locate)
    b_draw.on_clicked(on_draw)
    b_sim.on_clicked(on_sim)
    b_cont.on_clicked(on_continue)
    b_stop.on_clicked(lambda _: runner.stop())
    fig.canvas.mpl_connect("close_event", lambda _: runner.shutdown())

    def set_enabled(button, on):
        color = button._base if on else "#dcdbd5"
        button.color = button.hovercolor = color
        button.ax.set_facecolor(color)
        if button is not b_stop:
            button.label.set_color(vw.INK if on else "#9a998f")

    def refresh(frame):
        fig._viewer_update(frame)
        busy = runner.busy
        for b in (b_locate, b_draw, b_sim):
            set_enabled(b, not busy)
        waiting = state.prompt is not None and busy
        set_enabled(b_cont, waiting)
        b_cont.color = b_cont.hovercolor = "#cfe2fa" if waiting else "#dcdbd5"
        b_cont.ax.set_facecolor(b_cont.color)
        set_enabled(b_stop, busy)
        if waiting:
            prompt_txt.set_text("\n".join(textwrap.wrap(state.prompt, 40)[:8]))
        else:
            prompt_txt.set_text("(nothing, the arm is working)" if busy else "(idle: choose a pattern, then 1 or 2)")
        lines = []
        for line in list(runner.log)[-12:]:
            lines += textwrap.wrap(line, 46) or [""]
        log_txt.set_text("\n".join(lines[-9:]))
        return []

    anim = FuncAnimation(fig, refresh, interval=150, cache_frame_data=False)
    handlers = {"locate": on_locate, "draw": on_draw, "simulate": on_sim, "continue": on_continue,
                "stop": lambda _=None: runner.stop()}
    return {"fig": fig, "anim": anim, "runner": runner, "state": state, "refresh": refresh, "handlers": handlers,
            "radio": radio, "size": tb_size, "prompt": prompt_txt}


def main():
    app = build_app()  # noqa: F841 (the window and its animation must stay referenced)
    import matplotlib.pyplot as plt

    plt.show()
    app["runner"].shutdown()


if __name__ == "__main__":
    main()
