# so101-playground

Hobby projects based on the LeRobot SO-101 arm: teleoperation, and a follower arm that draws
shapes and rangoli/kolam designs on paper by itself, with no leader arm.

## Setup

- Hardware: SO-ARM101 leader and follower, Windows, no camera yet
- Install: `conda create -n lerobot python=3.12`, then `pip install -r requirements.txt`
  (LeRobot with the Feetech motor drivers, plus matplotlib for the viewer and drawing app)
- Ports on my PC: follower COM6, leader COM7 (run `lerobot-find-port` to find yours)

## Scripts

| File | What it does |
|---|---|
| `scripts/calibrate_follower.bat`, `scripts/calibrate_leader.bat` | LeRobot calibration of each arm |
| `scripts/teleop.bat` | Teleoperation: the follower copies the leader |
| `scripts/drawing_app.py` (`drawing_app.bat`) | **Drawing app**: pick a pattern and run everything with buttons |
| `scripts/draw_shape.py` | The drawing engine (command line): `preview`, `locate`, `draw` |
| `scripts/viewer.py` | Live view of the arm, the pen trace and the joints; `--demo` simulates a drawing |
| `scripts/so101_kinematics.py` | Forward/inverse kinematics from the official SO-101 URDF (numpy only) |
| `scripts/draw_square.bat` | Shortcut: draw a 10 cm square |
| `designs/*.json` | Drawable designs: `lotus`, `star_rangoli`, `kolam_star` |

## Drawing with the follower arm

The follower holds a pen and draws on paper lying flat in front of it. The script computes the
joint angles itself (inverse kinematics), so the leader stays unplugged.

My pen setup: the pen is strapped across the gripper, pointing straight down, its tip about 7 cm
below the gripper (`--mount side --stickout 7`). The base sits on a board about 5 cm above the paper.
A chalk or pen held between the jaws, pointing out of the gripper, also works (`--mount inline`).

### Using the drawing app

From the repo folder, in the lerobot environment: `python scripts/drawing_app.py` (or run `scripts\drawing_app.bat`).

1. Choose a **pattern** and its **size** (full width in cm; 15 cm works for all three designs).
2. **1 Locate centre**: the gripper closes on the pen, then the arm goes limp. Hold it, put the pen tip
   upright on the paper where the centre of the drawing should be, and press **Continue** when the top
   line says `FITS HERE`. The spot is saved in `surface.json`; only redo this when the pen, paper or
   arm position changes.
3. **2 Draw**: hold the arm with the pen a few cm above the paper and press Draw. The arm moves above the
   first point and pauses: check the pen points down at the paper, then press **Continue**.
4. **STOP** at any time: the arm lifts the pen and returns to its start pose.

Questions the arm would ask in the terminal appear in the panel and are answered with **Continue**.
**Simulate** plays the drawing on screen without the robot. The pen mounting used by Locate is set at
the top of `drawing_app.py`.

### Using the command line

```
python scripts/draw_shape.py preview --design lotus --size 15            # no robot: is it reachable? saves preview.png
python scripts/draw_shape.py locate --design lotus --size 15 --mount side --stickout 7
python scripts/draw_shape.py draw --design lotus --size 15 --press 1 --gui
```

- `--design NAME` draws a file from `designs/`; without it, `--shape square|rectangle|triangle|pentagon` (or ID 1-4).
- `--size` cm (full width), `--press` mm of pen pressure, `--speed` cm/s (default 2), `--port` (default COM6).
- `--gui` opens the live viewer next to it. Ctrl+C stops the arm, lifts the pen and returns to the start pose.
- After a drawing, `last_run.png` compares the planned lines with what the motors measured, and
  `draw_log.json` records joint angles and motor status for troubleshooting.

### Safety checks built in

- Before moving, the arm refuses to start if a motor is hot (50 C or more), reports a fault, or is pressed
  past the end of its calibrated range, and says which one.
- While moving, it stops if a joint falls more than 20 degrees behind its target (something blocking it).
- Garbled replies on the serial bus are retried instead of crashing.

### Reach

With the pen strapped across the gripper, a 15 cm design fits with its centre about 34-38 cm in front of
the base. `preview`, `locate` and the app tell you when a shape does not fit.

### Adding a design

A design is a JSON file in `designs/`: `{"name": "...", "strokes": [[[x, y], ...], ...]}`. Coordinates are
fractions of the full width (-0.5 to 0.5), x to the right and y up as seen standing behind the robot
(up = away from the base). Each stroke is drawn with the pen down; the pen lifts between strokes.

## Troubleshooting notes

- **"Motor check failed, missing motor IDs"**: the motors had no unique IDs. This happened because I tried
  registering the IDs after daisy-chaining them, but it needs to be done before. That was a time-consuming
  but insightful learning. Fixed with `lerobot-setup-motors`, connecting only ONE motor to the board per step.
- **Bad calibration (follower jumps, impossible coordinates)**: in the calibration file, any joint except
  wrist_roll with `range_min` near 0 and `range_max` near 4095 was swung through a full turn during calibration.
  Recalibrate: start with every joint in the middle of its range, then move one joint at a time to both end stops.
  When asked whether to use the existing calibration file, type `c` (plain ENTER keeps the old one).
- **Wrist motors stop following / overheat**: a pen or cable catching on the wrist, or starting from a folded
  pose with the wrist pressed against its stop. Unplug the power for 10 minutes and start from an open pose.
- **"Failed to write 'Lock' ... Incorrect status packet"**: a garbled serial reply; the scripts retry. If it
  keeps happening, check the 3-wire cables at the motor named in the message.
- **`No module named 'matplotlib'`**: `pip install matplotlib` in the lerobot environment.

## To do

- The leader arm still has its first calibration (full-turn ranges on shoulder lift, elbow and wrist flex);
  recalibrate it before teleoperating again. For a gentle first run, add `--robot.max_relative_target=10`
  to the teleop command.
