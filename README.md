# so101-playground

Hobby projects based on the LeRobot SO-101 ARM



\## Setup



\- Hardware: SO-ARM101 leader and follower, Windows, no camera yet

\- Install: conda create -n lerobot python=3.12, then pip install -r requirements.txt

\- Ports on my PC: follower COM6, leader COM7 (run lerobot-find-port to find yours)



\## Troubleshooting notes



\- "Motor check failed, missing motor IDs": the motors had no unique IDs. This happens mainly because i tried registering the IDs after daisz chaining them, but it needs to be done before. That was a time-consuming but insightful learning. 

&#x20; Fixed with lerobot-setup-motors, connecting only ONE motor to the board per step.

\- Follower jumps during teleoperation: recalibrate, starting with every joint

&#x20; in the middle of its range.

