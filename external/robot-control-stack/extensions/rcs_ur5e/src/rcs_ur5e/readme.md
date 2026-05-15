# UR5e Extension for Robot Control Stack 🤖
This repository contains a basic Python-based controller plugin for the UR5 robot. It allows for direct manipulation of the UR5.

## Warnings & Safety Notices
- Non-Compliant Operation: This controller does not implement any force-based control (e.g., force mode or impedance control). The robot will not be compliant to external forces. The robot will go into Emergency-Stop when it detects a collision.

- Rough Control: This is a basic joint-level or Cartesian controller implementation. Movements may be slightly rough or non-optimal compared to highly tuned industrial controllers. 

- Home Position Verification: Always check and verify the robot's starting (home) position configured in the plugin or the control stack before starting any program. An incorrect home position could lead to collisions.

- Gradual Velocity Increase: Start all operations using a very slow velocity setting on the pendant or control panel(i.e., 10%). Gradually increase the velocity only after confirming that the basic movements are safe, stable, and correct. Never immediately jump to high speeds.

## Installation
To install hardware extension use
```shell
pip install -ve rcs_ur5e
```

## Usage
You can initially test the connection to the robot and basic movements by running the scripts in the [scripts](./scripts/) folder.

For code examples see the [examples](../../../../examples/ur5e) folder.

You can switch between hardware and simulation by setting the following flag:
```python
ROBOT_INSTANCE = RobotPlatform.HARDWARE
# ROBOT_INSTANCE = RobotPlatform.SIMULATION
```
