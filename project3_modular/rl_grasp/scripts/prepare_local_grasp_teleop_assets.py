from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]

RCS_SO101_XML = (
    REPO_ROOT
    / "third_party"
    / "robot-control-stack"
    / "assets"
    / "robots"
    / "so101"
    / "so101.xml"
)

OUT_DIR = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "assets"
    / "teleop"
)

SO101_EE_XML = OUT_DIR / "so101_ee.xml"
SCENE_XML = OUT_DIR / "scene_local_grasp.xml"
TASK_XML = OUT_DIR / "so101_local_grasp_teleop.xml"
TEMP_XML = OUT_DIR / "_temp_pose_probe.xml"


# Keep this within the actuator ctrl ranges.
# This is a grasp-local start pose, close to the RCS home/pregrasp area,
# but avoids the idle drift we got from sending out-of-range actuator targets.
START_QPOS = np.array(
    [
        -0.01914895,
        -1.74533,
        1.56454447,
        1.04777133,
        -1.40323939,
        1.40,  # gripper open
    ],
    dtype=np.float64,
)

CUBE_CENTER = np.array([0.18, 0.03, 0.01], dtype=np.float64)


def patch_robot_xml() -> None:
    if not RCS_SO101_XML.exists():
        raise FileNotFoundError(f"Missing RCS SO101 XML: {RCS_SO101_XML}")

    xml = RCS_SO101_XML.read_text()

    # Fix meshdir because we copy the robot XML into our own asset folder.
    mesh_dir = RCS_SO101_XML.parent / "assets"
    xml = xml.replace(
        'meshdir="assets"',
        f'meshdir="{mesh_dir}"',
    )

    # Ground the model visually, matching the smoke-test scene we liked.
    xml, n = re.subn(
        r'(<body name="base" pos="0 0 )-0\.03(")',
        r'\g<1>-0.06\2',
        xml,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not patch SO101 base z offset.")

    # Add an explicit HW3-style EE site at the same gripper frame
    # used by the RCS model's existing 'gripper' site.
    gripper_site_pattern = (
        r'(<site group="3" name="gripper" '
        r'pos="-0\.0079 -0\.000218121 -0\.0981274" '
        r'quat="0\.5 -0\.5 0\.5 -0\.5" />)'
    )

    replacement = (
        r'\1\n'
        r'                                    '
        r'<site group="3" name="ee_site" '
        r'pos="-0.0079 -0.000218121 -0.0981274" '
        r'quat="0.5 -0.5 0.5 -0.5" />'
    )

    xml, n = re.subn(gripper_site_pattern, replacement, xml, count=1)
    if n != 1:
        raise RuntimeError("Could not insert ee_site next to the gripper site.")

    SO101_EE_XML.write_text(xml)


def write_scene_xml() -> None:
    scene = """<mujocoinclude>
  <visual>
    <map fogstart="1.5" fogend="5" force="0.1" znear="0.05"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.22 0.22 0.22" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>

  <asset>
    <!-- Evaluation-like light gray table, approximately #B8ADA9 -->
    <material name="eval_table_gray" rgba="0.722 0.678 0.663 1"/>
  </asset>

  <worldbody>
    <light castshadow="false" directional="true"
           diffuse="0.35 0.35 0.35"
           specular="0 0 0"
           pos="-1 -1 1"
           dir="1 1 -1"/>

    <light castshadow="false" directional="true"
           diffuse="0.25 0.25 0.25"
           specular="0 0 0"
           pos="1 -1 1"
           dir="-1 1 -1"/>

    <geom name="table"
          type="plane"
          size="0 0 0.05"
          material="eval_table_gray"
          conaffinity="1"
          contype="1"
          friction="1 0.3 0.1"/>

    <body name="workspace_focus" pos="0.18 0.03 0.03">
      <site name="workspace_focus_site"
            pos="0 0 0"
            size="0.01"
            rgba="0 0 0 0"/>
    </body>

    <camera name="angle"
            pos="0.42 -0.32 0.34"
            fovy="55"
            mode="targetbody"
            target="workspace_focus"/>

    <camera name="side"
            pos="-0.02 -0.48 0.22"
            fovy="52"
            mode="targetbody"
            target="workspace_focus"/>

    <camera name="top"
            pos="0.18 0.03 0.62"
            fovy="48"
            mode="targetbody"
            target="workspace_focus"/>
  </worldbody>
</mujocoinclude>
"""
    SCENE_XML.write_text(scene)


def build_temp_probe_xml() -> None:
    qpos_str = " ".join(f"{x:.10g}" for x in START_QPOS)
    ctrl_str = qpos_str

    temp = f"""<mujoco model="so101_pose_probe">
  <include file="{SCENE_XML.name}"/>
  <include file="{SO101_EE_XML.name}"/>

  <keyframe>
    <key name="student_start"
         qpos="{qpos_str}"
         ctrl="{ctrl_str}"/>
  </keyframe>
</mujoco>
"""
    TEMP_XML.write_text(temp)


def compute_initial_ee_pose() -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(TEMP_XML))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "student_start",
    )
    if key_id == -1:
        raise RuntimeError("Temporary keyframe not found.")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ee_site",
    )
    if site_id == -1:
        raise RuntimeError("ee_site not found in temp model.")

    pos = data.site_xpos[site_id].copy()

    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])

    return pos, quat


def write_task_xml(mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> None:
    q_robot = " ".join(f"{x:.10g}" for x in START_QPOS)
    q_cube = " ".join(
        f"{x:.10g}"
        for x in [
            CUBE_CENTER[0],
            CUBE_CENTER[1],
            CUBE_CENTER[2],
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    qpos_str = f"{q_robot} {q_cube}"
    ctrl_str = q_robot

    mpos = " ".join(f"{x:.10g}" for x in mocap_pos)
    mquat = " ".join(f"{x:.10g}" for x in mocap_quat)

    task = f"""<mujoco model="so101_local_grasp_teleop">
  <include file="{SCENE_XML.name}"/>
  <include file="{SO101_EE_XML.name}"/>

  <worldbody>
    <body name="green_cube" pos="{CUBE_CENTER[0]} {CUBE_CENTER[1]} {CUBE_CENTER[2]}">
      <joint name="green_cube_joint" type="free" frictionloss="0.01"/>
      <inertial pos="0 0 0" mass="0.004" diaginertia="0.000001 0.000001 0.000001"/>
      <geom name="green_cube_geom"
            type="box"
            size="0.01 0.01 0.01"
            rgba="0 0.984 0.373 1"
            condim="4"
            solimp="2 1 0.01"
            solref="0.01 1"
            friction="1 0.005 0.0001"/>
    </body>

    <body mocap="true"
          name="mocap_target"
          pos="{mpos}"
          quat="{mquat}">
      <site name="mocap_target_site"
            pos="0 0 0"
            size="0.005"
            type="sphere"
            rgba="0 0 0 0"/>
    </body>
  </worldbody>

  <equality>
    <weld site1="mocap_target_site" site2="ee_site"/>
  </equality>

  <keyframe>
    <key name="student_start"
         qpos="{qpos_str}"
         ctrl="{ctrl_str}"/>
  </keyframe>
</mujoco>
"""
    TASK_XML.write_text(task)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patch_robot_xml()
    write_scene_xml()
    build_temp_probe_xml()

    mocap_pos, mocap_quat = compute_initial_ee_pose()
    write_task_xml(mocap_pos, mocap_quat)

    if TEMP_XML.exists():
        TEMP_XML.unlink()

    print("Generated local-grasp teleop assets:")
    print(f"  Robot: {SO101_EE_XML}")
    print(f"  Scene: {SCENE_XML}")
    print(f"  Task:  {TASK_XML}")
    print()
    print("Initial EE pose used for mocap target:")
    print("  pos: ", mocap_pos)
    print("  quat:", mocap_quat)


if __name__ == "__main__":
    main()
