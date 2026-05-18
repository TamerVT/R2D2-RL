from __future__ import annotations

import re
from pathlib import Path


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

TELEOP_ASSET_DIR = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "assets"
    / "teleop"
)

GENERATED_ROBOT_XML = TELEOP_ASSET_DIR / "so101_grounded_for_teleop.xml"
GENERATED_SCENE_XML = TELEOP_ASSET_DIR / "so101_local_grasp_teleop_scene.xml"


def make_grounded_robot_xml() -> None:
    if not RCS_SO101_XML.exists():
        raise FileNotFoundError(f"Could not find RCS SO101 XML: {RCS_SO101_XML}")

    xml = RCS_SO101_XML.read_text()

    # The original SO101 XML uses meshdir="assets", relative to its RCS folder.
    # Since we copy the XML into our teleop asset directory, rewrite meshdir
    # to the original absolute mesh asset location.
    rcs_mesh_dir = RCS_SO101_XML.parent / "assets"
    xml = xml.replace(
        'meshdir="assets"',
        f'meshdir="{rcs_mesh_dir}"',
    )

    # RCS's raw SO101 file has:
    #   <body name="base" pos="0 0 -0.03" ...>
    # In our earlier scene tests, an additional -0.03 offset visually grounded it.
    old = r'<body name="base" pos="0 0 -0\.03"'
    new = '<body name="base" pos="0 0 -0.06"'

    xml_new, n = re.subn(old, new, xml, count=1)
    if n != 1:
        raise RuntimeError(
            "Could not patch the SO101 base z-offset. "
            "Expected to find body name='base' pos='0 0 -0.03'."
        )

    GENERATED_ROBOT_XML.write_text(xml_new)


def make_teleop_scene_xml() -> None:
    # Paths in MJCF includes are relative to this scene XML's location.
    scene_xml = """<mujoco model="so101_local_grasp_teleop">
  <include file="so101_grounded_for_teleop.xml"/>

  <option integrator="implicitfast" impratio="20" noslip_iterations="5" cone="elliptic">
    <flag multiccd="enable"/>
  </option>

  <statistic center="0.18 0.03 0.18" extent="0.55"/>

  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.18 0.18 0.18" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <!-- Evaluation-like light gray tabletop / floor: approximately #B8ADA9 -->
    <material name="eval_table_gray" rgba="0.722 0.678 0.663 1"/>
  </asset>

  <worldbody>
    <light
      pos="0.2 -0.2 1.2"
      dir="-0.1 0.1 -1"
      directional="true"
      diffuse="0.45 0.45 0.45"
      ambient="0.05 0.05 0.05"
      specular="0 0 0"
    />

    <geom
      name="floor"
      type="plane"
      size="0 0 0.05"
      material="eval_table_gray"
      friction="1 0.3 0.1"
    />

    <!-- Focus body for fixed teleop cameras -->
    <body name="workspace_focus" pos="0.18 0.03 0.03">
      <site name="workspace_focus_site" pos="0 0 0" size="0.004" rgba="0 0 0 0"/>
    </body>

    <!-- 2 cm cube -->
    <body name="green_cube_body" pos="0.18 0.03 0.01">
      <joint name="green_cube_joint" type="free" frictionloss="0.01"/>
      <inertial pos="0 0 0" mass="0.004" diaginertia="0.000001 0.000001 0.000001"/>
      <geom
        name="green_cube_geom"
        type="box"
        size="0.01 0.01 0.01"
        rgba="0 0.984 0.373 1"
        friction="1 0.3 0.1"
        solimp="2 1 0.01"
        solref="0.01 1"
        condim="4"
      />
    </body>

    <!-- Mocap target used for HW3-style Cartesian teleop -->
    <body mocap="true" name="mocap_target" pos="0 0 0">
      <site
        name="mocap_target_site"
        pos="0 0 0"
        size="0.004"
        type="sphere"
        rgba="0 0 0 0"
      />
    </body>

    <!-- Teleop cameras -->
    <camera
      name="angle"
      pos="0.43 -0.35 0.34"
      fovy="55"
      mode="targetbody"
      target="workspace_focus"
    />
    <camera
      name="side"
      pos="-0.02 -0.48 0.20"
      fovy="50"
      mode="targetbody"
      target="workspace_focus"
    />
    <camera
      name="top"
      pos="0.18 0.03 0.62"
      fovy="48"
      mode="targetbody"
      target="workspace_focus"
    />
  </worldbody>

  <!-- Weld the mocap target to the SO101 gripper site -->
  <equality>
    <weld site1="mocap_target_site" site2="gripper"/>
  </equality>
</mujoco>
"""

    GENERATED_SCENE_XML.write_text(scene_xml)


def main() -> None:
    TELEOP_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    make_grounded_robot_xml()
    make_teleop_scene_xml()

    print("Generated teleop assets:")
    print(f"  Robot XML: {GENERATED_ROBOT_XML}")
    print(f"  Scene XML: {GENERATED_SCENE_XML}")


if __name__ == "__main__":
    main()
