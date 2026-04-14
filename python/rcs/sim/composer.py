import os
from typing import List, Optional, Union

import mujoco


class ModelComposer:
    """
    Composes MuJoCo scenes using mjSpec with flexible positioning and prefixing.
    """

    def __init__(self, model_name: str = "rcs_scene", attachment_site_name: str = "attachment_site"):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = model_name
        self.spec.compiler.autolimits = True
        self.attachment_site_name = attachment_site_name


    def _resolve_asset_paths(self, spec: mujoco.MjSpec, xml_path: str):
        """Resolves relative paths to absolute ones."""
        if spec.compiler.autolimits:
            self.spec.compiler.autolimits = True

        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        mesh_base = os.path.join(xml_dir, spec.meshdir or "")
        tex_base = os.path.join(xml_dir, spec.texturedir or "")

        for mesh in spec.meshes:
            if mesh.file and not os.path.isabs(mesh.file):
                mesh.file = os.path.abspath(os.path.join(mesh_base, mesh.file))
        for tex in spec.textures:
            if tex.file and not os.path.isabs(tex.file):
                tex.file = os.path.abspath(os.path.join(tex_base, tex.file))

    def load_base_scene(self, xml_path: str):
        """Loads the base world XML."""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Base scene XML not found: {xml_path}")
        self.spec = mujoco.MjSpec.from_file(xml_path)
        self._resolve_asset_paths(self.spec, xml_path)

    def _find_site(self, name: str) -> Optional[mujoco._specs.MjsSite]:
        for s in self.spec.sites:
            if s.name == name:
                return s
        return None

    def _find_body(self, name: str) -> Optional[mujoco._specs.MjsBody]:
        try:
            return self.spec.find_body(name)
        except:
            return None

    def add_robot(
        self, xml_path: str, prefix: str, pos: Union[list, tuple] = (0, 0, 0), quat: Union[list, tuple] = (0, 0, 0, 1)
    ) -> mujoco._specs.MjsBody:
        """
        Attaches a robot MJCF at a specific pose.
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Robot MJCF not found: {xml_path}")

        child_spec = mujoco.MjSpec.from_file(xml_path)
        self._resolve_asset_paths(child_spec, xml_path)

        # 1. Use a frame to attach the child spec (most comprehensive)
        frame = self.spec.worldbody.add_frame()
        frame.attach(child_spec, prefix, "")

        # 2. Identify the robot root body (it was prefixed)
        child_root_name = child_spec.worldbody.first_body().name
        prefixed_root_name = f"{prefix}{child_root_name}"

        robot_root = self._find_body(prefixed_root_name)
        if not robot_root:
            raise ValueError(f"Could not find robot root body '{prefixed_root_name}' after attachment.")

        # 3. Apply the pose directly to the body
        robot_root.pos = pos
        # change quaternion rotation order to mujoco format
        quat_list = list(quat)
        robot_root.quat = quat_list[3:4] + quat_list[0:3]

        return robot_root

    def add_gripper(self, xml_path: str, robot_prefix: str, gripper_prefix: str = "gripper_") -> mujoco._specs.MjsBody:
        """Attaches a gripper to a robot's 'attachment_site'."""
        site_name = robot_prefix + self.attachment_site_name
        attachment_site = self._find_site(site_name)

        if not attachment_site:
            raise ValueError(f"Attachment site '{site_name}' not found.")

        gripper_spec = mujoco.MjSpec.from_file(xml_path)
        self._resolve_asset_paths(gripper_spec, xml_path)

        gripper_root = gripper_spec.worldbody.first_body()
        return attachment_site.attach(gripper_root, gripper_prefix, "")

    def add_object_from_xml(
        self, xml_path: str, prefix: str, pos: Union[list, tuple] = (0, 0, 0), quat: Union[list, tuple] = (0, 0, 0, 1)
    ) -> mujoco._specs.MjsBody:
        """
        Attaches a single object MJCF at a specific pose.
        Assumes the XML contains only one root body in the worldbody.
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Object MJCF not found: {xml_path}")

        # Load the child spec
        child_spec = mujoco.MjSpec.from_file(xml_path)
        self._resolve_asset_paths(child_spec, xml_path)

        # Attach using a frame
        frame = self.spec.worldbody.add_frame()
        frame.attach(child_spec, prefix, "")

        # Identify the root of the added object
        child_root_name = child_spec.worldbody.first_body().name
        prefixed_root_name = f"{prefix}{child_root_name}"

        obj_root = self._find_body(prefixed_root_name)
        if not obj_root:
            raise ValueError(f"Could not find object root body '{prefixed_root_name}' after attachment.")

        # Apply the pose
        obj_root.pos = pos
        # change quaternion rotation order to mujoco format
        quat_list = list(quat)
        obj_root.quat = quat_list[3:4] + quat_list[0:3]

        return obj_root

    def save_mjcf(self, output_path: str):
        """Compiles and saves the MJCF."""
        self.spec.compile()
        xml_str = self.spec.to_xml()
        with open(output_path, "w") as f:
            f.write(xml_str)
        return xml_str

    def get_model(self):
        return self.spec.compile()
