from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional
import numpy as np
import mujoco


@dataclass
class Workspace2D:
    # workspace in ROBOT BASE frame (RCS convention: x forward, y left, z up) [2](https://robotcontrolstack.org/user_guide/conventions.html)
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    yaw_range: tuple[float, float] = (-np.pi, np.pi)


def _wxyz_from_yaw(yaw: float) -> np.ndarray:
    # wxyz quaternion for yaw about +Z
    return np.array([np.cos(yaw/2.0), 0.0, 0.0, np.sin(yaw/2.0)], dtype=float)


def _get_freejoint_qadr(m: mujoco.MjModel, body_id: int) -> int:
    jadr = int(m.body_jntadr[body_id])
    if jadr < 0 or int(m.jnt_type[jadr]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Body must have a FREE joint as its first joint to be randomized.")
    return int(m.jnt_qposadr[jadr])


def _body_name(m: mujoco.MjModel, bid: int) -> str:
    return m.body(bid).name


def _geom_body_name(m: mujoco.MjModel, gid: int) -> str:
    return _body_name(m, int(m.geom_bodyid[gid]))


def _cube_halfheight_from_model(m: mujoco.MjModel, body_id: int) -> float:
    # choose first BOX geom on body; size[2] is half-height for boxes [5](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)[6](https://stackoverflow.com/questions/73296464/mujoco-what-is-the-difference-between-pos-qpos-and-xpos)
    for gid in range(m.ngeom):
        if int(m.geom_bodyid[gid]) == body_id and int(m.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_BOX:
            return float(m.geom_size[gid][2])
    # fallback: use bounding sphere approx from any geom size
    # (still works, but box is preferred)
    raise ValueError("Expected a box geom on cube body to infer half-height.")


def _cube_radius_xy(m: mujoco.MjModel, body_id: int, margin: float) -> float:
    # conservative 2D radius for a box: sqrt(hx^2 + hy^2) + margin
    for gid in range(m.ngeom):
        if int(m.geom_bodyid[gid]) == body_id and int(m.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_BOX:
            hx, hy = float(m.geom_size[gid][0]), float(m.geom_size[gid][1])
            return float(np.sqrt(hx*hx + hy*hy) + margin)
    return 0.03 + margin  # fallback


def randomize_cube_positions(
    sim,
    cube_body_names: Sequence[str],
    workspace: Workspace2D,
    *,
    surface_z: float = 0.0,
    base_body_name: str = "robotbase",
    robot_body_prefix: str = "robot",
    min_sep_margin: float = 0.005,
    forbid_env_contacts: bool = False,
    max_tries_per_cube: int = 200,
    seed: Optional[int] = None,
    debug: bool = False,
) -> None:
    """
    Randomize cube free bodies:
      - within workspace (robot base frame)
      - non-overlapping with each other
      - not colliding with the robot
      - (optionally) not colliding with environment
    """

    m = sim.model
    d = sim.data
    rng = np.random.default_rng(seed)

    # --- base transform: robot base -> world
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_id < 0:
        raise ValueError(f"Base body '{base_body_name}' not found. Use your robot root body name.")
    mujoco.mj_forward(m, d)
    p_base = d.xpos[base_id].copy()
    R_base = d.xmat[base_id].reshape(3, 3).copy()  # body->world rotation

    placed_xy_world: list[np.ndarray] = []
    placed_r: list[float] = []
    cube_ids: list[int] = []

    # pre-resolve cube ids and radii
    for nm in cube_body_names:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            raise ValueError(f"Cube body not found: {nm}")
        cube_ids.append(bid)

    cube_radii = [ _cube_radius_xy(m, bid, min_sep_margin) for bid in cube_ids ]

    # helper: check contact constraints after placing
    def contacts_ok(current_cube_name: str) -> bool:
        # run collision detection for current qpos
        mujoco.mj_forward(m, d)
        mujoco.mj_collision(m, d)  # fills d.contact / d.ncon [3](https://deepwiki.com/google-deepmind/mujoco/5.1-c-api-reference)[4](https://deepwiki.com/google-deepmind/mujoco/4.6-collision-detection)

        # check all contacts
        for ci in range(d.ncon):
            con = d.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            b1 = _geom_body_name(m, g1)
            b2 = _geom_body_name(m, g2)

            involved = {b1, b2}
            if current_cube_name not in involved:
                continue

            # collide with robot?
            if any(name.startswith(robot_body_prefix) for name in involved if name != current_cube_name):
                return False

            # optionally forbid cube-env contacts too (floor/table)
            if forbid_env_contacts:
                # env = anything that is not robot and not one of the cubes
                for name in involved:
                    if name == current_cube_name:
                        continue
                    if name.startswith(robot_body_prefix):
                        continue
                    if name in cube_body_names:
                        continue
                    return False

            # cube-cube contacts are forbidden
            if any((name in cube_body_names) for name in involved if name != current_cube_name):
                return False

        return True

    # --- place each cube with rejection sampling
    for bid, r, name in zip(cube_ids, cube_radii, cube_body_names):
        qadr = _get_freejoint_qadr(m, bid)

        half_h = _cube_halfheight_from_model(m, bid)
        z_world = surface_z + half_h

        success = False
        for attempt in range(max_tries_per_cube):

            # sample in robot-base frame (reachable domain) [2](https://robotcontrolstack.org/user_guide/conventions.html)
            x_b = rng.uniform(*workspace.x_range)
            y_b = rng.uniform(*workspace.y_range)
            yaw = rng.uniform(*workspace.yaw_range)

            # transform to world
            p_b = np.array([x_b, y_b, z_world], dtype=float)
            p_w = p_base + R_base @ p_b
            quat_wxyz = _wxyz_from_yaw(yaw)

            # fast non-overlap check in XY (world)
            xy = p_w[:2]
            ok_sep = True
            for (xy2, r2) in zip(placed_xy_world, placed_r):
                if np.linalg.norm(xy - xy2) < (r + r2):
                    ok_sep = False
                    break
            if not ok_sep:
                continue

            # set freejoint qpos: [x,y,z, qw,qx,qy,qz] [1](https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html)[2](https://robotcontrolstack.org/user_guide/conventions.html)
            d.qpos[qadr:qadr+3] = p_w
            d.qpos[qadr+3:qadr+7] = quat_wxyz

            # check collisions with robot and other cubes using contact list
            if not contacts_ok(name):
                continue

            placed_xy_world.append(xy.copy())
            placed_r.append(r)
            success = True
            if debug:
                print(f"[randomize_cubes] placed {name} in {attempt+1} tries at {p_w}")
            break

        if not success:
            raise RuntimeError(
                f"Failed to place cube '{name}' without collisions after {max_tries_per_cube} tries. "
                f"Try enlarging workspace or reducing cube count / size."
            )

    # finalize derived quantities
    mujoco.mj_forward(m, d)