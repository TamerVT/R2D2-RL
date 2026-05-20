"""Joint-controlled pick-and-place (Option 2).

Runs the *entire* task inside the joint-controlled ``LeRobotAlignGraspEnv`` --
the same env the SAC policy is trained in -- so there is no EE-vs-joint
control-mode bridge to break (see the hybrid-pipeline diagnosis).

Phases, all driven through ``env.step`` (joint targets, 5 deg/step clip):

1. reset            -> pregrasp regressor positions the arm above the cube
2. grasp            -> trained SAC policy aligns + closes the gripper
3. lift             -> move back up to the cube's pregrasp pose (gripper closed)
4. carry            -> move to ``regressor(bowl_xy)`` -- a pregrasp pose above
                       the bowl (gripper closed)
5. release          -> open the gripper; the cube drops at the bowl

Success is verified honestly: the cube's final xy must be within
``--place-tol`` of the bowl xy.

Example::

    MUJOCO_GL=egl python r2d2_rl/scripts/run_pick_place_joint.py \
        --checkpoint r2d2_rl/outputs/hil_bc_sac_20260520_0716/final_model.zip \
        --output-dir r2d2_rl/outputs/pick_place_joint --save-images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))

DEFAULT_CKPT = R2D2_RL / "outputs" / "hil_bc_sac_20260520_0716" / "final_model.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output-dir", type=Path, default=R2D2_RL / "outputs" / "pick_place_joint")
    p.add_argument("--cube-color", default="green")
    p.add_argument("--bowl-xy", type=float, nargs=2, default=[0.26, 0.10],
                   help="Bowl xy in the SO-101 shared/base frame.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--grasp-steps", type=int, default=80)
    p.add_argument("--move-steps", type=int, default=60,
                   help="Max env steps to reach each joint waypoint.")
    p.add_argument("--place-tol", type=float, default=0.06,
                   help="Cube-to-bowl xy tolerance (m) for honest success.")
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def setup_headless() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    cache = Path("/tmp") / "mesa_shader_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(cache))


def joints_deg(obs: dict) -> np.ndarray:
    """First 5 entries of observation.state are arm joints in degrees."""
    return np.asarray(obs["observation.state"][:5], dtype=np.float64)


def cube_xy(env) -> np.ndarray:
    return np.asarray(env._cube_xyz_in_shared_frame(), dtype=np.float64)[:2]


def cube_xyz(env) -> np.ndarray:
    return np.asarray(env._cube_xyz_in_shared_frame(), dtype=np.float64)


def render_external(env, path: Path, width: int = 640, height: int = 480) -> None:
    import mujoco

    sim = env.env.get_wrapper_attr("sim")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.20, 0.03, 0.05)
    cam.distance = 0.55
    cam.azimuth = 120.0
    cam.elevation = -25.0
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    renderer.update_scene(sim.data, camera=cam)
    rgb = renderer.render().copy()
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(path, rgb)


def move_to_joint_target(
    env, q_target_deg: np.ndarray, gripper_real: float, max_steps: int, tol_deg: float = 2.0
) -> tuple[dict, dict]:
    """Step the env toward an absolute joint target until reached or budget hit."""
    obs, info = None, {}
    for _ in range(max_steps):
        action = np.concatenate([q_target_deg, [gripper_real]]).astype(np.float32)
        obs, _reward, _term, _trunc, info = env.step(action)
        if np.max(np.abs(joints_deg(obs) - q_target_deg)) < tol_deg:
            break
    return obs, info


def main() -> int:
    args = parse_args()
    setup_headless()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import SAC

    from rl.lerobot_align_grasp_env import LeRobotAlignGraspEnv, LeRobotAlignGraspEnvConfig
    from rl.lerobot_compat import HIL_COLOR_TO_INDEX  # noqa: F401  (validates color set)

    env_cfg = LeRobotAlignGraspEnvConfig(
        cube_color=args.cube_color,
        use_pregrasp_regressor=True,
        max_episode_steps=400,  # long enough for grasp + carry + release
    )
    env = LeRobotAlignGraspEnv(env_cfg)
    if env._pregrasp_regressor is None:
        raise RuntimeError("Pregrasp regressor not loaded; cannot derive the bowl release pose.")

    model = SAC.load(str(args.checkpoint), device=args.device)
    print(f"[loaded] policy: {args.checkpoint}")

    obs, info = env.reset(seed=args.seed)
    q_pregrasp_cube = joints_deg(obs)        # arm above the cube (lift target)
    cube_start = cube_xyz(env)
    real_gripper_max = float(env_cfg.real_gripper_max)
    print(f"[reset]  cube xy={np.round(cube_start[:2], 3)}  pregrasp joints={np.round(q_pregrasp_cube, 1)}")

    if args.save_images:
        render_external(env, args.output_dir / "01_pregrasp.png")

    # --- Phase 2: grasp (trained SAC policy) ---------------------------------
    grasped = False
    grasp_streak = 0
    for step in range(args.grasp_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, _term, _trunc, info = env.step(action)
        terms = info.get("reward_terms", {})
        if float(terms.get("valid_cube_grasp", 0.0)) > 0.5:
            grasp_streak += 1
            if grasp_streak >= 3:
                grasped = True
                print(f"[grasp]  valid grasp held at step {step}")
                break
        else:
            grasp_streak = 0
    q_grasp = joints_deg(obs)
    print(f"[grasp]  grasped={grasped}  joints={np.round(q_grasp, 1)}")
    if args.save_images:
        render_external(env, args.output_dir / "02_grasped.png")

    # --- Phase 3: lift (back up to the cube pregrasp pose, gripper closed) ----
    obs, info = move_to_joint_target(env, q_pregrasp_cube, 0.0, args.move_steps)
    cube_after_lift = cube_xyz(env)
    print(f"[lift]   cube z {cube_start[2]:.4f} -> {cube_after_lift[2]:.4f}  "
          f"(rose {(cube_after_lift[2] - cube_start[2]) * 1000:.1f} mm)")
    if args.save_images:
        render_external(env, args.output_dir / "03_lifted.png")

    # --- Phase 4: carry (regressor pose above the bowl, gripper closed) ------
    bowl_xy = np.asarray(args.bowl_xy, dtype=np.float64)
    # Same shared-frame -> regressor-frame transform the env uses for the cube.
    bowl_regressor_in = np.array([bowl_xy[1] - 0.02, bowl_xy[0] + 0.04, 0.0], dtype=np.float32)
    q_bowl = env._pregrasp_regressor.predict(bowl_regressor_in)[:5].astype(np.float64)
    obs, info = move_to_joint_target(env, q_bowl, 0.0, args.move_steps)
    cube_after_carry = cube_xyz(env)
    print(f"[carry]  to bowl joints={np.round(q_bowl, 1)}  cube xy={np.round(cube_after_carry[:2], 3)}")
    if args.save_images:
        render_external(env, args.output_dir / "04_above_bowl.png")

    # --- Phase 5: release (open gripper) -------------------------------------
    for _ in range(15):
        action = np.concatenate([q_bowl, [real_gripper_max]]).astype(np.float32)
        obs, _r, _t, _tr, info = env.step(action)
    cube_final = cube_xyz(env)
    if args.save_images:
        render_external(env, args.output_dir / "05_released.png")

    # --- Honest success check ------------------------------------------------
    place_err = float(np.linalg.norm(cube_final[:2] - bowl_xy))
    carried = float(np.linalg.norm(cube_final[:2] - cube_start[:2]))
    success = bool(grasped and place_err <= args.place_tol)

    summary = {
        "checkpoint": str(args.checkpoint),
        "cube_start_xy": cube_start[:2].round(4).tolist(),
        "bowl_xy": bowl_xy.round(4).tolist(),
        "cube_final_xy": cube_final[:2].round(4).tolist(),
        "grasped": grasped,
        "cube_carried_distance_m": round(carried, 4),
        "place_error_m": round(place_err, 4),
        "place_tol_m": args.place_tol,
        "success": success,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=== pick-place (joint control) result ===")
    print(f"  grasped:          {grasped}")
    print(f"  cube start xy:    {cube_start[:2].round(3).tolist()}")
    print(f"  bowl xy:          {bowl_xy.round(3).tolist()}")
    print(f"  cube final xy:    {cube_final[:2].round(3).tolist()}")
    print(f"  carried distance: {carried:.3f} m")
    print(f"  place error:      {place_err:.3f} m  (tol {args.place_tol})")
    print(f"  SUCCESS:          {success}")
    if args.save_images:
        print(f"  images:           {args.output_dir}")
    env.close()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
