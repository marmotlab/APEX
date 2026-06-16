import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch, gymutil
import torch

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict, parse_sim_params


ROBOT_TO_TASK = {
    "go2": "go2_flat",
    "go2_flat": "go2_flat",
    "g1": "g1",
    "h1": "h1",
    "h1_2": "h1_2",
}


@dataclass(frozen=True)
class ReplaySchema:
    dof_start: int
    dof_count: Optional[int] = None
    height_col: Optional[int] = None
    quat_start: Optional[int] = None
    base_xy_start: Optional[int] = None
    lin_vel_start: Optional[int] = None
    ang_vel_start: Optional[int] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay an imitation CSV directly on a robot model without loading a policy."
    )
    parser.add_argument("--robot", required=True, help="Robot/task type, e.g. h1, h1_2, g1, go2.")
    parser.add_argument("--imitation-data", required=True, help="Path to the imitation CSV.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--physics_engine", type=str, default="physx", choices=["physx", "flex"])
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    args = parser.parse_args()

    args.task = ROBOT_TO_TASK.get(args.robot, args.robot)
    args.resume = False
    args.experiment_name = None
    args.run_name = None
    args.load_run = None
    args.checkpoint = None
    args.seed = None
    args.max_iterations = None

    sim_device_type, compute_device_id = gymutil.parse_device_str(args.sim_device)
    args.sim_device_type = sim_device_type
    args.compute_device_id = compute_device_id
    args.sim_device_id = compute_device_id
    args.device = args.sim_device
    args.physics_engine = gymapi.SIM_PHYSX if args.physics_engine == "physx" else gymapi.SIM_FLEX
    return args


def is_quat(values):
    if values.shape[0] != 4:
        return False
    norm = np.linalg.norm(values.astype(np.float64))
    return 0.8 <= norm <= 1.2


def infer_schema(robot, frame, num_dof):
    num_cols = frame.shape[0]

    # LAFAN-style CSVs in this repo use [base_xyz, base_quat_xyzw, dofs...].
    if num_cols >= 7 + num_dof and 0.2 < frame[2] < 2.5 and is_quat(frame[3:7]):
        return ReplaySchema(dof_start=7, dof_count=num_dof, height_col=2, quat_start=3, base_xy_start=0)

    if robot == "h1" and num_cols >= num_dof:
        quat_start = 17 if num_cols >= 21 and is_quat(frame[17:21]) else None
        return ReplaySchema(dof_start=0, dof_count=10, height_col=10 if num_cols > 10 else None, quat_start=quat_start)

    if robot == "h1_2" and num_cols >= num_dof:
        quat_start = 38 if num_cols >= 42 and is_quat(frame[38:42]) else None
        return ReplaySchema(dof_start=0, dof_count=num_dof, height_col=29 if num_cols > 29 else None, quat_start=quat_start)

    if robot == "g1" and num_cols >= 6 + num_dof:
        quat_start = 38 if num_cols >= 42 and is_quat(frame[38:42]) else None
        return ReplaySchema(
            dof_start=6,
            dof_count=num_dof,
            height_col=29 if num_cols > 29 else None,
            quat_start=quat_start,
            lin_vel_start=0,
            ang_vel_start=3,
        )

    if robot in {"go2", "go2_flat"} and num_cols >= 6 + num_dof:
        quat_start = 36 if num_cols >= 40 and is_quat(frame[36:40]) else None
        return ReplaySchema(
            dof_start=6,
            dof_count=num_dof,
            height_col=21 if num_cols > 21 else None,
            quat_start=quat_start,
            base_xy_start=34 if num_cols > 35 else None,
            lin_vel_start=0,
            ang_vel_start=3,
        )

    raise ValueError(
        f"Could not infer replay schema for robot={robot!r}, num_dof={num_dof}, csv_columns={num_cols}."
    )


def set_motion_frame(env, frame, schema):
    dof_count = schema.dof_count if schema.dof_count is not None else env.num_dof
    joint_pos = frame[schema.dof_start : schema.dof_start + dof_count]
    if joint_pos.shape[0] != dof_count:
        raise ValueError(f"Frame has {joint_pos.shape[0]} DOF values, but schema expects {dof_count}.")

    replay_dof_pos = env.default_dof_pos.clone().repeat(env.num_envs, 1)
    replay_dof_pos[:, :dof_count] = torch.as_tensor(
        joint_pos, dtype=env.dof_pos.dtype, device=env.device
    ).unsqueeze(0)
    env.dof_pos[:] = replay_dof_pos
    env.dof_vel.zero_()

    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins

    if schema.base_xy_start is not None:
        base_xy = frame[schema.base_xy_start : schema.base_xy_start + 2]
        env.root_states[:, 0:2] = env.env_origins[:, 0:2] + torch.as_tensor(
            base_xy, dtype=env.root_states.dtype, device=env.device
        ).unsqueeze(0)

    if schema.height_col is not None:
        height = float(frame[schema.height_col])
        env.root_states[:, 2] = env.env_origins[:, 2] + height

    if schema.quat_start is not None:
        quat = frame[schema.quat_start : schema.quat_start + 4]
        env.root_states[:, 3:7] = torch.as_tensor(
            quat, dtype=env.root_states.dtype, device=env.device
        ).unsqueeze(0)

    env.root_states[:, 7:13] = 0.0
    if schema.lin_vel_start is not None:
        env.root_states[:, 7:10] = torch.as_tensor(
            frame[schema.lin_vel_start : schema.lin_vel_start + 3],
            dtype=env.root_states.dtype,
            device=env.device,
        ).unsqueeze(0)
    if schema.ang_vel_start is not None:
        env.root_states[:, 10:13] = torch.as_tensor(
            frame[schema.ang_vel_start : schema.ang_vel_start + 3],
            dtype=env.root_states.dtype,
            device=env.device,
        ).unsqueeze(0)

    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def step_after_state_set(env):
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def render_kinematic_frame(env):
    if not env.viewer:
        return
    if env.gym.query_viewer_has_closed(env.viewer):
        sys.exit()
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.action == "QUIT" and evt.value > 0:
            sys.exit()
        if evt.action == "toggle_viewer_sync" and evt.value > 0:
            env.enable_viewer_sync = not env.enable_viewer_sync

    if env.enable_viewer_sync:
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
    else:
        env.gym.poll_viewer_events(env.viewer)


def print_schema_details(robot, env, schema, num_cols):
    print(f"CSV columns: {num_cols}")
    if robot == "h1":
        print("H1 column map:")
        h1_joint_names = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
        ]
        for col, name in enumerate(h1_joint_names):
            print(f"  col {col:02d}: {name}")
        print("  col 10: base height")
        print("  col 11:16: two foot/end-effector xyz targets used by rewards, not DOFs")
        if num_cols >= 21:
            print("  col 17:20: root quaternion xyzw")
    print(f"Simulator DOFs ({env.num_dof}):")
    for i, name in enumerate(getattr(env, "dof_names", [])):
        source = f"csv col {schema.dof_start + i}" if i < (schema.dof_count or env.num_dof) else "default pose"
        print(f"  dof {i:02d}: {name} <- {source}")


def make_replay_env(args):
    # The base LeggedRobot constructor reads this module global during env creation.
    import legged_gym.envs.base.legged_robot as legged_robot_module

    legged_robot_module.path_to_imitation_data = args.imitation_data

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False

    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    env_class = task_registry.get_task_class(args.task)
    return env_class(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )


def replay(args):
    csv_path = Path(args.imitation_data)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    motion = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
    env = make_replay_env(args)
    schema = infer_schema(args.robot, motion[args.start_frame], env.num_dof)
    print(f"Replaying {csv_path} on {args.task}: {len(motion)} frames, schema={schema}")
    print_schema_details(args.robot, env, schema, motion.shape[1])

    end_frame = args.end_frame if args.end_frame is not None else len(motion)
    end_frame = min(end_frame, len(motion))
    frame_ids = range(args.start_frame, end_frame)
    frame_sleep = env.dt / max(args.speed_scale, 1e-6)

    while True:
        for frame_id in frame_ids:
            set_motion_frame(env, motion[frame_id], schema)
            step_after_state_set(env)
            render_kinematic_frame(env)
            if frame_sleep > 0:
                sleep(frame_sleep)
        if not args.loop:
            break


if __name__ == "__main__":
    replay(parse_args())
