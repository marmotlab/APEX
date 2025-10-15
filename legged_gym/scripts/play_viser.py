import os
import time
import numpy as np
import yaml
from collections import deque
from typing import Optional

# Isaac Gym before torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, export_policy_as_jit

import torch

import viser
import viser.transforms as tf  # noqa: F401  (kept for runtime toggles)
import viser.uplot
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation as R

# ------------------------ Toggles ------------------------
EXPORT_POLICY         = True
RECORD_FRAMES         = False   # (kept — unused here but available)
LOG_IMITATION_ERROR   = False
VISUALIZE_IMITATION   = True    # keep feature; can toggle at runtime

# ------------------------ Config / assets ------------------------
with open("legged_gym/envs/param_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
PATH_TO_IMIT = config["path_to_imitation_data"]

# Load imitation (only if needed)
imit_np = None
df_imit = None
if LOG_IMITATION_ERROR or VISUALIZE_IMITATION:
    import pandas as pd
    df_imit = pd.read_csv(PATH_TO_IMIT, parse_dates=False)
    # feet EE cols 22:34 -> [T,4,3]
    imit_np = df_imit.iloc[:, 22:34].to_numpy().reshape(-1, 4, 3)

# ------------------------ Constants ------------------------
PINK_GRID = dict(
    cell_color=(235, 190, 235),
    cell_thickness=0.5,
    section_color=(245, 160, 245),
    section_thickness=0.8,
)
FOOT_COLORS = np.array(
    [
        [250, 120, 120],  # FL
        [120, 250, 160],  # FR
        [120, 140, 255],  # RL
        [255, 230, 120],  # RR
    ],
    dtype=np.uint8,
)

# ------------------------ Scene / UI ------------------------
def setup_scene(server: viser.ViserServer) -> None:
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(True, True)
    server.scene.add_grid(
        name="/grid_floor",
        width=100.0,
        height=100.0,
        width_segments=100,
        height_segments=100,
        plane="xy",
        cell_size=1.0,
        section_size=1.0,
        shadow_opacity=0.10,
        position=(0.0, 0.0, 0.0),
        visible=True,
        **PINK_GRID,
    )

def _uplot(server, series, aspect=2.0, init_dim=1):
    # small helper to avoid repetition
    xs = tuple(np.array([0]) for _ in range(init_dim))
    return server.gui.add_uplot(
        data=xs, series=series,
        scales={"x": viser.uplot.Scale(time=False, auto=True),
                "y": viser.uplot.Scale(auto=True)},
        legend=viser.uplot.Legend(show=True),
        aspect=aspect,
    )

def setup_real_time_plots(server: viser.ViserServer, max_points: int = 300) -> dict:
    plot_state = {
        "time": np.array([]),
        "max_points": max_points,
        "plots": {},
        "buf": {  # minimal set we actually render
            "dof_vel": [],
            "dof_torque": [],
            "base_vel_x": [],
            "base_vel_y": [],
            "base_vel_z": [],
            "base_vel_yaw": [],
            "command_x": [],
            "command_y": [],
            "command_yaw": [],
            "contact_forces_z": [],
        },
    }
    with server.gui.add_folder("📊 Real-time Plots"):
        with server.gui.add_folder("Base Velocity"):
            plot_state["plots"]["base_vel_x"] = _uplot(
                server,
                (
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="measured", stroke="blue", width=2),
                    viser.uplot.Series(label="commanded", stroke="orange", width=2, points_show=False),
                ),
                init_dim=3,
            )
            plot_state["plots"]["base_vel_y"] = _uplot(
                server,
                (
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="measured", stroke="blue", width=2),
                    viser.uplot.Series(label="commanded", stroke="orange", width=2, points_show=False),
                ),
                init_dim=3,
            )
        with server.gui.add_folder("Forces & Torques"):
            plot_state["plots"]["contact_forces"] = _uplot(
                server,
                (
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="FL", stroke="red", width=1.5),
                    viser.uplot.Series(label="FR", stroke="green", width=1.5),
                    viser.uplot.Series(label="RL", stroke="blue", width=1.5),
                    viser.uplot.Series(label="RR", stroke="orange", width=1.5),
                ),
                init_dim=5,
            )
            plot_state["plots"]["torques"] = _uplot(
                server,
                (
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="torque", stroke="purple", width=2),
                ),
                init_dim=2,
            )
    return plot_state

def _trim_timebuf(ps: dict):
    max_pts = ps["max_points"]
    if len(ps["time"]) > max_pts:
        ps["time"] = ps["time"][-max_pts:]
        for k in ps["buf"]:
            if ps["buf"][k]:
                ps["buf"][k] = ps["buf"][k][-max_pts:]

def update_real_time_plots(ps: dict, log: dict, t: float) -> None:
    ps["time"] = np.append(ps["time"], t)
    for k, v in log.items():
        if k in ps["buf"]:
            ps["buf"][k].append(v)
    _trim_timebuf(ps)
    time_data = ps["time"]

    # base vel X/Y with commands
    for axis, cmd in (("base_vel_x", "command_x"), ("base_vel_y", "command_y")):
        if ps["buf"][axis]:
            measured = np.array(ps["buf"][axis])
            commanded = np.array(ps["buf"][cmd]) if ps["buf"][cmd] else np.zeros_like(measured)
            ps["plots"][axis].data = (time_data, measured, commanded)

    # contact forces FL,FR,RL,RR (z)
    if ps["buf"]["contact_forces_z"]:
        forces = np.array(ps["buf"]["contact_forces_z"])
        if forces.ndim == 2 and forces.shape[1] >= 4:
            ps["plots"]["contact_forces"].data = (
                time_data, forces[:, 0], forces[:, 1], forces[:, 2], forces[:, 3]
            )

    # torques (single joint stream shown as example)
    if ps["buf"]["dof_torque"]:
        ps["plots"]["torques"].data = (time_data, np.array(ps["buf"]["dof_torque"]))

# ------------------------ Camera helpers ------------------------
def _exp_smooth_factor(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0.0 else 1.0 - np.exp(-dt / max(1e-6, tau))

def _alpha_beta_update(pos_est, vel_est, meas, dt, tau_pos, tau_vel):
    pred = pos_est + vel_est * dt
    residual = meas - pred
    alpha = _exp_smooth_factor(dt, tau_pos)
    beta  = _exp_smooth_factor(dt, tau_vel)
    pos_est = pred + alpha * residual
    vel_est = vel_est + (beta / max(1e-6, dt)) * residual
    return pos_est, vel_est

def _limit_vec(vec, max_norm):
    n = np.linalg.norm(vec)
    return vec * (max_norm / n) if (n > max_norm and n > 0) else vec

def _quat_xyzw_to_yaw(q_xyzw: np.ndarray) -> float:
    x, y, z, w = q_xyzw
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(s, c))

def _unwrap_angle(curr: float, last_unwrapped: Optional[float]) -> float:
    if last_unwrapped is None:
        return curr
    delta = curr - (last_unwrapped % (2.0 * np.pi))
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return last_unwrapped + delta

def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def setup_camera_gui(server: viser.ViserServer, connected: dict) -> dict:
    state = dict(
        follow_enabled=True, mode="side",
        distance=2.4,      # zoom out
        height=0.6,        # a bit higher
        angle=-28.0,
        orbit_speed=0.5, orbit_theta=0.0,
        # stronger smoothing by default
        tau_pos=0.30, tau_vel=0.45, tau_z=1.20, tau_cam=0.35,
        max_speed=6.0, max_accel=30.0,
        pos_est=None, vel_est=None, z_follow=None,
        tau_yaw=0.35, yaw_est=None, yaw_vel=0.0, yaw_look_ahead=0.6, _yaw_last=None,
        smooth=0.3, z_fixed=None,
        # --- z stabilization + auto zoom ---
        z_gain=0.0,        # 0=lock horizon, 1=follow Z fully
        lock_z=True,       # <— hard lock Z by default
        auto_zoom=True,
        min_dist=2.0,
        max_dist=4.0,
        speed_zoom_gain=0.25,  # m per (m/s)
        # orbit zoom scale
        orbit_zoom=1.6,    # <— more zoomed out in ORBIT
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        connected[client.client_id] = client
        client.camera.near, client.camera.far = 0.05, 80.0
        client.camera.up_direction = (0.0, 0.0, 1.0)
        with client.gui.add_folder("Camera") as _:
            follow = client.gui.add_checkbox("Follow", initial_value=state["follow_enabled"])
            mode   = client.gui.add_dropdown("Mode", ("behind","side","top","cinematic","orbit","auto_angle"), initial_value=state["mode"])
            dist   = client.gui.add_slider("Distance", 0.6, 5.0, 0.1, state["distance"])
            height = client.gui.add_slider("Height", 0.3, 3.0, 0.1, state["height"])
            angle  = client.gui.add_slider("Angle", -90.0, 90.0, 1.0, state["angle"])
            smooth = client.gui.add_slider("Smooth", 0.05, 0.5, 0.01, state["smooth"])
            orbit  = client.gui.add_slider("Orbit Speed", 0.1, 3.0, 0.1, state["orbit_speed"])
            yaw_la = client.gui.add_slider("Yaw Look-ahead", 0.0, 2.0, 0.1, state["yaw_look_ahead"])
            z_gain = client.gui.add_slider("Z Gain (0=lock,1=follow)", 0.0, 1.0, 0.05, state["z_gain"])
            lock_z = client.gui.add_checkbox("Lock Z (horizon)", initial_value=True)
            autozm = client.gui.add_checkbox("Auto Zoom w/ Speed", initial_value=state["auto_zoom"])
            mindst = client.gui.add_slider("Min Distance", 0.8, 6.0, 0.1, state["min_dist"])
            maxdst = client.gui.add_slider("Max Distance", 1.0, 10.0, 0.1, state["max_dist"])
            zgainv = client.gui.add_slider("Speed→Zoom Gain", 0.0, 0.6, 0.05, state["speed_zoom_gain"])
            orbz   = client.gui.add_slider("Orbit Zoom Scale", 1.0, 3.0, 0.1, state["orbit_zoom"])
            preset_front = client.gui.add_button("Front")
            preset_side  = client.gui.add_button("Side")
            preset_top   = client.gui.add_button("Top")
            reset_btn    = client.gui.add_button("Reset")

        follow.on_update(lambda _: state.update(follow_enabled=follow.value))
        def _set(k, w): w.on_update(lambda _ : state.update(**{k: w.value}))
        _set("mode", mode); _set("distance", dist); _set("height", height); _set("angle", angle)
        _set("smooth", smooth); _set("orbit_speed", orbit); _set("yaw_look_ahead", yaw_la)
        _set("z_gain", z_gain)
        lock_z.on_update(lambda _: state.update(lock_z=lock_z.value))
        autozm.on_update(lambda _: state.update(auto_zoom=autozm.value))
        _set("min_dist", mindst); _set("max_dist", maxdst); _set("speed_zoom_gain", zgainv)
        _set("orbit_zoom", orbz)

        def _preset(pos, look):
            state["follow_enabled"] = False; follow.value = False
            with client.atomic():
                client.camera.position = pos; client.camera.look_at = look

        preset_front.on_click(lambda _ : _preset((0.0, -4.0, 1.5), (0.0, 0.0, 0.5)))
        preset_side .on_click(lambda _ : _preset((-4.0,  0.0, 1.5), (0.0, 0.0, 0.5)))
        preset_top  .on_click(lambda _ : _preset((0.0,  0.0, 8.0),  (0.0, 0.0, 0.0)))
        reset_btn   .on_click(lambda _ : (state.update(follow_enabled=True), setattr(follow, "value", True)))

    @server.on_client_disconnect
    def _(client: viser.ClientHandle):
        connected.pop(client.client_id, None)

    return state

def update_camera(connected, state, robot_pos, dt, robot_quat_xyzw=None):
    if not state["follow_enabled"] or not connected:
        return

    # lazy init
    if state.get("pos_est") is None:
        p = np.array(robot_pos, dtype=np.float64)
        state.update(
            pos_est=p, vel_est=np.zeros(3, np.float64),
            z_follow=float(p[2]), z_fixed=float(p[2]),
            _sm_cam_pos=p.copy(), _sm_cam_look=p.copy(),
        )
        yaw0 = _quat_xyzw_to_yaw(np.asarray(robot_quat_xyzw, np.float64)) if robot_quat_xyzw is not None else 0.0
        state.update(yaw_est=yaw0, yaw_vel=0.0, _yaw_last=yaw0)

    # α–β pos
    meas = np.array(robot_pos, np.float64)
    pos_est, vel_est = _alpha_beta_update(state["pos_est"], state["vel_est"], meas, dt, state["tau_pos"], state["tau_vel"])
    vel_est = _limit_vec(vel_est, state["max_speed"])
    vel_est = state["vel_est"] + _limit_vec(vel_est - state["vel_est"], state["max_accel"] * dt)

    # yaw α–β
    yaw_wrapped = _quat_xyzw_to_yaw(np.asarray(robot_quat_xyzw, np.float64)) if robot_quat_xyzw is not None else 0.0
    yaw_meas = _unwrap_angle(yaw_wrapped, state["_yaw_last"]); state["_yaw_last"] = yaw_meas
    yaw_est, yaw_vel = _alpha_beta_update(
        state["yaw_est"], state["yaw_vel"], yaw_meas, dt, state["tau_yaw"], state["tau_yaw"] * 1.25
    )
    # gentle clamp to avoid whip-pans
    yaw_vel = float(np.clip(yaw_vel, -2.0*np.pi, 2.0*np.pi))

    # Z stabilization: hard lock if requested (fully ignore vertical robot motion)
    if state["lock_z"] or state["z_gain"] == 0.0:
        z_target = state["z_fixed"]
    else:
        z_target = state["z_fixed"] * (1.0 - state["z_gain"]) + pos_est[2] * state["z_gain"]
    az = _exp_smooth_factor(dt, state["tau_z"])
    z_follow = (1.0 - az) * state["z_follow"] + az * z_target
    follow_pos = np.array([pos_est[0], pos_est[1], z_follow], np.float64)

    # Optional auto-zoom based on planar speed
    d_base = state["distance"]
    if state["auto_zoom"]:
        planar_speed = float(np.linalg.norm(vel_est[:2]))
        d_base = np.clip(state["min_dist"] + state["speed_zoom_gain"] * planar_speed,
                         state["min_dist"], state["max_dist"])
    d, h, mode = d_base, state["height"], state["mode"]
    Rz = _Rz(yaw_est)
    look_ahead_vec = np.array([state["yaw_look_ahead"], 0.0, 0.3])
    if mode == "behind":
        rel = np.array([-d, 0.0, h]); cam_t = follow_pos + Rz @ rel; look_t = follow_pos + Rz @ look_ahead_vec
    elif mode == "side":
        rel = np.array([0.0, -d, h]); cam_t = follow_pos + Rz @ rel; look_t = follow_pos + Rz @ look_ahead_vec
    elif mode == "cinematic":
        rel = np.array([-0.7*d, 0.5*d, h]); cam_t = follow_pos + Rz @ rel; look_t = follow_pos + Rz @ look_ahead_vec
    elif mode == "top":
        cam_t = follow_pos + np.array([0.0, 0.0, max(d * 1.5, h + 1.0)]); look_t = follow_pos
    elif mode == "orbit":
        state["orbit_theta"] += state["orbit_speed"] * dt
        r = d * state["orbit_zoom"]  # <— zoom out more in orbit mode
        rel = np.array([np.cos(state["orbit_theta"]) * r, np.sin(state["orbit_theta"]) * r, h])
        cam_t = follow_pos + Rz @ rel; look_t = follow_pos + Rz @ look_ahead_vec
    else:  # auto_angle
        ang = np.radians(state["angle"]); rel = np.array([-d*np.cos(ang), -d*np.sin(ang), h])
        cam_t = follow_pos + Rz @ rel; look_t = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.0])

    # ease
    ac = _exp_smooth_factor(dt, state["tau_cam"])
    state["_sm_cam_pos"]  = (1.0 - ac) * state["_sm_cam_pos"]  + ac * cam_t
    state["_sm_cam_look"] = (1.0 - ac) * state["_sm_cam_look"] + ac * look_t

    # commit + push
    state.update(pos_est=pos_est, vel_est=vel_est, z_follow=z_follow, yaw_est=float(yaw_est), yaw_vel=float(yaw_vel))
    for c in connected.values():
        try:
            with c.atomic():
                c.camera.position = tuple(state["_sm_cam_pos"])
                c.camera.look_at  = tuple(state["_sm_cam_look"])
                c.camera.up_direction = (0.0, 0.0, 1.0)
        except Exception:
            pass

# ------------------------ Small utils ------------------------
def overwrite_point_cloud(server, name: str, points: np.ndarray, colors, point_size: float):
    server.scene.add_point_cloud(name, points=points, colors=colors, point_size=point_size)

def _get_foot_positions(env, robot_i: int):
    # Try fast path
    try:
        fp = env.foot_positions[robot_i].detach().cpu().numpy()
        if not np.allclose(fp, 0.0):
            return fp
    except AttributeError:
        pass
    # Fallback via rigid body state
    if hasattr(env, "rigid_body_state") and hasattr(env, "feet_indices"):
        rbs = env.rigid_body_state
        feet_idx = env.feet_indices.detach().cpu().numpy()
        num_rb = rbs.shape[0] // env.num_envs
        base_idx = robot_i * num_rb
        return np.stack([rbs[base_idx + fi, :3].detach().cpu().numpy() for fi in feet_idx], axis=0)
    return None

# ------------------------ Main loop ------------------------
def play(args):
    # Env + policy
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, out_dir)
        print("Exported policy:", out_dir)

    # Viser
    server = viser.ViserServer()
    setup_scene(server)
    connected = {}
    cam_state = setup_camera_gui(server, connected)
    plot_state = setup_real_time_plots(server)

    # URDF
    go2_urdf_path = "/home/marmot/Sood/delete_later/kakashi/resources/robots/go2/urdf/go2.urdf"
    urdf = yourdfpy.URDF.load(go2_urdf_path, load_meshes=True, build_scene_graph=True,
                              load_collision_meshes=False, build_collision_scene_graph=False)
    root = server.scene.add_frame("/robot", axes_length=0.0, axes_radius=0.0)
    vbot = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)

    # GUI toggles / HUD
    with server.gui.add_folder("Viz"):
        show_trail = server.gui.add_checkbox("Motion Trail", initial_value=True)
        show_foot  = server.gui.add_checkbox("Foot Trails", initial_value=True)
        show_imit  = server.gui.add_checkbox("Imitation Targets", initial_value=False)
    with server.gui.add_folder("Perf"):
        fps_txt  = server.gui.add_text("FPS", "—", disabled=True)
        step_txt = server.gui.add_text("Step", "0", disabled=True)
        pos_txt  = server.gui.add_text("Base Pos", "—", disabled=True)
        vel_txt  = server.gui.add_text("Base Vel", "—", disabled=True)

    # Indices / limits
    robot_i, joint_i = 0, 2
    stop_state_log = 1000
    stop_rew_log = env.max_episode_length + 1

    # Histories
    path_pts = deque(maxlen=600)
    foot_hist = deque(maxlen=240)

    # Timing
    target_dt = 1.0 / 60.0
    t_prev = time.perf_counter()
    fps_alpha, fps_est = 0.1, None

    # Imitation error logs
    FF_counter = 0
    err_angles, err_vel, err_h = [], [], []

    for i in range(10 * int(env.max_episode_length)):
        loop_t0 = time.perf_counter()

        with torch.no_grad():
            act = policy(obs)
            obs, _, rews, dones, infos = env.step(act)

        # State fetch
        dof_pos = env.dof_pos[robot_i].detach().cpu().numpy()
        base_state = env.root_states[robot_i].detach().cpu().numpy()
        base_pos = base_state[:3]
        base_quat_xyzw = base_state[3:7]
        base_vel = env.base_lin_vel[robot_i].detach().cpu().numpy()

        # URDF pose
        vbot.update_cfg(dof_pos)
        root.position = base_pos
        root.wxyz = np.array([base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]])

        # Camera
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        update_camera(connected, cam_state, base_pos, dt, base_quat_xyzw)

        # Trails
        path_pts.append(base_pos.copy())
        if show_trail.value and len(path_pts) >= 2:
            overwrite_point_cloud(server, "/trail", np.array(path_pts), colors=(0, 220, 200), point_size=0.014)

        foot_pos = _get_foot_positions(env, robot_i)
        if foot_pos is not None:
            foot_hist.append(foot_pos.copy())
            if show_foot.value and len(foot_hist) >= 2:
                recent = np.stack(foot_hist, axis=0)  # [T,4,3]
                for k in range(4):
                    overwrite_point_cloud(server, f"/foot_trail_{k}", recent[:, k, :], colors=FOOT_COLORS[k], point_size=0.011)

        # Imitation targets
        if show_imit.value and (imit_np is not None):
            if hasattr(env, "imitation_index"):
                idx = int(env.imitation_index[robot_i].detach().cpu().numpy())
                idx = max(0, min(idx, len(imit_np) - 1))
            else:
                idx = i % len(imit_np)
            ee_world = R.from_quat(base_quat_xyzw).apply(imit_np[idx]) + base_pos
            overwrite_point_cloud(server, "/imit_targets", ee_world, colors=(0, 255, 0), point_size=0.02)

        # HUD (~10 Hz)
        if i % 6 == 0:
            inst_fps = 1.0 / max(1e-6, (time.perf_counter() - loop_t0))
            fps_est = inst_fps if fps_est is None else (1 - fps_alpha) * fps_est + fps_alpha * inst_fps
            fps_txt.value  = f"{fps_est:5.1f}"
            step_txt.value = f"{i}"
            pos_txt.value  = f"{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}"
            vel_txt.value  = f"{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}"

        # Optional imitation error logging
        if LOG_IMITATION_ERROR and i > 100 and (df_imit is not None):
            dof_pos_imit = df_imit.iloc[FF_counter, 6:18].values
            err_angles.append(np.square(np.abs(dof_pos - dof_pos_imit)))
            base_vx_cmd = obs[robot_i, 3].detach().cpu().numpy() / 2.0
            err_vel.append(np.square(np.abs(base_vel[0] - base_vx_cmd)))
            height_imit = df_imit.iloc[FF_counter, 21]
            err_h.append(np.square(np.abs(base_pos[2] - height_imit)))
            FF_counter += 1

        # Episode rollups
        if i % env.max_episode_length == 0 and i > 0:
            path_pts.clear(); foot_hist.clear()
            if LOG_IMITATION_ERROR and err_angles:
                print("Avg Err Angles", float(np.sqrt(np.mean(err_angles))))
                print("Avg Err Vel   ", float(np.sqrt(np.mean(err_vel))))
                print("Avg Err Height", float(np.sqrt(np.mean(err_h))))
                err_angles.clear(); err_vel.clear(); err_h.clear(); FF_counter = 0

        # Real-time plots (compact)
        if i <= stop_state_log:
            log = dict(
                dof_vel=float(env.dof_vel[robot_i, joint_i].item()),
                dof_torque=float(env.torques[robot_i, joint_i].item()),
                command_x=float(env.commands[robot_i, 0].item()),
                command_y=float(env.commands[robot_i, 1].item()),
                command_yaw=float(env.commands[robot_i, 2].item()),
                base_vel_x=float(env.base_lin_vel[robot_i, 0].item()),
                base_vel_y=float(env.base_lin_vel[robot_i, 1].item()),
                base_vel_z=float(env.base_lin_vel[robot_i, 2].item()),
                base_vel_yaw=float(env.base_ang_vel[robot_i, 2].item()),
                contact_forces_z=env.contact_forces[robot_i, env.feet_indices, 2].detach().cpu().numpy(),
            )
            if i % 3 == 0:
                update_real_time_plots(plot_state, log, i * env.dt)
            if i == stop_state_log:
                print("📊 Real-time plotting complete - all plots updated in browser")

        # Reward prints (without matplotlib)
        if 0 < i < stop_rew_log and infos and "episode" in infos:
            if torch.sum(env.reset_buf).item() > 0:
                for k, v in infos["episode"].items():
                    if "rew" in k:
                        print(f"Episode {k}: {v.item():.3f}")
        elif i == stop_rew_log:
            print("📊 Episode reward logging complete")

        # frame pacing
        elapsed = time.perf_counter() - loop_t0
        sleep = target_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

if __name__ == "__main__":
    args = get_args()
    play(args)
