import xml.etree.ElementTree as ET
import math
import torch


WN = 2.0 * math.pi * 10.0   # natural frequency (rad/s): 10 Hz
Z  = 2.0                    # damping ratio (overdamped)

#Rough Inertia values for GO2 Links
I_HIP   = 0.000482559   # kg·m^2  
I_THIGH = 0.007047936   # kg·m^2  
I_CALF  = 0.003141275   # kg·m^2  

def load_effort_limits_from_urdf(path: str):
    """
    Returns a dict {joint_name: tau_max} from URDF <joint><limit effort="..."/>.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    tau = {}
    for j in root.findall("joint"):
        name = j.get("name")
        lim = j.find("limit")
        if lim is not None and lim.get("effort") is not None:
            try:
                tau[name] = float(lim.get("effort"))
            except ValueError:
                pass
    return tau

def load_velocity_limits_from_urdf(path: str):
    """
    Returns a dict {joint_name: vel_max} from URDF <joint><limit velocity="..."/>.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    vel = {}
    for j in root.findall("joint"):
        name = j.get("name")
        lim = j.find("limit")
        if lim is not None and lim.get("velocity") is not None:
            try:
                vel[name] = float(lim.get("velocity"))
            except ValueError:
                pass
    return vel


def kp_from_I(I: float) -> float:
    return I * (WN ** 2)

def kd_from_I(I: float) -> float:
    return 2.0 * I * Z * WN

def build_pd_gains_for_go2():
    """
    Returns two dicts: Kp[joint_name], Kd[joint_name] for the GO2 joint names.
    """
    Kp, Kd = {}, {}
    hips   = ["FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint"]
    thighs = ["FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint"]
    calves = ["FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint"]

    for j in hips:
        Kp[j], Kd[j] = kp_from_I(I_HIP),   kd_from_I(I_HIP)
    for j in thighs:
        Kp[j], Kd[j] = kp_from_I(I_THIGH), kd_from_I(I_THIGH)
    for j in calves:
        Kp[j], Kd[j] = kp_from_I(I_CALF),  kd_from_I(I_CALF)
    return Kp, Kd

def build_alpha_tensor(dof_names, kp_dict, tau_max_dict, device) -> torch.Tensor:
    """
    alpha_j = 0.25 * tau_max_j / kp_j  (radians of target per |action|=1)
    Returns tensor [num_dof].
    """
    alpha = torch.ones(len(dof_names), device=device)
    for i, name in enumerate(dof_names):
        kp   = kp_dict.get(name, None)
        tauM = tau_max_dict.get(name, None)
        if kp is not None and tauM is not None and kp > 0.0:
            alpha[i] = 0.25 * (tauM / kp)
    return alpha