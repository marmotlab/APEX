# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 36
        num_actions = 10

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0., 0., 0.75]        # x,y,z [m]
        # pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = {
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': -0.2,
            'left_knee': 0.25,  # 0.6
            'left_ankle': 0.0,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': -0.2,
            'right_knee': 0.25,  # 0.6
            'right_ankle': 0.0,
        }

        damping = {
            'left_hip_yaw': 5.,
            'left_hip_abad': 5.,
            'left_hip_pitch': 5.,
            'left_knee': 5.,
            'left_ankle': 5.,
            'right_hip_yaw': 5.,
            'right_hip_abad': 5.,
            'right_hip_pitch': 5.,
            'right_knee': 5.,
            'right_ankle': 5.
        }

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'T_ref_decay'
        # control_type = 'P_ref_decay'
        # control_type = 'T'
        # control_type = 'P'
        control_type = 'T_vanish_humanoid'
        # control_type = 'T_low_gain_inference'
        exp_avg_decay = False
        limit_dof_pos = False
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 10.0
        # action_scale = 1.0
        # hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 4
        decimation = 4

    class domain_rand:
        # randomize_friction = True
        # friction_range = [0.5, 1.25]
        # randomize_base_mass = True
        # added_mass_range = [-1., 1.]
        # push_robots = True
        # push_interval_s = 5
        # max_push_vel_xy = 1.
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

  
    class asset( LeggedRobotCfg.asset ):
        fix_base_link = False # fixe the base of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf'
        name = "humanoid"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = [
            'base',
            'left_upper_leg',
            'left_lower_leg',
            'right_upper_leg',
            'right_lower_leg',
            'left_upper_arm',
            'right_upper_arm',
            'left_lower_arm',
            'right_lower_arm',
            'left_hand',
            'right_hand',
        ]
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.28
        only_positive_rewards = False
        sigma_rew_neg = 0.02
        only_positive_rewards_ji22_style = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -10.0
            orientation_penalty = -5.0
            # torques =  -0.0001
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.9
            # lin_vel_z = -2.0
            # dof_pos_limits = -10.0
            # feet_air_time = 0.0
            # base_height = -30.0
            # no_fly = 0.0
            # action_rate = -0.01
            # collision = -5.0

            # feet_slip = -0.0

            # termination = -0.0
            # orientation = -5.0
            # torques =  -0.00000
            # tracking_lin_vel = 0.0
            # tracking_ang_vel = 0.0
            # lin_vel_z = -0.0
            # dof_pos_limits = 0.0
            # dof_acc = -2.5e-7
            # dof_vel = -0.00000
            # feet_air_time = 0.0
            # base_height = -30.0
            # no_fly = 0.0
            # action_rate = -0.0
            # collision = -0.0
            # ang_vel_xy = 0.0

            # imitation_angles = 0.9
            # imitation_lin_vel = 0.0
            # imitation_ang_vel = 0.0
            # # imitation_height = 0.0
            # feet_slip = -0.04
            # imitation_height_penalty = -10.0
            # imitate_quat_penalty = -0.0

            # imitate_quat = 0.0
            # imitate_end_effector_pos = 1.2
            # imitate_base_pos = 0.0
            # imitate_base_end_effector_pos_world = 0.0
            # imitate_foot_height = 2.0
            
            # imitation_angles_indiv_legs = 2.0

            # imitation_angles_pb =  0.0
            # ori_pb = 0.0
            # imitation_height_pb = 0.0

            raibert_heuristic = 0.0
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0

            feet_clearance_cmd_linear = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        resampling_time = 10.  # time before command are changed[s]
        use_heading = False
        heading_command = False
        use_imitation_commands = True
        class ranges:
            # lin_vel_x = [0.8, 0.8]  # min max [m/s]
            # lin_vel_y = [0., 0.]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [-0.0, 0.0]
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            dof_imit = 1.0
        clip_observations = 100.
        clip_actions = 100.

class HumanoidCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        # experiment_name = 'flat_humanoid_imit_torque_low_gain'
        experiment_name = 'flat_humanoid_rough'
        max_iterations = 2100