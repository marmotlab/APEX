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

class A1FlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 48

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.3] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        exp_avg_decay = False
        limit_dof_pos = False
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 3.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
  
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.28
        only_positive_rewards = False
        sigma_rew_neg = 0.02
        only_positive_rewards_ji22_style = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -100.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -6.0 
            feet_air_time = 2.0
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            hip_nominal = 0.0
            thigh_nominal = 0.0
            # torque_symmetry = -1.0
            # imitation_angles = 4.0
            # imitation_lin_vel = 1.0
            # imitation_ang_vel = 0.5
            # imitation = -1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        resampling_time = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]
            # lin_vel_x = [0.0, 0.0]  # min max [m/s]
            # lin_vel_y = [0.0, 0.0]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [-0.0, 0.0]

class A1FlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_a1'