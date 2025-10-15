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
import yaml

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    control_type = config['control_type']
    action_scale = config['action_scale']
    decimation = config['control_decimation']
    num_envs = config['num_envs']

class CassieRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = num_envs
        # num_observations = 169
        num_observations = 48   
        num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        }

    class control( LeggedRobotCfg.control ):
        control_type = control_type
        action_scale = action_scale

        stiffness = {   'hip_abduction': 100.0, 'hip_rotation': 100.0,
                        'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                        'toe_joint': 40.}  # [N*m/rad]
        damping = { 'hip_abduction': 3.0, 'hip_rotation': 3.0,
                    'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                    'toe_joint': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = decimation
        
    class viewer:
        ref_env = 0
        pos = [0.5, -1.5, 0.9]  # [m]
        lookat = [0.5, 0, 0.6]  # [m]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        only_positive_rewards_ji22_style = False
        class scales( LeggedRobotCfg.rewards.scales ):
            orientation_penalty = -5.0
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            # feet_air_time = 5.
            feet_air_time = 2.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

            #Imitation rewards
            imitation_angles = 15.0
            imitate_end_effector_pos = 15.0
            imitate_foot_height = 15.0
            imitation_height_penalty = -10.0

            # imitation_angles = 0.0

            # imitation_height_penalty = -0.0
            # imitate_end_effector_pos = 0.0
            # imitate_foot_height = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        resampling_time = 10.  # time before command are changed[s]
        use_heading = False
        heading_command = False
        use_imitation_commands = True
        class ranges:
            # lin_vel_x = [0.4, 0.8]  # min max [m/s]
            # lin_vel_y = [0., 0.]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [-0.0, 0.0]
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]
class CassieRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'cassie_IROS'
        max_iterations = 1000

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        


  