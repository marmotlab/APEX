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

class YunaRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = num_envs
        # num_observations = 253
        num_observations = 66
        num_actions = 18

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

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

    class init_state( LeggedRobotCfg.init_state ):
        # pos = [0.0, 0.0, 0.225] # x,y,z [m]
        pos = [0.0, 0.0, 0.3] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'base1': 0.,
            'base2': 0.,
            'base3': 0.,
            'base4': 0.,
            'base5': 0.,
            'base6': 0.,

            'shoulder1': 0.,
            'shoulder2': 0.,
            'shoulder3': 0.,
            'shoulder4': 0.,
            'shoulder5': 0.,
            'shoulder6': 0.,

            'elbow1': -1.5708,    # - pi/2
            'elbow2': 1.5708,     #   pi/2
            'elbow3': -1.5708,    # - pi/2 
            'elbow4': 1.5708,     #   pi/2
            'elbow5': -1.5708,    # - pi/2
            'elbow6': 1.5708,     #   pi/2
        }

    class control( LeggedRobotCfg.control ):
        control_type = control_type
        action_scale = action_scale

        exp_avg_decay = False
        limit_dof_pos = False
        # PD Drive parameters:
        stiffness = {'base1': 300.0,'base2': 300.0,'base3': 300.0,'base4': 300.0,'base5': 300.0,'base6': 300.0,
                'shoulder1': 300.0,'shoulder2': 300.0,'shoulder3': 300.0,'shoulder4': 300.0,'shoulder5': 300.0,'shoulder6': 300.0,
                'elbow1': 300.0,'elbow2': 300.0,'elbow3': 300.0,'elbow4': 300.0,'elbow5': 300.0,'elbow6': 300.0}  # [N*m/rad]
        damping = {'base1': 0.05,'base2': 0.05,'base3': 0.05,'base4': 0.05,'base5': 0.05,'base6': 0.05,
                     'shoulder1': 0.01,'shoulder2': 0.01,'shoulder3': 0.01,'shoulder4': 0.01,'shoulder5': 0.01,'shoulder6': 0.01,
                     'elbow1': 0.05,'elbow2': 0.05,'elbow3': 0.05,'elbow4': 0.05,'elbow5': 0.05,'elbow6': 0.05}  # [N*m/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = decimation
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/yuna/urdf/yuna.urdf'
        name = "yuna"
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)

        foot_name = "FOOT"      
        penalize_contacts_on = ["LINK"]  # , 'INTERFACE', 'LINK', 'INPUT'
        terminate_after_contacts_on = ["base_link"
                                       "elbow1__INPUT_INTERFACE",
                                       "elbow2__INPUT_INTERFACE",
                                       "elbow3__INPUT_INTERFACE",
                                       "elbow4__INPUT_INTERFACE",
                                       "elbow5__INPUT_INTERFACE",
                                       "elbow6__INPUT_INTERFACE",
                                       "leg1__LAST_LINK",
                                       "leg2__LAST_LINK",
                                       "leg3__LAST_LINK",
                                       "leg4__LAST_LINK",
                                       "leg5__LAST_LINK",
                                       "leg6__LAST_LINK"]

        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class viewer:
        ref_env = 0
        pos = [0.2, -1.5, 0.8]  # [m]
        lookat = [0.2, 0, 0.4]  # [m]

    class rewards( LeggedRobotCfg.rewards):
        only_positive_rewards = False
        sigma_rew_neg = 0.02
        only_positive_rewards_ji22_style = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -10.0
            orientation_penalty = -5.0
            # feet_air_time = 0.0
            # torques =  -0.0001
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.9
            feet_slip = -0.04

            #Imitation rewards
            imitation_height_penalty = -10.0
            imitation_angles = 15.0
            imitate_end_effector_pos = 15.0
            imitate_foot_height = 15.0
        
            # imitation_angles_pb =  0.0
            # ori_pb = 0.0
            # imitation_height_pb = 0.0
    
    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 8.
        use_imitation_commands = True
        class ranges( LeggedRobotCfg.commands.ranges ):
            # lin_vel_x = [0.3, 0.3] # min max [m/s]
            # lin_vel_y = [-0., 0.]   # min max [m/s]
            # ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0.0, 0.0]
            lin_vel_x = [-0.4, 0.4] # min max [m/s]
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            # dof_vel = 0.05
            dof_vel = 0.05
            dof_imit = 2.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.0       #0.2/0.25


class YunaRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        # run_name = 'feet_contact_force'
        experiment_name = 'yuna_IROS'
        max_iterations = 1000

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  