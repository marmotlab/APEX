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

class YunaActualRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 18
        num_observations = 253

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'

    class init_state( LeggedRobotCfg.init_state ):
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

            'elbow1': -1.5708 + 0.05,    # - pi/2
            'elbow2': 1.5708 - 0.05,     #   pi/2
            'elbow3': -1.5708 + 0.05,    # - pi/2 
            'elbow4': 1.5708 - 0.05,     #   pi/2
            'elbow5': -1.5708 + 0.05,    # - pi/2
            'elbow6': 1.5708 - 0.05,     #   pi/2
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'base1': 100.,'base2': 100.,'base3': 100.,'base4': 100.,'base5': 100.,'base6': 100.,
                     'shoulder1': 100.,'shoulder2': 100.,'shoulder3': 100.,'shoulder4': 100.,'shoulder5': 100.,'shoulder6': 100.,
                     'elbow1': 100.,'elbow2': 100.,'elbow3': 100.,'elbow4': 100.,'elbow5': 100.,'elbow6': 100.}  # [N*m/rad]
        damping = {'base1': 0.05,'base2': 0.05,'base3': 0.05,'base4': 0.05,'base5': 0.05,'base6': 0.05,
                     'shoulder1': 0.05,'shoulder2': 0.05,'shoulder3': 0.05,'shoulder4': 0.05,'shoulder5': 0.05,'shoulder6': 0.05,
                     'elbow1': 0.05,'elbow2': 0.05,'elbow3': 0.05,'elbow4': 0.05,'elbow5': 0.05,'elbow6': 0.05}  # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.5

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        flip_visual_attachments = False
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/yuna/urdf/yuna.urdf'
        name = "yuna"
        """
        Foot contact forces not being detected! In urdf it's __FOOT but still
        """
        foot_name = "FOOT"      
        penalize_contacts_on = ["LINK"]  # , 'INTERFACE', 'LINK', 'INPUT'
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 300.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            pass
        
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.1, 0.1]    # min max [rad/s]
            heading = [-3.14, 3.14]

class YunaActualRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'yuna_rough'
        load_run = -1

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01