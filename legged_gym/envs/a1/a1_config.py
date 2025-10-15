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
	reward_group_num = config['reward_group_num']

class A1RoughCfg( LeggedRobotCfg ):
	class env( LeggedRobotCfg.env ):
		num_observations = 39
		num_privileged_obs = 48
		num_envs = num_envs
		episode_length_s = 20 # episode length in seconds

	class init_state( LeggedRobotCfg.init_state ):
		pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
		control_type = control_type
		exp_avg_decay = False
		limit_dof_pos = False
		stiffness = {'joint': 20.}  # [N*m/rad]
		damping = {'joint': 0.5}     # [N*m*s/rad]
		# action scale: target angle = actionScale * action + defaultAngle
		action_scale = action_scale
		# decimation: Number of control action updates @ sim DT per policy DT
		decimation = decimation

	class terrain:
		mesh_type = 'plane'
		# mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
		horizontal_scale = 0.1 # [m]
		vertical_scale = 0.2 # [m]
		border_size = 25 # [m]
		curriculum = False
		static_friction = 1.0
		dynamic_friction = 1.0
		restitution = 0.
		# rough terrain only:
		measure_heights = False
		measured_points_x = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # 1mx1.6m rectangle (without center line)
		measured_points_y = [-0.1, -0.1, -0.1, -0.1, -0.1, 0., 0.1, 0.1, 0.1, 0.1, 0.1]
		selected = False # select a unique terrain type and pass all arguments
		terrain_kwargs = None # Dict of arguments for selected terrain
		max_init_terrain_level = 5 # starting curriculum state
		terrain_length = 1.
		terrain_width = 1.
		num_rows= 1 # number of terrain rows (levels)
		num_cols = 1 # number of terrain cols (types)
		# terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
		# terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
		terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
		# trimesh only:
		slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
		
	class asset( LeggedRobotCfg.asset ):
		file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
		name = "a1"
		foot_name = "foot"
		penalize_contacts_on = ["thigh", "calf"]
		terminate_after_contacts_on = ["base"]
		self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
	class rewards( LeggedRobotCfg.rewards ):
		soft_dof_pos_limit = 0.9
		base_height_target = 0.29
		only_positive_rewards = False
		sigma_rew_neg = 0.02
		only_positive_rewards_ji22_style = False
		multi_critic = False
		reward_group_num = 1
		class scales( LeggedRobotCfg.rewards.scales ):
			#Imitation rewards 
			# class group2:
			# 	imitation_height_penalty = -10.0

			# 	imitation_angles = 1.0
			# 	imitate_end_effector_pos = 1.0
			# 	imitate_foot_height = 0.0
			# 	imitate_base_end_effector_pos_world = 0.0
			# 	imitate_quat = 0.5
			# 	imitate_lin_vel = 0.0
			# 	imitate_ang_vel = 0.0

			# # termination = -100.0
			# # # tracking_lin_vel = 1.0
			# # # tracking_ang_vel = 0.5
			# class group1:
			# 	lin_vel_z = -2.0
			# 	ang_vel_xy = -0.0
			# 	orientation = -1.0
			# 	torques = -0.0002
			# 	# dof_vel = -0.
			# 	# dof_acc = -0.0
			# 	# base_height = -0. 
			# 	feet_air_time =  1.0
			# 	collision = -1.0
			# 	feet_stumble = -0.0 
			# 	action_rate = -0.01
			# 	orientation_penalty = -0.0
			# 	torques =  -0.0001
			# 	tracking_lin_vel = 1.0
			# 	tracking_ang_vel = 0.5

			# 	feet_slip = -0.4

			# 	#Extra rewards to play with
			# 	raibert_heuristic = 0.0
			# 	tracking_contacts_shaped_force = 0.0
			# 	tracking_contacts_shaped_vel = 0.0
				imitation_height_penalty = -10.0

				imitation_angles = 1.0
				imitate_end_effector_pos = 1.0
				imitate_foot_height = 0.0
				imitate_base_end_effector_pos_world = 0.0
				imitate_quat = 0.5
				imitate_lin_vel = 0.0
				imitate_ang_vel = 0.0

			# termination = -100.0
			# # tracking_lin_vel = 1.0
			# # tracking_ang_vel = 0.5
				lin_vel_z = -2.0
				ang_vel_xy = -0.0
				orientation = -1.0
				torques = -0.0002
				# dof_vel = -0.
				# dof_acc = -0.0
				# base_height = -0. 
				feet_air_time =  1.0
				collision = -1.0
				feet_stumble = -0.0 
				action_rate = -0.01
				orientation_penalty = -0.0
				torques =  -0.0001
				tracking_lin_vel = 1.0
				tracking_ang_vel = 0.5

				feet_slip = -0.4

				#Extra rewards to play with
				raibert_heuristic = 0.0
				tracking_contacts_shaped_force = 0.0
				tracking_contacts_shaped_vel = 0.0
	
	class commands(LeggedRobotCfg.commands):
		curriculum = False
		resampling_time = 10.  # time before command are changed[s]
		use_heading = False
		heading_command = False
		use_imitation_commands = False
		class ranges:
			lin_vel_x = [0.7, 0.7]  # min max [m/s]
			lin_vel_y = [0., 0.]  # min max [m/s]
			ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
			heading = [-0.0, 0.0]
			# lin_vel_x = [-1.0, 1.0]  # min max [m/s]
			# lin_vel_y = [-0.6, 0.6]  # min max [m/s]
			# ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
			# heading = [-3.14, 3.14]

	class domain_rand:
		randomize_friction = True
		friction_range = [0.3, 1.25]
		randomize_base_mass = True
		added_mass_range = [-1., 3.]
		push_robots = True
		push_interval_s = 10
		max_push_vel_xy = 1.2
		max_push_vel_ang = 1.0
		randomize_lag_timesteps = False
		lag_timesteps = 6
		randomize_motor_offset = True
		motor_offset_range = [-0.03, 0.03]
		randomize_motor_strength = True
		motor_strength_range = [0.8, 1.2]
		randomize_Kp_factor = True
		Kp_factor_range = [0.7, 1.3]
		randomize_Kd_factor = True
		Kd_factor_range = [0.7, 1.3]
		shifted_com_range_x = [-0.2, 0.2]
		shifted_com_range_y = [-0.1, 0.1]
		shifted_com_range_z = [-0.1, 0.1]

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
	class algorithm( LeggedRobotCfgPPO.algorithm ):
		entropy_coef = 0.01
	class runner( LeggedRobotCfgPPO.runner ):
		run_name = ''
		experiment_name = 'rough_a1'
		policy_class_name = "MultiCriticActorCritic"  #'ActorCritic'
		algorithm_class_name = "MultiCriticPPO"     # 'PPO'
		critic_num = reward_group_num

  