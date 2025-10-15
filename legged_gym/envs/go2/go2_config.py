from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import yaml

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	control_type = config['control_type']
	action_scale = config['action_scale']
	decimation = config['control_decimation']
	num_envs = config['num_envs']
	multi_critic = config['multi_critic']
	reward_group_num = config['reward_group_num']
	rough_terrain = config['rough_terrain']
	episode_length = config['episode_length']
	fine_tune = config['fine_tune']
	number_observations = config['number_observations']
	number_privileged_obs = config['number_privileged_observations']
	measure_heights = config['measure_heights']
	use_one_critic_ablation = config['use_one_critic_ablation']
	use_imitation_commands = config['use_imitation_commands']

class GO2FlatCfg( LeggedRobotCfg ):
	class env( LeggedRobotCfg.env ):
		num_observations = number_observations
		num_privileged_obs = number_privileged_obs
		num_envs = num_envs
		episode_length_s = episode_length # episode length in seconds

	class init_state( LeggedRobotCfg.init_state ):
		if control_type == 'apex_torque':
			pos = [0.0, 0.0, 0.33] # x,y,z [m]
		else:
			pos = [0.0, 0.0, 0.35] # x,y,z [m]
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
		control_type = control_type
		action_scale = action_scale		
		exp_avg_decay = False
		limit_dof_pos = False
		stiffness = {'joint': 20}  # [N*m/rad]
		damping = {'joint': 0.5}     # [N*m*s/rad]
		# decimation: Number of control action updates @ sim DT per policy DT
		decimation = decimation

	class terrain:
		if rough_terrain:
			mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
			curriculum = True
		else:
			mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
			curriculum = False
		horizontal_scale = 0.1 # [m]
		vertical_scale = 0.005 # [m]
		border_size = 25 # [m]
		static_friction = 1.0
		dynamic_friction = 1.0
		restitution = 0.
		# rough terrain only:
		if rough_terrain:
			measure_heights = measure_heights
			measure_heights_priv = True
		else:
			measure_heights = measure_heights
			measure_heights_priv = False
		measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
		measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
		selected = False # select a unique terrain type and pass all arguments
		terrain_kwargs = None # Dict of arguments for selected terrain
		max_init_terrain_level = 5 # starting curriculum state
		terrain_length = 8.
		terrain_width = 8.
		num_rows= 10 # number of terrain rows (levels)
		num_cols = 20 # number of terrain cols (types)
		# terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
		terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
		# trimesh only:
		slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

	class asset( LeggedRobotCfg.asset ):
		file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
		name = "go2"
		foot_name = "foot"
		penalize_contacts_on = ["base", "hip","thigh", "calf","trunk"]
		# terminate_after_contacts_on = ["base", "hip","thigh","trunk"]
		terminate_after_contacts_on = ["base", "hip"]
		self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
	class rewards( LeggedRobotCfg.rewards ):
		soft_dof_pos_limit = 0.9
		base_height_target = 0.28
		only_positive_rewards = False
		sigma_rew_neg = 0.02
		only_positive_rewards_ji22_style = False
		multi_critic = multi_critic
		reward_group_num = reward_group_num
		tracking_sigma = 0.25 # for lin and ang vel
		class scales( LeggedRobotCfg.rewards.scales ):
			if multi_critic and not use_one_critic_ablation:
				class group1:
					# imitation_height_penalty = -30.0
					imitation_angles = 3.5#3.0
					imitate_end_effector_pos = 2.5#2.0#2.5
					imitate_joint_vel = 0.0
					imitate_foot_height = 0.0
					imitation_height = 0.0
					imitate_quat = 0.5#0.0

					imitate_base_end_effector_pos_world = 0.0

				class group2:
					termination = -0.0#0.0
					tracking_lin_vel = 2.0 #2.0
					tracking_ang_vel = 1.5#1.5
					lin_vel_z = 0.0#-1.0
					orientation = -0.0#-0.0
					# imitate_quat_penalty = -3.0
					torques = -0.00001#-0.00001
					# dof_vel = -0.
					dof_acc = -2.5e-7#2.5e-7
					base_height = -0. 
					feet_air_time =  0.0#1.0
					collision = -1.
					action_rate = -0.01#-0.01
					stand_still = -0. 

					feet_slip = -0.04#-0.04
					#Extra rewards to play with
					raibert_heuristic = 0.0
					tracking_contacts_shaped_force = 0.0
					tracking_contacts_shaped_vel = 0.0
					feet_clearance_cmd_linear = -0.0

					imitate_quat = 0.0#1.5#0.5
					energy = -0.0 #-0.001

					if rough_terrain:
						imitation_height_penalty = -0.0
						stumble = -0.2
						ang_vel_xy = -0.02#-0.05
						straight_heading = -0.0


					else:
						imitation_height_penalty = -30.0
						stumble = 0.0 
						ang_vel_xy = -0.05#-0.05

			elif multi_critic and use_one_critic_ablation:
				class group1:
					imitation_angles = 3.5
					imitate_end_effector_pos = 2.0#2.5

					tracking_lin_vel = 2.0
					tracking_ang_vel = 1.5
					# imitate_quat_penalty = -3.0
					torques = -0.00001#-0.00001
					dof_acc = -2.5e-7#2.5e-7
					collision = -1.
					action_rate = -0.01#-0.01
					feet_slip = -0.04#-0.04
					imitate_quat = 0.5#1.5#0.5

					if rough_terrain:
						imitation_height_penalty = -0.0
						stumble = -0.2
						ang_vel_xy = -0.02#-0.05
						straight_heading = -0.0

					else:
						imitation_height_penalty = -30.0
						stumble = 0.0 
						ang_vel_xy = -0.05#-0.05

				class group2:
					termination = -0.0#0.0


			else:
				imitation_angles = 2.0#3.0
				imitate_end_effector_pos = 3.5#2.5

				tracking_lin_vel = 2.0 #2.0
				tracking_ang_vel = 1.5#1.5
				# imitate_quat_penalty = -3.0
				torques = -0.00001#-0.00001
				dof_acc = -2.5e-7#2.5e-7
				collision = -1.
				action_rate = -0.01#-0.01
				feet_slip = -0.04#-0.04
				imitate_quat = 0.5#1.5#0.5

				if rough_terrain:
					imitation_height_penalty = -0.0
					stumble = -0.2
					ang_vel_xy = -0.02#-0.05
					straight_heading = -0.0

				else:
					imitation_height_penalty = -30.0
					stumble = 0.0 
					ang_vel_xy = -0.05#-0.05

				#Extra rewards to play with
				raibert_heuristic = 0.0
				tracking_contacts_shaped_force = 0.0
				tracking_contacts_shaped_vel = 0.0
				feet_clearance_cmd_linear = -0.0

				energy = -0.0 #-0.001

	class domain_rand:
		randomize_friction = True
		friction_range = [0.3, 1.25]
		randomize_base_mass = True
		added_mass_range = [-1., 1.]
		if not rough_terrain:
			push_robots = True
			push_interval_s = 4 #5
			max_push_vel_xy = 0.4 #1.5
			max_push_vel_ang = 0.6 #1.2
		else:
			push_robots = False
			push_interval_s = 5
			max_push_vel_xy = 0.5
			max_push_vel_ang = 0.2
		randomize_motor_offset = True
		motor_offset_range = [-0.035, 0.035]
		randomize_motor_strength = False
		motor_strength_range = [0.8, 1.2]
		randomize_Kp_factor = True
		Kp_factor_range = [0.9, 1.1]
		randomize_Kd_factor = True
		Kd_factor_range = [0.9, 1.1]
		shifted_com_range_x = [-0.2, 0.2]
		shifted_com_range_y = [-0.1, 0.1]
		shifted_com_range_z = [-0.1, 0.1]
		
		randomize_link_mass = True
		multiplied_link_mass_range = [0.9, 1.1]

		randomize_lag_timesteps = False
		lag_timesteps = 2

	class termination:
		# Enable imitation-based termination conditions
		enable_imitation_termination = False
		
		# Simple curriculum settings
		use_curriculum = True                        # Enable curriculum learning for termination
		curriculum_start_iteration = 200            # When to start tightening (more initial learning time)
		curriculum_end_iteration = 800              # When curriculum reaches final values (shorter curriculum)
		curriculum_type = "linear"                  
		
		# Initial termination thresholds
		initial_max_joint_angle_error = 2.0         # radians
		initial_max_height_error = 0.5              # meters
		initial_max_orientation_error = 0.7         # quaternion distance
		initial_max_end_effector_error = 1.0        # meters

		# Final termination thresholds
		final_max_joint_angle_error = 0.8           # radians
		final_max_height_error = 0.15               # meters
		final_max_orientation_error = 0.3           # quaternion distance
		final_max_end_effector_error = 0.5          # meters

		# Fallback static thresholds (used when curriculum is disabled)
		max_joint_angle_error = 1.0                 # radians
		max_height_error = 0.25                     # meters
		max_orientation_error = 0.5                 # quaternion distance
		max_end_effector_error = 0.6                # meters

		# Optional: log when environments terminate due to imitation deviation
		log_termination_reasons = False

	class commands(LeggedRobotCfg.commands):
		# curriculum = False
		resampling_time = 5  # time before command are changed[s]
		use_heading = False
		heading_command = False
		use_imitation_commands = use_imitation_commands
		class ranges:
			lin_vel_x = [0.2, 2.0]  # min max [m/s]
			lin_vel_y = [-0.0, 0.0]  # min max [m/s]
			ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
			heading = [-0.0, 0.0]
		
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

	class noise:
		add_noise = True
		noise_level = 1.0 # scales other values
		class noise_scales:
			dof_pos = 0.02
			dof_vel = 1.5
			lin_vel = 0.1
			ang_vel = 0.2
			gravity = 0.1
			height_measurements = 0.1

		
class GO2FlatCfgPPO( LeggedRobotCfgPPO ):
	seed = 39
	class algorithm( LeggedRobotCfgPPO.algorithm ):
		if fine_tune:
			entropy_coef = 0.0
			learning_rate = 1e-4
		else:
			entropy_coef = 0.01
	class runner( LeggedRobotCfgPPO.runner ):
		run_name = ''
		experiment_name = 'apex_go2_flat'
		max_iterations = 1000
		save_interval = 200 # check for potential saves every this many iterations
		policy_class_name = "MultiCriticActorCritic"  #'ActorCritic'
		algorithm_class_name = "MultiCriticPPO"     # 'PPO'
		critic_num = reward_group_num
