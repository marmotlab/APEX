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

class G1RoughCfg( LeggedRobotCfg ):
	class init_state( LeggedRobotCfg.init_state ):
		pos = [0.0, 0.0, 0.8] # x,y,z [m]
		rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
		default_joint_angles = { # = target angles [rad] when action = 0.0
				'left_hip_pitch_joint': -0.1,
				'left_hip_roll_joint': 0. ,
				'left_hip_yaw_joint': 0. ,
				'left_knee_joint': 0.3 ,
				'left_ankle_pitch_joint': -0.2 , 
				'left_ankle_roll_joint': 0. ,
				'right_hip_pitch_joint': -0.1 ,
				'right_hip_roll_joint': 0. ,
				'right_hip_yaw_joint': 0. ,
				'right_knee_joint': 0.3 ,
				'right_ankle_pitch_joint': -0.2 ,
				'right_ankle_roll_joint': 0. ,
				'waist_yaw_joint': 0. ,
				'waist_roll_joint': 0. ,
				'waist_pitch_joint': 0. ,
				'left_shoulder_pitch_joint': 0. ,
				'left_shoulder_roll_joint': 0. ,
				'left_shoulder_yaw_joint': 0. ,
				'left_elbow_joint': 0. ,
				'right_shoulder_pitch_joint': 0. ,
				'right_shoulder_roll_joint': 0. ,
				'right_shoulder_yaw_joint': 0. ,
				'right_elbow_joint': 0.
		}
	
	class env(LeggedRobotCfg.env):
		num_observations = 9+23+23+23
		num_privileged_obs = 9+23+23+23+3
		num_actions = 23


	class domain_rand:
		randomize_friction = True
		friction_range = [0.3, 1.25]
		randomize_base_mass = False
		added_mass_range = [-1., 2.]
		# if not rough_terrain:
		push_robots = True
		push_interval_s = 5
		max_push_vel_xy = 1.0
		max_push_vel_ang = 0.0
		randomize_lag_timesteps = False
		lag_timesteps = 2
		randomize_motor_offset = False
		motor_offset_range = [-0.03, 0.03]
		randomize_motor_strength = False
		motor_strength_range = [0.8, 1.2]
		randomize_Kp_factor = False
		Kp_factor_range = [0.7, 1.3]
		randomize_Kd_factor = False
		Kd_factor_range = [0.7, 1.3]
		shifted_com_range_x = [-0.2, 0.2]
		shifted_com_range_y = [-0.1, 0.1]
		shifted_com_range_z = [-0.1, 0.1]
	  

	class control( LeggedRobotCfg.control ):
		# PD Drive parameters:
		control_type = control_type
		  # PD Drive parameters:
		stiffness = {'hip_yaw': 100,
			 'hip_roll': 100,
			 'hip_pitch': 100,
			 'knee': 200,
			 'ankle_pitch': 20,
			 'ankle_roll': 20,
			 'waist_yaw': 400,
			 'waist_roll': 400,
			 'waist_pitch': 400,
			 'shoulder_pitch': 90,
			 'shoulder_roll': 60,
			 'shoulder_yaw': 20,
			 'elbow': 60}  # [N*m/rad]

		damping = {'hip_yaw': 2.5,
		   'hip_roll': 2.5,
		   'hip_pitch': 2.5,
		   'knee': 5.0,
		   'ankle_pitch': 0.2,
		   'ankle_roll': 0.1,
		   'waist_yaw': 5.0,
		   'waist_roll': 5.0,
		   'waist_pitch': 5.0,
		   'shoulder_pitch': 2.0,
		   'shoulder_roll': 1.0,
		   'shoulder_yaw': 0.4,
		   'elbow': 1.0}  # [N*m/rad]  # [N*m*s/rad]

		# action scale: target angle = actionScale * action + defaultAngle
		action_scale = 0.25
		# decimation: Number of control action updates @ sim DT per policy DT
		decimation = 4

	class asset( LeggedRobotCfg.asset ):
		file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof.urdf'
		name = "g1"
		foot_name = "ankle_roll_link"
		penalize_contacts_on = ["pelvis", "shoulder", "hip"]
		terminate_after_contacts_on = ["pelvis", "shoulder"]
		# terminate_after_contacts_on = []
		self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
		flip_visual_attachments = False
  
	class rewards( LeggedRobotCfg.rewards ):
		soft_dof_pos_limit = 0.9
		base_height_target = 0.28
		only_positive_rewards = False
		sigma_rew_neg = 0.02
		only_positive_rewards_ji22_style = False
		multi_critic = multi_critic
		reward_group_num = reward_group_num
		class scales( LeggedRobotCfg.rewards.scales ):
			#Imitation rewards 
			if multi_critic:
				class group1:
					imitation_height_penalty = -30.0
					imitation_angles = 3.0
					imitate_end_effector_pos = 0.00001
					imitate_joint_vel = 0.0
					imitate_foot_height = 0.0
					# imitate_quat = 0.5

					imitate_base_end_effector_pos_world = 0.0

				class group2:
					tracking_lin_vel = 0.0
					tracking_ang_vel = 0.0
					lin_vel_z = -2.0
					ang_vel_xy = -0.05
					orientation = -1.0
					base_height = -10.0
					dof_acc = -2.5e-7
					dof_vel = -1e-3
					feet_air_time = 0.0
					collision = 0.0
					action_rate = -0.01
					dof_pos_limits = -5.0
					alive = 0.15
					hip_pos = -1.0
					contact_no_vel = -0.2
					feet_swing_height = -20.0
					contact = 0.18
			else:
				#Imitation rewards 
				imitation_height_penalty = -30.0

				imitation_angles = 2.0
				imitate_end_effector_pos = 3.0
				imitate_foot_height = 0

				imitate_base_end_effector_pos_world = 0.0

				# penalty_foot_dragging = -0.0
				# min_clearance = 0.0
				# target_clearance = 0.0
				# termination = -0.0
				# lin_vel_z = -1.0
				# ang_vel_xy = -0.0
				# orientation = -0.
				# dof_vel = -0.
				# dof_acc = -0.0
				# base_height = -0. 
				# feet_air_time =  0.0
				# collision = -0.
				# feet_stumble = -0.0 
				# action_rate = -0.01
				# orientation_penalty = -0.0
				# torques =  -0.0001
				tracking_lin_vel = 1.0
				tracking_ang_vel = 0.9

				# feet_slip = -0.04   

				#Extra rewards to play with
				raibert_heuristic = 0.0
				tracking_contacts_shaped_force = 0.0
				tracking_contacts_shaped_vel = 0.0
				feet_clearance_cmd_linear = -0.0

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
		measure_heights = False
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
		# terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
		terrain_proportions = [0.3, 0.3, 0.4, 0.0, 0.0]
		# trimesh only:
		slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
	class policy:
		init_noise_std = 0.8
		actor_hidden_dims = [32]
		critic_hidden_dims = [32]
		activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
		# only for 'ActorCriticRecurrent':
		rnn_type = 'lstm'
		rnn_hidden_size = 64
		rnn_num_layers = 1
		
	class algorithm( LeggedRobotCfgPPO.algorithm ):
		entropy_coef = 0.01
	class runner( LeggedRobotCfgPPO.runner ):
		policy_class_name = "MultiCriticActorCritic"
		algorithm_class_name = "MultiCriticPPO"     # 'PPO'
		critic_num = reward_group_num
		max_iterations = 10000
		run_name = ''
		experiment_name = 'g1'

  
