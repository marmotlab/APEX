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

class H1RoughCfg( LeggedRobotCfg ):

	class init_state( LeggedRobotCfg.init_state ):
		pos = [0.0, 0.0, 1.0] # x,y,z [m]
		default_joint_angles = { # = target angles [rad] when action = 0.0
		   'left_hip_yaw_joint' : 0. ,   
		   'left_hip_roll_joint' : 0,               
		   'left_hip_pitch_joint' : -0.1,         
		   'left_knee_joint' : 0.3,       
		   'left_ankle_joint' : -0.2,     
		   'right_hip_yaw_joint' : 0., 
		   'right_hip_roll_joint' : 0, 
		   'right_hip_pitch_joint' : -0.1,                                       
		   'right_knee_joint' : 0.3,                                             
		   'right_ankle_joint' : -0.2,                                     
		   'torso_joint' : 0., 
		   'left_shoulder_pitch_joint' : 0., 
		   'left_shoulder_roll_joint' : 0, 
		   'left_shoulder_yaw_joint' : 0.,
		   'left_elbow_joint'  : 0.,
		   'right_shoulder_pitch_joint' : 0.,
		   'right_shoulder_roll_joint' : 0.0,
		   'right_shoulder_yaw_joint' : 0.,
		   'right_elbow_joint' : 0.,
		}

	class commands(LeggedRobotCfg.commands):
		curriculum = False
		resampling_time = 10.  # time before command are changed[s]
		use_heading = False
		heading_command = False
		use_imitation_commands = False
		class ranges:
			lin_vel_x = [0.4, 0.8] # min max [m/s]
			lin_vel_y = [0.0, 0.0]   # min max [m/s]
			ang_vel_yaw = [0, 0]    # min max [rad/s]
			heading = [0, 0]
	
	class env(LeggedRobotCfg.env):
		# 3 + 3 + 3 + 10 + 10 + 10 + 2 = 41
		num_observations = 41
		num_privileged_obs = 44
		num_actions = 10
		num_envs = num_envs

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


	class domain_rand(LeggedRobotCfg.domain_rand):
		randomize_friction = True
		friction_range = [0.1, 1.25]
		randomize_base_mass = True
		added_mass_range = [-1., 1.]
		push_robots = True
		push_interval_s = 5
		max_push_vel_xy = 1.5
		max_push_vel_ang = 0.2
		randomize_lag_timesteps = False
		lag_timesteps = 6
		randomize_motor_offset = False
		motor_offset_range = [-0.03, 0.03]
		randomize_motor_strength = False
		motor_strength_range = [0.8, 1.2]
		randomize_Kp_factor = False
		Kp_factor_range = [0.7, 1.3]
		randomize_Kd_factor = False
		Kd_factor_range = [0.5, 1.5]
		shifted_com_range_x = [-0.0, 0.0]
		shifted_com_range_y = [-0.0, 0.0]
		shifted_com_range_z = [-0.0, 0.0]

	class control( LeggedRobotCfg.control ):
		# PD Drive parameters:
		control_type = control_type
		  # PD Drive parameters:
		stiffness = {'hip_yaw': 150,
					 'hip_roll': 150,
					 'hip_pitch': 150,
					 'knee': 200,
					 'ankle': 40,
					 'torso': 300,
					 'shoulder': 150,
					 "elbow":100,
					 }  # [N*m/rad]
		damping = {  'hip_yaw': 2,
					 'hip_roll': 2,
					 'hip_pitch': 2,
					 'knee': 4,
					 'ankle': 2,
					 'torso': 6,
					 'shoulder': 2,
					 "elbow":2,
					 }  # [N*m/rad]  # [N*m*s/rad]
		# action scale: target angle = actionScale * action + defaultAngle
		action_scale = action_scale
		# decimation: Number of control action updates @ sim DT per policy DT
		decimation = decimation

	class asset( LeggedRobotCfg.asset ):
		file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
		name = "h1"
		foot_name = "ankle"
		penalize_contacts_on = ["hip", "knee"]
		terminate_after_contacts_on = ["pelvis"]
		self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
		flip_visual_attachments = False
  
	class rewards( LeggedRobotCfg.rewards ):
		soft_dof_pos_limit = 0.9
		base_height_target = 1.05
		multi_critic = multi_critic
		reward_group_num = reward_group_num
		class scales( LeggedRobotCfg.rewards.scales ):
			if multi_critic:				
				class group1:
					imitation_height_penalty = -30.0
					imitation_angles = 3.0
					imitate_end_effector_pos = 3.0
					imitate_joint_vel = 0.0
					imitate_foot_height = 0.0
					# imitate_quat = 0.5

					imitate_base_end_effector_pos_world = 0.0
				class group2:
					tracking_lin_vel = 2.0
					tracking_ang_vel = 1.5
					lin_vel_z = -2.0
					ang_vel_xy = -0.05
					orientation = -1.0
					base_height = 0.0
					dof_acc = -2.5e-7
					feet_air_time = 0.0
					collision = -1.0
					action_rate = -0.01
					torques = 0.0
					dof_pos_limits = -5.0
					alive = 0.15
					hip_pos = -0.0
					contact_no_vel = 0.0
					feet_swing_height = -20.0
					contact_no_vel = -0.2
					feet_slip = -0.0
					
					#Imitation Rewards
					# imitation_angles = 15.0
					# imitate_end_effector_pos = 15.0
					# imitate_foot_height = 15.0
					# imitation_height_penalty = -40.0

			else:
				tracking_lin_vel = 1.0
				tracking_ang_vel = 0.5
				lin_vel_z = -2.0
				ang_vel_xy = -0.05
				orientation = -1.0
				base_height = 0.0
				dof_acc = -2.5e-7
				feet_air_time = 0.0
				collision = -1.0
				action_rate = -0.01
				torques = 0.0
				dof_pos_limits = -5.0
				alive = 0.15
				hip_pos = -0.0
				contact_no_vel = 0.0
				feet_swing_height = 0.0
				contact = 0.0
					
				#Imitation Rewards
				imitation_angles = 15.0
				imitate_end_effector_pos = 15.0
				imitate_foot_height = 15.0
				imitation_height_penalty = -40.0

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
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
		# policy_class_name = "ActorCriticRecurrent"
		policy_class_name = "MultiCriticActorCritic"
		algorithm_class_name = "MultiCriticPPO"     # 'PPO'
		critic_num = reward_group_num
		max_iterations = 1000
		run_name = ''
		experiment_name = 'h1'

  
