from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import yaml

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	control_type = config['control_type']
	action_scale = config['action_scale']
	decimation = config['control_decimation']
	num_envs = config['num_envs']

class H1_2Cfg( LeggedRobotCfg ):

	class init_state( LeggedRobotCfg.init_state ):
		pos = [0.0, 0.0, 1.0] # x,y,z [m]
		rot = [0.0, 0.0, 0.0, 1.0]
		# rot = [0.0, 0.0, 1.0, 0.0]
		default_joint_angles = { # = target angles [rad] when action = 0.0
		   'left_hip_yaw_joint' : 0 ,   
		   'left_hip_roll_joint' : 0,               
		   'left_hip_pitch_joint' : -0.3, # -0.32,         
		   'left_knee_joint' : 0.6, # 0.5,       
		   'left_ankle_pitch_joint' : -0.3, # -0.18,  
		   'left_ankle_roll_joint' : 0,   
		   'right_hip_yaw_joint' : -0, 
		   'right_hip_roll_joint' : 0, 
		   'right_hip_pitch_joint' : -0.3, # -0.32,                                       
		   'right_knee_joint' : 0.6, # 0.5,                                             
		   'right_ankle_pitch_joint' : -0.3, # -0.18,   
		   'right_ankle_roll_joint' : 0,                                  
		   'torso_joint' : 0., 
		   'left_shoulder_pitch_joint' : 0., 
		   'left_shoulder_roll_joint' : 0, 
		   'left_shoulder_yaw_joint' : 0.,
		   'left_elbow_pitch_joint'  : 0.,
		   'left_elbow_roll_joint' : 0.,
		   'left_wrist_pitch_joint'  : 0.,
		   'left_wrist_yaw_joint' : 0.,
		   'right_shoulder_pitch_joint' : 0.,
		   'right_shoulder_roll_joint' : 0.0,
		   'right_shoulder_yaw_joint' : 0.,
		   'right_elbow_pitch_joint' : 0.,
		   'right_elbow_roll_joint' : 0.,
		   'right_wrist_pitch_joint'  : 0.,
		   'right_wrist_yaw_joint' : 0.,
		}

	class commands(LeggedRobotCfg.commands):
		curriculum = False
		resampling_time = 10.  # time before command are changed[s]
		use_heading = False
		heading_command = False
		use_imitation_commands = False
		class ranges:
			lin_vel_x = [0.0, 0.0] # min max [m/s]
			lin_vel_y = [0.0, 0.0]   # min max [m/s]
			ang_vel_yaw = [0, 0]    # min max [rad/s]
			heading = [0, 0]
	
	class env(LeggedRobotCfg.env):
		# 3 + 3 + 3 + 10 + 10 + 10 + 2 = 41
		#Change observations
		num_observations = 41 + 17 + 17 +17
		num_privileged_obs = 44 + 17 + 17 + 17
		num_actions = 27
		num_envs = num_envs
		episode_length_s = 20 # episode length in seconds

	class terrain:
		mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
		horizontal_scale = 0.1 # [m]
		vertical_scale = 0.005 # [m]
		border_size = 25 # [m]
		curriculum = True
		static_friction = 1.0
		dynamic_friction = 1.0
		restitution = 0.
		# rough terrain only:
		measure_heights = True
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
		terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
		# trimesh only:
		slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
	  

	class sim( LeggedRobotCfg.sim ):
		dt = 0.005
		
	class domain_rand(LeggedRobotCfg.domain_rand):
		randomize_friction = True
		friction_range = [0.1, 1.25]
		randomize_base_mass = True
		added_mass_range = [-1., 1.]
		push_robots = False
		push_interval_s = 15
		max_push_vel_xy = 1.0
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

	class control( LeggedRobotCfg.control ):
		# PD Drive parameters:
		control_type = control_type
		stiffness = {
			'hip_yaw_joint': 200.,
			'hip_roll_joint': 200.,
			'hip_pitch_joint': 200.,
			'knee_joint': 300.,
			'ankle_pitch_joint': 60.,
			'ankle_roll_joint': 40.,
			'torso_joint': 600.,
			'shoulder_pitch_joint': 80.,
			'shoulder_roll_joint': 80.,
			'shoulder_yaw_joint': 40.,
			'elbow_pitch_joint': 60.,
		}  # [N*m/rad]
		damping = {
			'hip_yaw_joint': 5.0,
			'hip_roll_joint': 5.0,
			'hip_pitch_joint': 5.0,
			'knee_joint': 7.5,
			'ankle_pitch_joint': 1.0,
			'ankle_roll_joint': 0.3,
			'torso_joint': 15.0,
			'shoulder_pitch_joint': 2.0,
			'shoulder_roll_joint': 2.0,
			'shoulder_yaw_joint': 1.0,
			'elbow_pitch_joint': 1.0,
		}  # [N*m/rad]  # [N*m*s/rad]
		'''
		stiffness = {'hip_yaw': 200,
					 'hip_roll': 200,
					 'hip_pitch': 200,
					 'knee': 200,
					 'ankle_pitch': 140,
					 'ankle_roll': 200,
					 'torso': 200,
					 'shoulder': 140,
					 'elbow_pitch': 140,
					 'elbow_roll': 35,
					 'wrist': 35,
					 }  # [N*m/rad]
		damping = {  'hip_yaw': 5,
					 'hip_roll': 5,
					 'hip_pitch': 5,
					 'knee': 5,
					 'ankle_pitch': 7.5,
					 'ankle_roll': 5,
					 'torso': 5,
					 'shoulder': 7.5,
					 'elbow_pitch': 7.5,
					 'elbow_roll': 6,
					 'wrist': 6,
					 }  # [N*m/rad]  # [N*m*s/rad]
		'''
		# action scale: target angle = actionScale * action + defaultAngle
		action_scale = action_scale
		# decimation: Number of control action updates @ sim DT per policy DT
		decimation = decimation

	class asset( LeggedRobotCfg.asset ):
		file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_handless.urdf'
		name = "h1_2"
		torso_name = "torso_link"
		foot_name = "ankle_roll"
		penalize_contacts_on = ["shoulder", "elbow", "hip"]
		terminate_after_contacts_on = ["torso_link", "shoulder", "elbow", "hip"]
		self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
		flip_visual_attachments = False
  
	class rewards( LeggedRobotCfg.rewards ):
		soft_dof_pos_limit = 0.9
		base_height_target = 1.0
		# only_positive_rewards = False
		# only_positive_rewards_ji22_style = False
		class scales( LeggedRobotCfg.rewards.scales ):
			tracking_lin_vel = 1.0
			tracking_ang_vel = 0.5
			lin_vel_z = -2.0
			ang_vel_xy = -0.05
			orientation = -1.0
			base_height = -10.0
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
			imitate_end_effector_pos = 0.0
			imitate_foot_height = 0.0
			imitation_height_penalty = 0.0


class H1_2CfgPPO( LeggedRobotCfgPPO ):
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
		policy_class_name = "ActorCriticRecurrent"
		max_iterations = 1000
		run_name = ''
		experiment_name = 'h1_2'

  
