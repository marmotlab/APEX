from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.helpers import class_to_dict
import pandas as pd
from legged_gym.utils.math import quat_apply_yaw, exp_avg_filter
import yaml

from legged_gym.utils.calculate_raibert_gains import load_effort_limits_from_urdf, build_pd_gains_for_go2, build_alpha_tensor
from legged_gym import LEGGED_GYM_ROOT_DIR, envs

def parallel_axis_theorem(I_com, mass, d):
	"""
	Compute the inertia matrix after shifting the centroid using the parallel axis theorem.

	Parameters:
	I_com (np.array): Inertia matrix at the centroid (3x3)  
	mass (float): Mass of the object  
	d (np.array): Displacement of the centroid relative to the new origin (3,)

	Returns:
	np.array: Inertia matrix after shifting the centroid (3x3)
	"""
	d_x, d_y, d_z = d
	d_squared = np.array([
		[d_y ** 2 + d_z ** 2, -d_x * d_y, -d_x * d_z],
		[-d_x * d_y, d_x ** 2 + d_z ** 2, -d_y * d_z],
		[-d_x * d_z, -d_y * d_z, d_x ** 2 + d_y ** 2]
	])

	# Compute the new inertia matrix using the parallel axis theorem
	return I_com + mass * d_squared


def update_inertia(I_box, mass_box, com_box, mass_point, point_pos):
	"""
	Update the inertia matrix and compute the new inertia matrix after adding a point mass.

	Parameters:
	I_box (np.array): Inertia matrix of the original cuboid (3x3)  
	mass_box (float): Mass of the original cuboid  
	com_box (np.array): Centroid position of the original cuboid (3,)  
	mass_point (float): Mass of the point mass  
	point_pos (np.array): Position of the point mass (3,)

	Returns:
	I_total (np.array): New inertia matrix (3x3)  
	new_com (np.array): New centroid position (3,)
	"""
	# Update centroid
	new_com = update_com(mass_box, com_box, mass_point, point_pos)

	# Inertia matrix of the cuboid after shifting
	displacement_box = com_box - new_com  #  Compute displacement of the cuboid centroid relative to the new centroid
	I_box_new = parallel_axis_theorem(I_box, mass_box, displacement_box)

	# Inertia matrix of the point mass
	displacement_point = point_pos - new_com  # Compute displacement of the point mass relative to the new centroid
	I_point = parallel_axis_theorem(np.zeros((3, 3)), mass_point, displacement_point)

	# Combine the inertia matrices of the cuboid and the point mass
	I_total = I_box_new + I_point

	return I_total, new_com

def update_com(mass_box, com_box, mass_point, point_pos):
	#Thanks Hongyi!
	"""
	Update the centroid position and compute the new centroid coordinates.

	Parameters:
	I_box (np.array): Inertia matrix of the original cuboid (3x3)
	mass_box (float): Mass of the original cuboid
	com_box (np.array): Centroid position of the original cuboid (3,)
	mass_point (float): Mass of the point mass
	point_pos (np.array): Position of the point mass (3,)

	Returns:
	new_com (np.array): New centroid position
	"""
	# Formula for the new centroid position
	new_com = (mass_box * com_box + mass_point * point_pos) / (mass_box + mass_point)
	return new_com


class Go2(LeggedRobot):

	def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
		"""Initialize Go2 environment with preprocessed imitation data"""
		# Load config parameters
		with open(f"legged_gym/envs/param_config.yaml", "r") as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
			self.gamma_decap = config["gamma"]
			self.k_decap = config["k"]
			self.visualize_imitation_data = config["visualize_imitation_data"]
			self.path_to_imitation_data = config["path_to_imitation_data"]
			self.number_observations = config["number_observations"]
			self.number_privileged_observations = config["number_privileged_observations"]
			self.reference_state_init = config['reference_state_init']
			self.decap_type = config["decap_type"]
			self.cosine_constant_prior_iterations = config["constant_prior_iterations"]
			self.cosine_decay_iterations = config["cosine_decay_iterations"]
			self.sigma_imit_angles = config["sigma_imit_angles"]
			self.sigma_imit_foot_pos = config["sigma_imit_foot_pos"]
			self.train_multi_skills = config["train_multi_skills"]
		
		# Call parent constructor
		super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
		# self._init_joint_impedance_and_scaling()
		self._preprocess_imitation_data()
	
	def _init_joint_impedance_and_scaling(self):
		"""
		- Parse URDF effort limits (tau_max per joint).
		- Build per-joint PD gains (low-gain impedance) from (I, wn, zeta).
		- Build per-DOF action scales alpha_j to keep P-term torque <= ~0.25*tau_max.
		Overwrites self.p_gains / self.d_gains (tensors) used in _compute_torques.
		"""
		# map URDF file path
		urdf_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
		tau_max = load_effort_limits_from_urdf(urdf_path)

		# Kp/Kd per joint name
		kp_dict, kd_dict = build_pd_gains_for_go2()

		# Build tensors aligned to sim DOF order
		p_list, d_list = [], []
		for name in self.dof_names:
			p_list.append(kp_dict.get(name, 0.0))
			d_list.append(kd_dict.get(name, 0.0))
		self.p_gains = torch.tensor(p_list, device=self.device).view(1, -1)  # [1,num_dof] for broadcasting
		self.d_gains = torch.tensor(d_list, device=self.device).view(1, -1)

		# 4) Per-DOF action scaling alpha (radians)
		self.action_alpha = build_alpha_tensor(self.dof_names, kp_dict, tau_max, device=self.device).view(1, -1)

		# Optional: log a quick sanity line
		if hasattr(self, "rank") and self.rank == 0:
			mins = float(self.action_alpha.min().cpu())
			maxs = float(self.action_alpha.max().cpu())
			print(f"[Go2] action_alpha in radians: min={mins:.4f}, max={maxs:.4f}")


	def _preprocess_imitation_data(self):
		# Convert entire dataframe to tensor on GPU
		self.df_imit_tensor = torch.from_numpy(self.df_imit.values).float().to(self.device)
		self.df_imit_length = len(self.df_imit)
		
		# Pre-slice commonly used columns for faster access
		self.imit_lin_vel = self.df_imit_tensor[:, 0:3]           # columns 0-2: linear velocity
		self.imit_ang_vel = self.df_imit_tensor[:, 3:6]           # columns 3-5: angular velocity
		self.imit_joint_pos = self.df_imit_tensor[:, 6:18]        # columns 6-17: joint positions
		self.imit_height = self.df_imit_tensor[:, 21:22]          # column 21: height
		self.imit_end_effector = self.df_imit_tensor[:, 22:34]    # columns 22-33: end effector pos
		self.imit_base_pos = self.df_imit_tensor[:, 34:36]        # columns 34-35: base position
		self.imit_quaternions = self.df_imit_tensor[:, 36:40]     # columns 36-39: quaternions
		self.imit_end_effector_world = self.df_imit_tensor[:, 40:52]  # columns 40-51: end effector world
		self.imit_joint_vel = self.df_imit_tensor[:, 52:64]       # columns 52-63: joint velocities
		
		# Pre-slice individual foot heights for convenience
		self.imit_foot_z1 = self.df_imit_tensor[:, 42:43]         # column 42: foot 1 z
		self.imit_foot_z2 = self.df_imit_tensor[:, 45:46]         # column 45: foot 2 z
		self.imit_foot_z3 = self.df_imit_tensor[:, 48:49]         # column 48: foot 3 z
		self.imit_foot_z4 = self.df_imit_tensor[:, 51:52]         # column 51: foot 4 z
		
		# Pre-slice leg-wise joint positions
		self.imit_joint_leg1 = self.df_imit_tensor[:, 6:9]        # columns 6-8: leg 1
		self.imit_joint_leg2 = self.df_imit_tensor[:, 9:12]       # columns 9-11: leg 2
		self.imit_joint_leg3 = self.df_imit_tensor[:, 12:15]      # columns 12-14: leg 3
		self.imit_joint_leg4 = self.df_imit_tensor[:, 15:18]      # columns 15-17: leg 4
		
		print(f"Preprocessed {self.df_imit_length} imitation data samples to GPU")

	def reset_idx(self, env_ids):
		""" Reset some environments.
			Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
			Logs episode info
			Resets some buffers

		Args:
			env_ids (list[int]): List of environment ids which must be reset
		"""
		if len(env_ids) == 0:
			return

		# update curriculum
		if self.cfg.terrain.curriculum:
			self._update_terrain_curriculum(env_ids)
		# avoid updating command curriculum at each step since the maximum command is common to all envs
		if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
			self.update_command_curriculum(env_ids)

		# reset robot states
		self._reset_dofs(env_ids)
		self._reset_root_states(env_ids)
		self._randomize_dof_props(env_ids, self.cfg)

		if self.cfg.commands.use_imitation_commands:
			self._sample_imitation_commands()
		
		else:
			self._resample_commands(env_ids)

		# reset buffers
		self.last_actions[env_ids] = 0.
		self.last_dof_vel[env_ids] = 0.
		self.feet_air_time[env_ids] = 0.
		self.episode_length_buf[env_ids] = 0
		self.reset_buf[env_ids] = 1
		# fill extras
		self.extras["episode"] = {}
		if getattr(self.cfg.rewards, "multi_critic", False):
			# For each reward group, update each reward's episode sum.
			for group_name, rewards_dict in self.episode_sums.items():
				for reward_name, tensor in rewards_dict.items():
					key = f"{group_name}_{reward_name}"
					self.extras["episode"]['rew_' + key] = torch.mean(tensor[env_ids]) / self.max_episode_length_s
					tensor[env_ids] = 0.
		else:
			# Single-group mode remains unchanged.
			for key in self.episode_sums.keys():
				self.extras["episode"]['rew_' + key] = torch.mean(
					self.episode_sums[key][env_ids]) / self.max_episode_length_s
				self.episode_sums[key][env_ids] = 0.
		
		# send timeout info to the algorithm
		if self.cfg.env.send_timeouts:
			self.extras["time_outs"] = self.time_out_buf
		if self.cfg.domain_rand.randomize_lag_timesteps:
			for i in range(len(self.lag_buffer)):
				self.lag_buffer[i][env_ids, :] = 0
		self.extras['decap_factor'] = float(self.decap_factor[0].item())

	def _reset_dofs(self, env_ids):
		""" Resets DOF position and velocities of selected environmments
		Positions are randomly selected within 0.5:1.5 x default positions.
		Velocities are set to zero.

		Args:
			env_ids (List[int]): Environemnt ids
		"""
		#Initialize the root states to the imitation index reference angles
		if not self.reference_state_init:
			self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
			# self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.0, 2.0, (len(env_ids), self.num_dof), device=self.device)
		else:
			#Convert to tensor - using preprocessed data
			self.reference_state_angles = self.imit_joint_pos[self.imitation_index.long()]
			self.dof_pos[env_ids] = self.reference_state_angles[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
		self.dof_vel[env_ids] = 0.

		env_ids_int32 = env_ids.to(dtype=torch.int32)
		self.gym.set_dof_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.dof_state),
											  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

	def _reset_root_states(self, env_ids):
		""" Resets ROOT states position and velocities of selected environmments
			Sets base position based on the curriculum
			Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
		Args:
			env_ids (List[int]): Environemnt ids
		"""
		# base position
		if self.custom_origins:
			self.root_states[env_ids] = self.base_init_state
			self.root_states[env_ids, :3] += self.env_origins[env_ids]
			self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
		else:
			self.root_states[env_ids] = self.base_init_state
			self.root_states[env_ids, :3] += self.env_origins[env_ids]
		# base velocities
		if self.reference_state_init:
			self.reference_velocity = self.imit_lin_vel[self.imitation_index.long()]
			self.root_states[env_ids, 7:13] = torch.cat([
				self.reference_velocity[env_ids], 
				self.imit_ang_vel[self.imitation_index.long()][env_ids]
			], dim=1)  # [7:10]: lin vel, [10:13]: ang vel
			self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
			self.reference_state_height = self.imit_height[self.imitation_index.long()].squeeze(-1)
			self.root_states[env_ids, 2] = self.reference_state_height[env_ids]  # [2]: height
			self.reference_quaternions = self.imit_quaternions[self.imitation_index.long()]
			self.root_states[env_ids, 3:7] = self.reference_quaternions[env_ids]  # [3:7]: quaternion

		else:	
			self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
		env_ids_int32 = env_ids.to(dtype=torch.int32)
		self.gym.set_actor_root_state_tensor_indexed(self.sim,
													 gymtorch.unwrap_tensor(self.root_states),
													 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

	def check_termination(self):
		""" Check if environments need to be reset, including imitation data thresholds
		"""
		# First call the base termination logic (contact and timeout)
		super().check_termination()
		
		# Add imitation-based termination conditions
		if hasattr(self, 'cfg') and hasattr(self.cfg, 'termination') and hasattr(self.cfg.termination, 'enable_imitation_termination'):
			if self.cfg.termination.enable_imitation_termination:
				self._check_imitation_termination()
	
	def _get_curriculum_termination_thresholds(self):
		""" Calculate current termination thresholds based on simple linear curriculum """
		if not hasattr(self.cfg.termination, 'use_curriculum') or not self.cfg.termination.use_curriculum:
			# Use static thresholds if curriculum is disabled
			return {
				'angle_threshold': getattr(self.cfg.termination, 'max_joint_angle_error', 1.0),
				'height_threshold': getattr(self.cfg.termination, 'max_height_error', 0.4),
				'orientation_threshold': getattr(self.cfg.termination, 'max_orientation_error', 0.7),
				'end_effector_threshold': getattr(self.cfg.termination, 'max_end_effector_error', 0.6)
			}
		
		# Get curriculum parameters
		current_iteration = getattr(self, 'global_training_iteration', 0)
		start_iter = getattr(self.cfg.termination, 'curriculum_start_iteration', 200)
		end_iter = getattr(self.cfg.termination, 'curriculum_end_iteration', 800)
		
		# Before curriculum starts, use initial (relaxed) values
		if current_iteration < start_iter:
			return {
				'angle_threshold': getattr(self.cfg.termination, 'initial_max_joint_angle_error', 2.0),
				'height_threshold': getattr(self.cfg.termination, 'initial_max_height_error', 0.8),
				'orientation_threshold': getattr(self.cfg.termination, 'initial_max_orientation_error', 1.0),
				'end_effector_threshold': getattr(self.cfg.termination, 'initial_max_end_effector_error', 1.0)
			}
		
		# After curriculum ends, use final (moderately strict) values
		if current_iteration >= end_iter:
			return {
				'angle_threshold': getattr(self.cfg.termination, 'final_max_joint_angle_error', 0.8),
				'height_threshold': getattr(self.cfg.termination, 'final_max_height_error', 0.3),
				'orientation_threshold': getattr(self.cfg.termination, 'final_max_orientation_error', 0.6),
				'end_effector_threshold': getattr(self.cfg.termination, 'final_max_end_effector_error', 0.5)
			}
		
		# During curriculum: simple linear interpolation
		progress = (current_iteration - start_iter) / (end_iter - start_iter)
		progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
		
		# Get initial and final values
		initial_angle = getattr(self.cfg.termination, 'initial_max_joint_angle_error', 2.0)
		final_angle = getattr(self.cfg.termination, 'final_max_joint_angle_error', 0.8)
		initial_height = getattr(self.cfg.termination, 'initial_max_height_error', 0.8)
		final_height = getattr(self.cfg.termination, 'final_max_height_error', 0.3)
		initial_orientation = getattr(self.cfg.termination, 'initial_max_orientation_error', 1.0)
		final_orientation = getattr(self.cfg.termination, 'final_max_orientation_error', 0.6)
		initial_end_effector = getattr(self.cfg.termination, 'initial_max_end_effector_error', 1.0)
		final_end_effector = getattr(self.cfg.termination, 'final_max_end_effector_error', 0.5)
		
		# Simple linear interpolation
		return {
			'angle_threshold': initial_angle + progress * (final_angle - initial_angle),
			'height_threshold': initial_height + progress * (final_height - initial_height),
			'orientation_threshold': initial_orientation + progress * (final_orientation - initial_orientation),
			'end_effector_threshold': initial_end_effector + progress * (final_end_effector - initial_end_effector)
		}
	
	def _check_imitation_termination(self):
		""" Check termination conditions based on deviation from imitation data """
		indices = self.imitation_index.long()
		termination_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
		
		# Get current termination thresholds (either from curriculum or static config)
		thresholds = self._get_curriculum_termination_thresholds()
		angle_threshold = thresholds['angle_threshold']
		height_threshold = thresholds['height_threshold']
		orientation_threshold = thresholds['orientation_threshold']
		end_effector_threshold = thresholds['end_effector_threshold']
		
		# Track individual termination reasons for debugging
		termination_reasons = {
			'angles': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
			'height': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
			'orientation': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
			'end_effector': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
		}
		
		# 1. Check joint angle deviation
		if angle_threshold > 0:
			dof_imit_arr = self.imit_joint_pos[indices]
			joint_angle_errors = torch.norm(self.dof_pos - dof_imit_arr, dim=-1)
			angle_violation = joint_angle_errors > angle_threshold
			termination_mask |= angle_violation
			termination_reasons['angles'] = angle_violation
		
		# 2. Check height deviation  
		if height_threshold > 0:
			height_ref = self.imit_height[indices].squeeze(-1)
			base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
			height_errors = torch.abs(base_height - height_ref)
			height_violation = height_errors > height_threshold
			termination_mask |= height_violation
			termination_reasons['height'] = height_violation
		
		# 3. Check orientation deviation (quaternion distance)
		if orientation_threshold > 0:
			quat_ref = self.imit_quaternions[indices]
			# Compute quaternion distance (1 - |dot_product|)
			quat_dot = torch.sum(self.base_quat * quat_ref, dim=-1)
			quat_distance = 1.0 - torch.abs(quat_dot)
			orientation_violation = quat_distance > orientation_threshold
			termination_mask |= orientation_violation
			termination_reasons['orientation'] = orientation_violation
		
		# 4. Check end effector position deviation
		if end_effector_threshold > 0:
			end_effector_ref = self.imit_end_effector[indices]
			
			# Transform current foot positions to body frame (same as in reward function)
			cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
			footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
			for i in range(4):
				footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
																  cur_footsteps_translated[:, i, :])
			footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 12)
			
			end_effector_errors = torch.norm(end_effector_ref - footsteps_in_body_frame, dim=-1)
			end_effector_violation = end_effector_errors > end_effector_threshold
			termination_mask |= end_effector_violation
			termination_reasons['end_effector'] = end_effector_violation
		
		# Apply termination mask to reset buffer
		self.reset_buf |= termination_mask
		
		# Optional: Log termination reasons (useful for debugging)
		if hasattr(self.cfg.termination, 'log_termination_reasons') and self.cfg.termination.log_termination_reasons:
			if torch.any(termination_mask):
				self._log_termination_reasons(termination_reasons, termination_mask, thresholds)
	
	def get_termination_curriculum_info(self):
		""" Get current curriculum progress for monitoring (simplified) """
		if not hasattr(self.cfg.termination, 'use_curriculum') or not self.cfg.termination.use_curriculum:
			return None
		
		current_iteration = getattr(self, 'global_training_iteration', 0)
		start_iter = getattr(self.cfg.termination, 'curriculum_start_iteration', 200)
		end_iter = getattr(self.cfg.termination, 'curriculum_end_iteration', 800)
		
		# Simple progress calculation
		if current_iteration < start_iter:
			progress = 0.0
			phase = "initial"
		elif current_iteration >= end_iter:
			progress = 1.0
			phase = "final"
		else:
			progress = (current_iteration - start_iter) / (end_iter - start_iter)
			phase = "curriculum"
		
		thresholds = self._get_curriculum_termination_thresholds()
		
		return {
			'iteration': current_iteration,
			'progress': progress,
			'phase': phase,
			'thresholds': thresholds
		}
	
	def _log_termination_reasons(self, reasons, termination_mask, thresholds=None):
		""" Log detailed termination statistics for debugging """
		num_terminated = torch.sum(termination_mask).item()
		if num_terminated > 0:
			step = getattr(self, 'common_step_counter', 0)
			iteration = getattr(self, 'global_training_iteration', 0)
			
			# Count terminations by reason
			angle_count = torch.sum(reasons['angles']).item()
			height_count = torch.sum(reasons['height']).item()
			orientation_count = torch.sum(reasons['orientation']).item()
			end_effector_count = torch.sum(reasons['end_effector']).item()
			
			log_msg = f"Iter {iteration}, Step {step}: {num_terminated} envs terminated - "
			breakdown = []
			if angle_count > 0:
				breakdown.append(f"angles: {angle_count}")
			if height_count > 0:
				breakdown.append(f"height: {height_count}")
			if orientation_count > 0:
				breakdown.append(f"orientation: {orientation_count}")
			if end_effector_count > 0:
				breakdown.append(f"end_effector: {end_effector_count}")
			
			if breakdown:
				log_msg += ", ".join(breakdown)
				
			# Add current threshold values if available
			if thresholds is not None:
				log_msg += f" | Thresholds: angle={thresholds['angle_threshold']:.3f}, " \
						   f"height={thresholds['height_threshold']:.3f}, " \
						   f"orient={thresholds['orientation_threshold']:.3f}, " \
						   f"end_eff={thresholds['end_effector_threshold']:.3f}"
			
			print(log_msg)

	# Thank you for the multi-critic code Dr. Sun!
	def compute_reward(self):
		"""
		Compute rewards.

		In multi–reward group mode, self.rew_buf is assumed to have shape
		[num_env, num_reward_groups] (or [num_env, num_reward_groups + 1] if termination is defined).

		For each reward group (excluding "termination"), this function computes the cumulative reward
		(by summing each reward function's output scaled by its factor) and assigns the result to the corresponding
		column. Then, if a termination reward is defined, it is computed and concatenated as an extra column.

		In single–group mode, self.rew_buf is assumed to be [num_env] and the original logic applies.

		Returns:
			rewards_tensor (torch.Tensor): In multi–group mode, tensor of shape
				[num_env, num_reward_groups (+1 if termination)], or in single–group mode, the overall reward tensor.
		"""
		if getattr(self.cfg.rewards, "multi_critic", False):
			# MULTI-GROUP MODE
			# Assume self.rew_buf is already allocated with shape [num_env, num_reward_groups]
			# We'll compute a new tensor with the same shape.
			group_names = [gn for gn in self.reward_scales.keys() if gn != "termination"]
			num_groups = len(group_names)
			# Create an empty tensor for computed rewards for each group.
			computed_rewards = torch.zeros(self.num_envs, num_groups, dtype=torch.float, device=self.device)

			# Process each reward group (except "termination")
			for idx, group_name in enumerate(group_names):
				group_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
				# For each reward in the current group, accumulate its contribution.
				for reward_name in self.reward_names[group_name]:
					r = self.reward_functions[group_name][reward_name]() * self.reward_scales[group_name][reward_name]
					group_reward += r
					# Update episode sums for logging.
					self.episode_sums[group_name][reward_name] += r
				computed_rewards[:, idx] = group_reward
			# If only positive rewards are allowed, clip the computed rewards.
			if self.cfg.rewards.only_positive_rewards:
				computed_rewards = torch.clamp(computed_rewards, min=0.)

			# Handle termination reward if defined.
			if "termination" in self.reward_scales:
				term_rew = self._reward_termination() * self.reward_scales["termination"]
				# Update termination episode sum.
				if "termination" in self.episode_sums:
					self.episode_sums["termination"] += term_rew
				else:
					self.episode_sums["termination"] = term_rew.clone()
				term_rew = term_rew.unsqueeze(1)  # shape: [num_env, 1]
				computed_rewards = torch.cat([computed_rewards, term_rew], dim=1)

			# Copy the computed rewards into self.rew_buf.
			self.rew_buf.copy_(computed_rewards)
			return computed_rewards
		else:
			# SINGLE-GROUP MODE (original logic)
			self.rew_buf[:] = 0.
			for i in range(len(self.reward_functions)):
				name = self.reward_names[i]
				rew = self.reward_functions[i]() * self.reward_scales[name]
				self.rew_buf += rew
				self.episode_sums[name] += rew
			if self.cfg.rewards.only_positive_rewards:
				self.rew_buf[:] = torch.clamp(self.rew_buf[:], min=0.)
			if "termination" in self.reward_scales:
				rew = self._reward_termination() * self.reward_scales["termination"]
				self.rew_buf += rew
				self.episode_sums["termination"] += rew
			return self.rew_buf
			
	def post_physics_step(self):
		"""Override to add instantaneous RMSE error computation and logging to wandb"""
		# Call parent method first
		super().post_physics_step()
		
		# Compute instantaneous RMSE errors for imitation
		if hasattr(self, 'imitation_index') and hasattr(self, 'df_imit'):
			rmse_errors = self._compute_instantaneous_rmse_errors()
			
			# Add instantaneous RMSE errors to extras for immediate wandb logging
			if "step" not in self.extras:
				self.extras["step"] = {}
			
			# Log instantaneous RMSE values at each step
			for key, value in rmse_errors.items():
				self.extras["step"][key] = value
			
			# Debug print for first few steps to verify logging is working
			# if hasattr(self, 'common_step_counter') and self.common_step_counter % 1000 == 0 and self.common_step_counter < 10000:
			# 	print(f"Step {self.common_step_counter} - Instantaneous RMSE: Joint={rmse_errors['instantaneous_joint_pos_rmse']:.4f}, Height={rmse_errors['instantaneous_imitation_height_rmse']:.4f}, EndEff={rmse_errors['instantaneous_end_effector_pos_rmse']:.4f}, Quat={rmse_errors['instantaneous_quaternion_rmse']:.4f}, LinVel={rmse_errors['instantaneous_lin_vel_tracking_rmse']:.4f}, AngVel={rmse_errors['instantaneous_ang_vel_tracking_rmse']:.4f}")


			
	def _prepare_reward_function(self):
		"""
		Prepares reward functions to compute the total reward.

		For a single reward group, looks for self._reward_<REWARD_NAME> for each reward name
		(from non-zero scales in self.reward_scales), multiplies the scale by self.dt,
		and prepares self.reward_functions, self.reward_names, and self.episode_sums.

		For multiple reward groups (when self.cfg.rewards.multi_critic is True),
		self.reward_scales is expected to be a dict with keys as group names (e.g., "group1", "group2")
		and values as dicts mapping reward names to scales. For each group (ignoring "termination"),
		it removes zero scales, multiplies nonzero scales by self.dt, and searches for reward functions
		following the naming convention: _reward_<group_name>_<reward_name>. It then creates nested dictionaries
		for self.reward_functions, self.reward_names, and self.episode_sums.
		"""
		if getattr(self.cfg.rewards, "multi_critic", False):
			# MULTI-GROUP MODE
			self.reward_functions = {}  # {group_name: {reward_name: function, ...}, ...}
			self.reward_names = {}  # {group_name: [reward_name, ...], ...}

			# Process each reward group (except termination)
			for group_name, group_scales in self.reward_scales.items():
				if group_name == "termination":
					continue  # Skip termination; it will be handled separately.
				valid_scales = {}
				names = []
				functions = {}
				for key in list(group_scales.keys()):
					scale = group_scales[key]
					if scale == 0:
						continue
					else:
						# Multiply non-zero scales by dt.
						valid_scales[key] = scale * self.dt
						names.append(key)
						# Expected function name: _reward_<group_name>_<key>
						func_name = f"_reward_{key}"
						functions[key] = getattr(self, func_name)
				# Update the group's scales with only the valid ones.
				self.reward_scales[group_name] = valid_scales
				self.reward_names[group_name] = names
				self.reward_functions[group_name] = functions

			# Prepare episode sums as a nested dictionary per group.
			self.episode_sums = {
				group_name: {
					reward_name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
					for reward_name in self.reward_scales[group_name].keys()
				}
				for group_name in self.reward_scales.keys() if group_name != "termination"
			}
		else:
			# SINGLE-GROUP MODE (original logic)
			for key in list(self.reward_scales.keys()):
				scale = self.reward_scales[key]
				if scale == 0:
					self.reward_scales.pop(key)
				else:
					self.reward_scales[key] *= self.dt
			self.reward_functions = []
			self.reward_names = []
			for name, scale in self.reward_scales.items():
				if name == "termination":
					continue
				self.reward_names.append(name)
				func_name = '_reward_' + name
				self.reward_functions.append(getattr(self, func_name))
			# Prepare episode sums.
			self.episode_sums = {
				name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
				for name in self.reward_scales.keys()
			}

	def _compute_decap_factor(self):
		if self.decap_type == "exp":
			self.decap_factor = self.gamma_decap**(self.torque_ref_decay_factor/self.k_decap)
			return self.decap_factor 
		elif self.decap_type == "cosine":
			# Calculate the cosine decay factor based on the current iteration
			if self.global_training_iteration < self.cosine_constant_prior_iterations:
				#1.0 for the first cosine_constant_prior_iterations iterations with size equal to actions
				self.decap_factor = 1.0
			elif self.global_training_iteration < self.cosine_constant_prior_iterations + self.cosine_decay_iterations:
				# Calculate the cosine decay factor
				decay_factor = 0.5 * (1 + np.cos(
					np.pi * (self.global_training_iteration - self.cosine_constant_prior_iterations) / self.cosine_decay_iterations))
				self.decap_factor = decay_factor 
			else:
				#Zero otherwise
				self.decap_factor = 0.0
			return self.decap_factor
		elif self.decap_type == 'discrete':
			#Keep decap factor 1.0 for the first discrete_constant_prior_iterations, then half and then zero
			if self.global_training_iteration < 500:
				self.decap_factor = 1.0
			elif self.global_training_iteration < 1000 and self.global_training_iteration >= 500:
				self.decap_factor = 0.5
			else:
				self.decap_factor = 0.0
			return self.decap_factor
		else:
			raise ValueError(f"Unknown decap type: {self.decap_type}")
		

	def _compute_torques(self, actions):
		# Per-DOF action scaling (fallback to global if alpha missing)
		if hasattr(self, "action_alpha"):
			# actions: [num_env, num_dof], alpha: [1, num_dof] -> broadcast
			actions_scaled = actions * self.action_alpha
			# print("NEW KP and KD", self.p_gains, self.d_gains)
		else:
			actions_scaled = actions * self.cfg.control.action_scale

		actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
		
		if self.cfg.domain_rand.randomize_lag_timesteps:
			self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
			self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
		else:
			self.joint_pos_target = actions_scaled + self.default_dof_pos

		control_type = self.cfg.control.control_type
		if self.cfg.control.exp_avg_decay:
			self.action_avg = exp_avg_filter(actions_scaled, self.action_avg,
											self.cfg.control.exp_avg_decay)
			actions_scaled = self.action_avg

		if self.cfg.control.limit_dof_pos:
			actions_scaled = torch.clip(actions_scaled, -0.2, 0.2)

		dof_imit_arr = self.imit_joint_pos[self.imitation_index.long()]
		self.decap_factor = self._compute_decap_factor()
		self.decap_factor = np.clip(self.decap_factor, 0.0, 1.0)
		self.decap_factor = torch.full((self.num_envs, 1), self.decap_factor, device=self.device)

		if control_type == 'apex_position':
			#Decap factor can empirically set as max (0.06, decap_factor) for more stable training after decay, can be removed without any major performance loss
			self.decap_factor = torch.clip(self.decap_factor, 0.06, 1.0)
			#For decap p_gains can have stronger gains for more bias if needed
			# p_gains_decap = self.p_gains * 1.3
			torques =  self.p_gains*self.Kp_factors*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.Kd_factors*self.d_gains*self.dof_vel +  self.decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos))
			# torques =  self.p_gains*self.Kp_factors*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.Kd_factors*self.d_gains*self.dof_vel +  self.decap_factor*(p_gains_decap*(dof_imit_arr - self.dof_pos))
		
		elif control_type == 'apex_torque':
			#Add the imitation bias along with the decap factor
			torques = actions_scaled + self.decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)

		elif control_type=="position":
			torques = self.p_gains* self.Kp_factors*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains*self.Kd_factors*self.dof_vel

		elif control_type == 'torque':
			torques = actions_scaled

		elif control_type == 'apex_position_grow':
			# decap_factor = gamma_decap**(self.torque_ref_decay_factor/k_decap)
			torques = (1-self.decap_factor)*(self.p_gains*self.Kp_factors*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.Kd_factors*self.d_gains*self.dof_vel) \
				 	 + self.decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)

		elif control_type == 'play_retarget_angles':
			torques = self.p_gains*(dof_imit_arr - self.dof_pos) - self.d_gains*self.dof_vel
			
		else:
			raise ValueError(f"Unknown control type: {control_type}")
		
		torques = torques * self.motor_strengths 
		return torch.clip(torques, -self.torque_limits, self.torque_limits)
	
	#Observations
	def compute_observations(self):
		""" Computes observations
		"""
		# sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
		# cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
	
		# Add a phase observation based on imitation index and time
		imitation_phase = self.imitation_index / self.df_imit_length
		skill_number = self._compute_skill_number(imitation_phase) if self.train_multi_skills else None


		if self.number_observations == 39:
			self.obs_buf = torch.cat((  self.projected_gravity,
							(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
							self.dof_vel * self.obs_scales.dof_vel,
							self.actions,
							),dim=-1)
		elif self.number_observations == 42:
			self.obs_buf = torch.cat((  self.projected_gravity,
							self.commands[:, :3] * self.commands_scale,
							(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
							self.dof_vel * self.obs_scales.dof_vel,
							self.actions,
							),dim=-1)
		elif self.number_observations == 45:
			self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
							self.projected_gravity,
							self.commands[:, :3] * self.commands_scale,
							(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
							self.dof_vel * self.obs_scales.dof_vel,
							self.actions,
							),dim=-1)
		
		elif self.number_observations == 43 and not self.train_multi_skills:
			self.obs_buf = torch.cat((  self.projected_gravity,
				self.commands[:, :3] * self.commands_scale,
				(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
				self.dof_vel * self.obs_scales.dof_vel,
				self.actions,
				imitation_phase.unsqueeze(1),
				),dim=-1)
		elif self.number_observations == 77:
			indices = self.imitation_index.long()
			# Reference joint positions (12 values)
			dof_imit_arr = self.imit_joint_pos[indices]
			# Reference end effector positions (12 values) 
			end_effector_ref = self.imit_end_effector[indices]
			# Reference quaternions (4 values)
			quat_ref = self.imit_quaternions[indices]
			self.obs_buf = torch.cat((
				self.base_lin_vel * self.obs_scales.lin_vel,        # 3
				self.base_ang_vel * self.obs_scales.ang_vel,        # 3
				self.projected_gravity,                             # 3
				self.commands[:, :3] * self.commands_scale,         # 3
				(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
				self.dof_vel * self.obs_scales.dof_vel,             # 12
				self.actions,                                       # 12
				imitation_phase.unsqueeze(1),                       # 1
				dof_imit_arr,                                       # 12 (reference joint positions)
				end_effector_ref,                                   # 12 (reference foot positions)
				quat_ref,                                           # 4 (reference quaternions)
			), dim=-1)
			# Total: 3+3+3+3+12+12+12+1+12+12+4 = 77
		else:
			self.obs_buf = torch.cat((  self.projected_gravity,
							self.commands[:, :3] * self.commands_scale,
							(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
							self.dof_vel * self.obs_scales.dof_vel,
							self.actions,
							skill_number.unsqueeze(1),
							),dim=-1)
			
		if self.number_privileged_observations == 48:
			self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
								self.base_ang_vel  * self.obs_scales.ang_vel,
								self.projected_gravity,
								self.commands[:, :3] * self.commands_scale,
								(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
								self.dof_vel * self.obs_scales.dof_vel,
								self.actions,
								),dim=-1)
		elif self.number_privileged_observations == 49 and self.train_multi_skills:
						self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
								self.base_ang_vel  * self.obs_scales.ang_vel,
								self.projected_gravity,
								self.commands[:, :3] * self.commands_scale,
								(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
								self.dof_vel * self.obs_scales.dof_vel,
								self.actions,
								skill_number.unsqueeze(1),
								),dim=-1)
		elif self.number_privileged_observations == 49 and not self.train_multi_skills:
				self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
						self.base_ang_vel  * self.obs_scales.ang_vel,
						self.projected_gravity,
						self.commands[:, :3] * self.commands_scale,
						(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
						self.dof_vel * self.obs_scales.dof_vel,
						self.actions,
						imitation_phase.unsqueeze(1),
						),dim=-1)
		
		elif self.number_privileged_observations == 50:
			self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
						self.base_ang_vel  * self.obs_scales.ang_vel,
						self.projected_gravity,
						self.commands[:, :3] * self.commands_scale,
						(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
						self.dof_vel * self.obs_scales.dof_vel,
						self.actions,
						imitation_phase.unsqueeze(1),
						self.decap_factor,	
						),dim=-1)
		elif self.number_privileged_observations == 77:
			# Get imitation data for current timestep - using preprocessed tensors
			indices = self.imitation_index.long()
			# Reference joint positions (12 values)
			dof_imit_arr = self.imit_joint_pos[indices]			
			# Reference end effector positions (12 values) 
			end_effector_ref = self.imit_end_effector[indices]			
			# Reference quaternions (4 values)
			quat_ref = self.imit_quaternions[indices]
			
			self.privileged_obs_buf = torch.cat((
				self.base_lin_vel * self.obs_scales.lin_vel,        # 3
				self.base_ang_vel * self.obs_scales.ang_vel,        # 3
				self.projected_gravity,                             # 3
				self.commands[:, :3] * self.commands_scale,         # 3
				(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
				self.dof_vel * self.obs_scales.dof_vel,             # 12
				self.actions,                                       # 12
				imitation_phase.unsqueeze(1),                       # 1
				dof_imit_arr,                                       # 12 (reference joint positions)
				end_effector_ref,                                   # 12 (reference foot positions)
				quat_ref,                                           # 4 (reference quaternions)
			), dim=-1)
			# Total: 3+3+3+3+12+12+12+1+12+12+4 = 77
		else:
			self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
						self.base_ang_vel  * self.obs_scales.ang_vel,
						self.projected_gravity,
						self.commands[:, :3] * self.commands_scale,
						(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
						self.dof_vel * self.obs_scales.dof_vel,
						self.actions,
						imitation_phase.unsqueeze(1)
						),dim=-1)
				
		# add perceptive inputs if not blind
		if self.cfg.terrain.measure_heights:
			heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
			self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
			self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
					# --- Heightmap visualization for the first robot ---
			if self.visualize_imitation_data:
				# self.gym.clear_lines(self.viewer)
				sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
				base_pos = self.root_states[0, :3].cpu().numpy()
				heights0 = self.measured_heights[0].cpu().numpy()
				# height_points: (num_envs, num_height_points, 3)
				height_points = quat_apply_yaw(self.base_quat[0].repeat(heights0.shape[0]), self.height_points[0]).cpu().numpy()
				for j in range(heights0.shape[0]):
					x = height_points[j, 0] + base_pos[0]
					y = height_points[j, 1] + base_pos[1]
					z = heights0[j]
					sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
					gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

		if self.cfg.terrain.measure_heights_priv and not self.cfg.terrain.measure_heights:
			heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
			self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
		
		# add noise if needed
		if self.add_noise:
			self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

	def _compute_skill_number(self, imitation_phase):
		"""Compute skill number based on imitation phase"""
		#TODO For now hard coding four skills based on the gaits available 
		skill_number = torch.zeros_like(imitation_phase)
		skill_number = torch.where((imitation_phase >= 0.0) & (imitation_phase < 0.25), 0.0, skill_number)
		skill_number = torch.where((imitation_phase >= 0.25) & (imitation_phase < 0.5), 0.25, skill_number)
		skill_number = torch.where((imitation_phase >= 0.5) & (imitation_phase < 0.75), 0.5, skill_number)
		skill_number = torch.where((imitation_phase >= 0.75) & (imitation_phase <= 1.0), 0.75, skill_number)
		return skill_number

	def _process_dof_props(self, props, env_id):
		""" Callback allowing to store/change/randomize the DOF properties of each environment.
			Called During environment creation.
			Base behavior: stores position, velocity and torques limits defined in the URDF

		Args:
			props (numpy.array): Properties of each DOF of the asset
			env_id (int): Environment id

		Returns:
			[numpy.array]: Modified DOF properties
		"""
		if env_id == 0:
			self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
											  requires_grad=False)
			self.termination_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
														  requires_grad=False)
			self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
												   requires_grad=False)
			self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
			self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
			for i in range(len(props)):
				self.dof_pos_limits[i, 0] = props["lower"][i].item()
				self.dof_pos_limits[i, 1] = props["upper"][i].item()

				self.termination_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] - 0.05
				self.termination_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] + 0.05

				self.dof_vel_limits[i] = props["velocity"][i].item()
				self.torque_limits[i] = props["effort"][i].item()

				m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
				r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
				self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
				self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
		return props

	def _process_rigid_body_props(self, props, env_id):
		# randomize base mass
		if self.cfg.domain_rand.randomize_base_mass:
			rng = self.cfg.domain_rand.added_mass_range
			com_rng_x = self.cfg.domain_rand.shifted_com_range_x
			com_rng_y = self.cfg.domain_rand.shifted_com_range_y
			com_rng_z = self.cfg.domain_rand.shifted_com_range_z
			c_x = np.random.uniform(com_rng_x[0], com_rng_x[1])
			c_y = np.random.uniform(com_rng_y[0], com_rng_y[1])
			c_z = np.random.uniform(com_rng_z[0], com_rng_z[1])
			rnd_mass = np.random.uniform(rng[0], rng[1])
			point_mass_pos = np.array([c_x, c_y, c_z])

			# Store original mass before modification
			original_mass = props[0].mass
			# Add a point mass to the base
			props[0].mass += rnd_mass
			com_prev = np.array([props[0].com.x, props[0].com.y, props[0].com.z])
			inertia_prev = np.array([[props[0].inertia.x.x, props[0].inertia.x.y, props[0].inertia.x.z],
									 [props[0].inertia.y.x, props[0].inertia.y.y, props[0].inertia.y.z],
									 [props[0].inertia.z.x, props[0].inertia.z.y, props[0].inertia.z.z]])
			# Use original mass, not updated mass
			inertia, com = update_inertia(inertia_prev, original_mass, com_prev, rnd_mass, point_mass_pos)
			# Assign (not add) the new total inertia
			props[0].inertia.x = gymapi.Vec3(inertia[0, 0], inertia[0, 1], inertia[0, 2])
			props[0].inertia.y = gymapi.Vec3(inertia[1, 0], inertia[1, 1], inertia[1, 2])
			props[0].inertia.z = gymapi.Vec3(inertia[2, 0], inertia[2, 1], inertia[2, 2])
			props[0].com = gymapi.Vec3(com[0], com[1], com[2])
			for i in range(len(props)):
				props[i].mass += np.random.uniform(rng[0] / 16, rng[1] / 16)

		# randomize link masses
		if self.cfg.domain_rand.randomize_link_mass:
			self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)
	
			for i in range(1, len(props)):
				props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]

		return props

	def _parse_cfg(self, cfg):
		self.dt = self.cfg.control.decimation * self.sim_params.dt
		self.obs_scales = self.cfg.normalization.obs_scales
		# Assuming self.cfg is an instance of RewardsConfig:
		if self.cfg.rewards.multi_critic:
			self.reward_scales = {}
			for i in range(self.cfg.rewards.reward_group_num):
				group_name = f"group{i + 1}"
				self.reward_scales[group_name] = class_to_dict(getattr(self.cfg.rewards.scales, group_name))
		else:
			self.reward_scales = class_to_dict(self.cfg.rewards.scales)
		self.command_ranges = class_to_dict(self.cfg.commands.ranges)
		self.max_episode_length_s = self.cfg.env.episode_length_s
		self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

		self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

	def _reward_no_fly(self):
		contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
		single_contact = torch.sum(1.*contacts, dim=1)==2
		return 1.*single_contact
	
	def _compute_instantaneous_rmse_errors(self):
		"""Compute instantaneous RMSE errors for joint positions, imitation height, end effector positions, velocity tracking, and quaternion orientation"""
		indices = self.imitation_index.long()
		
		# Joint position RMSE - using preprocessed data
		dof_imit_arr = self.imit_joint_pos[indices]
		joint_pos_rmse = torch.sqrt(torch.mean(torch.square(self.dof_pos - dof_imit_arr) * self.obs_scales.dof_imit, dim=-1))
		
		# Imitation height RMSE - using preprocessed data
		height_ref = self.imit_height[indices].squeeze(-1)
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height_rmse = torch.sqrt(torch.square(base_height - height_ref))
		
		# End effector position RMSE - using preprocessed data
		end_effector_ref = self.imit_end_effector[indices]
		
		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
		for i in range(4):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])
		footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 12)
		end_effector_rmse = torch.sqrt(torch.mean(torch.square(end_effector_ref - footsteps_in_body_frame), dim=-1))
		
		# Quaternion RMSE - using quaternion distance for meaningful orientation error
		quat_ref = self.imit_quaternions[indices]
		quat_dot = torch.sum(self.base_quat * quat_ref, dim=-1)
		quat_distance = 1.0 - torch.abs(quat_dot)  # Quaternion distance: 0 = identical, 2 = opposite
		quat_rmse = torch.sqrt(torch.mean(torch.square(quat_distance), dim=0))
		
		# Velocity tracking RMSE - Linear velocity (X, Y only for commanded)
		lin_vel_error = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
		lin_vel_rmse = torch.sqrt(torch.mean(lin_vel_error, dim=-1))
		
		# Velocity tracking RMSE - Angular velocity (Yaw only for commanded)
		ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
		ang_vel_rmse = torch.sqrt(ang_vel_error)
		
		return {
			'instantaneous_joint_pos_rmse': torch.mean(joint_pos_rmse).item(),
			'instantaneous_imitation_height_rmse': torch.mean(height_rmse).item(), 
			'instantaneous_end_effector_pos_rmse': torch.mean(end_effector_rmse).item(),
			'instantaneous_quaternion_rmse': quat_rmse.item(),
			'instantaneous_lin_vel_tracking_rmse': torch.mean(lin_vel_rmse).item(),
			'instantaneous_ang_vel_tracking_rmse': torch.mean(ang_vel_rmse).item()
		}
	
	def _reward_imitation_angles(self):
		# Use preprocessed imitation data
		dof_imit_arr = self.imit_joint_pos[self.imitation_index.long()]
		
		dof_imit_error = torch.mean(torch.square(self.dof_pos - dof_imit_arr)*self.obs_scales.dof_imit, dim=-1)  
		
		return torch.exp(-dof_imit_error / self.sigma_imit_angles)
	
	def _reward_imitation_height(self):
		# Use preprocessed imitation data
		height = self.imit_height[self.imitation_index.long()].squeeze(-1)
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height_error = torch.square(base_height - height)
		
		return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)


	def _reward_imitation_angles_indiv_legs(self):
		# Use preprocessed imitation data - columns 6:18 split into 3 per leg
		joint_pos_imit = self.imit_joint_pos[self.imitation_index.long()]
		
		dof_imit_arr_leg1 = joint_pos_imit[:, 0:3]   # columns 6:9
		dof_imit_arr_leg2 = joint_pos_imit[:, 3:6]   # columns 9:12
		dof_imit_arr_leg3 = joint_pos_imit[:, 6:9]   # columns 12:15
		dof_imit_arr_leg4 = joint_pos_imit[:, 9:12]  # columns 15:18

		dof_imit_error_leg1 = torch.sum(torch.square((self.dof_pos[:,0:3] - dof_imit_arr_leg1)*self.obs_scales.dof_imit), dim=1)    
		dof_imit_error_leg2 = torch.sum(torch.square((self.dof_pos[:,3:6] - dof_imit_arr_leg2)*self.obs_scales.dof_imit), dim=1)
		dof_imit_error_leg3 = torch.sum(torch.square((self.dof_pos[:,6:9] - dof_imit_arr_leg3)*self.obs_scales.dof_imit), dim=1)
		dof_imit_error_leg4 = torch.sum(torch.square((self.dof_pos[:,9:12] - dof_imit_arr_leg4)*self.obs_scales.dof_imit), dim=1)

		reward = torch.exp(-dof_imit_error_leg1/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg2/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg3/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg4/self.cfg.rewards.tracking_sigma) 
		return reward

	def _reward_imitation_lin_vel(self):
		# Use preprocessed imitation data
		lin_vel_imit_arr = self.imit_lin_vel[self.imitation_index.long()]
		
		lin_imit_error = torch.sum(torch.abs(self.base_lin_vel - lin_vel_imit_arr), dim=1)    
		return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)
	
	def _reward_imitation_ang_vel(self):
		# Use preprocessed imitation data
		ang_vel_imit_arr = self.imit_ang_vel[self.imitation_index.long()]
		
		lin_imit_error = torch.sum(torch.abs(self.base_ang_vel - ang_vel_imit_arr), dim=1)
		return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)
	
	def _reward_imitation_height_penalty(self):
		# Use preprocessed imitation data
		height = self.imit_height[self.imitation_index.long()].squeeze(-1)  # Only squeeze last dim
		
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		
		#Plot imitation height with draw_sphere:
		if self.visualize_imitation_data:
			# self.clear_lines()
			sphere_radius = 0.07  # Adjust as needed
			sphere_color = (0.1, 0.8, 0.07)  # Green for target positions

			for env_id in range(1):
				# Extract robot's (x, y) position from root_states
				base_x = self.root_states[env_id, 0].item()
				base_y = self.root_states[env_id, 1].item()
				pos_world_frame = [base_x, base_y, height[env_id].item()]
				self.draw_sphere(pos_world_frame, sphere_radius, sphere_color, env_id)

		return torch.square(base_height - height)
	
	def _reward_imitate_end_effector_pos(self):
		# Use preprocessed imitation data
		end_effector_ref = self.imit_end_effector[self.imitation_index.long()]

		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
		for i in range(4):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])
		footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 12)

		end_effector_error = torch.sum(torch.square(end_effector_ref - footsteps_in_body_frame), dim=-1)
		
		if self.visualize_imitation_data:
			#Clear lines after 10 steps
			self.clear_lines()
			sphere_radius = 0.03  # Adjust as needed
			sphere_color = (0.1, 0.8, 0.07)  # Green for target positions

			for env_id in range(1):
				for i in range(4):
					# Extract reference end-effector position (body frame)
					pos_body_frame = end_effector_ref[env_id, i * 3: (i + 1) * 3]

					# Transform to world frame
					pos_world_frame = quat_apply_yaw(self.base_quat[env_id], pos_body_frame) + self.base_pos[env_id]

					# Convert to numpy and visualize
					self.draw_sphere(pos_world_frame.cpu().numpy(), sphere_radius, sphere_color, env_id)

		return torch.exp(-end_effector_error/self.sigma_imit_foot_pos)
	
	def _reward_imitate_base_end_effector_pos_world(self):
		# Use preprocessed imitation data
		end_effector_ref_wf = self.imit_end_effector_world[self.imitation_index.long()]

		# Get translation offset from default base position
		init_pos = self.default_base_pos.clone()
		init_pos[:,2] = 0
		#Translate the reference for each robot by adding the (x,y) of the default base position
		init_pos_footsteps_translation = torch.cat([init_pos, init_pos, init_pos, init_pos], dim=1)
		end_effector_ref_wf = end_effector_ref_wf + init_pos_footsteps_translation

		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		end_effector_error = torch.sum(torch.square(end_effector_ref_wf - cur_foot_pos), dim=1)
		if self.visualize_imitation_data:
			self.clear_lines()
			sphere_radius = 0.04  # Adjust as needed
			sphere_color = (0.1, 0.07, 0.8)  # Green for target positions

			for env_id in range(1):
				for i in range(4):
					# Extract reference end-effector position (body frame)
					pos_world_frame = end_effector_ref_wf[env_id, i * 3: (i + 1) * 3]
					self.draw_sphere(pos_world_frame.cpu().numpy(), sphere_radius, sphere_color, env_id)
		return torch.exp(-10*end_effector_error)
		
	def _reward_imitate_foot_height(self):
		# Use preprocessed imitation data
		end_effector_z1_ref = self.imit_foot_z1[self.imitation_index.long()]
		end_effector_z2_ref = self.imit_foot_z2[self.imitation_index.long()]
		end_effector_z3_ref = self.imit_foot_z3[self.imitation_index.long()]
		end_effector_z4_ref = self.imit_foot_z4[self.imitation_index.long()]
		
		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		foot_height_error = torch.sum(torch.square(end_effector_z1_ref - cur_foot_pos[:,2].unsqueeze(1)), dim=1) \
			+ torch.sum(torch.square(end_effector_z2_ref - cur_foot_pos[:,5].unsqueeze(1)), dim=1) + torch.sum(torch.square(end_effector_z3_ref - cur_foot_pos[:,8].unsqueeze(1)), dim=1) \
				  + torch.sum(torch.square(end_effector_z4_ref - cur_foot_pos[:,11].unsqueeze(1)), dim=1)
		return torch.exp(-90*foot_height_error)


	def _reward_imitate_base_pos(self):
		# Use preprocessed imitation data
		base_pos_ref = self.imit_base_pos[self.imitation_index.long()]

		#Add the initial base position to convert to world frame of each robot
		base_pos_ref = base_pos_ref + self.default_base_pos[:,0:2]

		base_pos_error = torch.sum(torch.square(base_pos_ref - self.base_pos[:,0:2]), dim=1)
		return torch.exp(-4*base_pos_error)


	def _reward_imitate_quat(self):
		# Use preprocessed imitation data
		quat_ref = self.imit_quaternions[self.imitation_index.long()]

		quat_error = torch.sum(torch.square(quat_ref - self.base_quat), dim=1)
		return torch.exp(-quat_error/0.5)
	
	def _reward_imitate_quat_penalty(self):
		# Use preprocessed imitation data
		quat_ref = self.imit_quaternions[self.imitation_index.long()]

		quat_error = torch.sum(torch.square(quat_ref - self.base_quat), dim=1)
		return quat_error

	def _reward_imitate_joint_vel(self):
		# Use preprocessed imitation data
		dof_imit_arr = self.imit_joint_vel[self.imitation_index.long()]
		
		#Reward for tracking joint velocities
		dof_imit_error = torch.sum(torch.square(self.dof_vel - dof_imit_arr)*self.obs_scales.dof_vel, dim=1)
		return torch.exp(-10*dof_imit_error)

	def _reward_min_clearance(self):
		
		# Define thresholds for penalties and rewards
		min_clearance = 0.05  # Minimum foot height to prevent dragging
		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		# Extract foot heights
		foot_heights = torch.stack([
			cur_foot_pos[:,2], cur_foot_pos[:,5], cur_foot_pos[:,8], cur_foot_pos[:,11]
		], dim=1)

		# Penalty for dragging (feet below min_clearance)
		min_height_penalty = torch.sum((foot_heights < min_clearance).float(), dim=1)

		return min_height_penalty
	
	def _reward_target_clearance(self):
		
		# Define thresholds for penalties and rewards
		target_clearance = 0.08  # Encouraged clearance height
		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		# Extract foot heights
		foot_heights = torch.stack([
			cur_foot_pos[:,2], cur_foot_pos[:,5], cur_foot_pos[:,8], cur_foot_pos[:,11]
		], dim=1)

		# Reward for sufficient foot clearance
		clearance_reward = torch.sum((foot_heights > target_clearance).float(), dim=1)

		return clearance_reward

	def _reward_penalty_foot_dragging(self, clearance_ratio=0.8):
		# Use preprocessed imitation data
		end_effector_z1_ref = self.imit_foot_z1[self.imitation_index.long()].squeeze(-1)
		end_effector_z2_ref = self.imit_foot_z2[self.imitation_index.long()].squeeze(-1)
		end_effector_z3_ref = self.imit_foot_z3[self.imitation_index.long()].squeeze(-1)
		end_effector_z4_ref = self.imit_foot_z4[self.imitation_index.long()].squeeze(-1)

		end_effector_refs = torch.stack([
			end_effector_z1_ref, end_effector_z2_ref, end_effector_z3_ref, end_effector_z4_ref
		], dim=1)

		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)

		# Extract current foot heights
		foot_heights = torch.stack([
			cur_foot_pos[:, 2], cur_foot_pos[:, 5], cur_foot_pos[:, 8], cur_foot_pos[:, 11]
		], dim=1)

		# Compute adaptive minimum clearance as a fraction of the reference height
		min_clearance = clearance_ratio * end_effector_refs

		# Compute penalty for feet below their adaptive clearance threshold
		dragging_penalty = torch.sum((foot_heights < min_clearance).float(), dim=1)

		return dragging_penalty

	def _reward_energy(self):
		# Penalize energy
		return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

	'''
	Rewards from walk these ways
	'''
	
	def _reward_feet_slip(self):
		contact = self.contact_forces[:, self.feet_indices, 2] > 1.
		contact_filt = torch.logical_or(contact, self.last_contacts)
		self.last_contacts = contact
		foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
		rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
		return rew_slip
	
	def _reward_tracking_contacts_shaped_force(self):
		foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
		desired_contact = self.desired_contact_states

		reward = 0
		gait_force_sigma = 100.0
		for i in range(4):
			# reward += - (1 - desired_contact[:, i]) * (
			#             1 - torch.exp(-1 * foot_forces[:, i] ** 2 / gait_force_sigma))
			reward += - (1 - desired_contact[:, i]) * (
						1 - torch.exp(-1 * foot_forces[:, i] ** 2 / gait_force_sigma))
		return reward / 4

	def _reward_tracking_contacts_shaped_vel(self):
		foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
		desired_contact = self.desired_contact_states
		reward = 0
		gait_vel_sigma = 10.0
		for i in range(4):
			reward += - (desired_contact[:, i] * (
						1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / gait_vel_sigma)))
		return reward / 4
	
	def _reward_raibert_heuristic(self):
		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
		for i in range(4):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])

		# nominal positions: [FR, FL, RR, RL]
		desired_stance_width = 0.3
		desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

		desired_stance_length = 0.45
		desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

		# raibert offsets
		phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
		frequencies = 3.0
		x_vel_des = self.commands[:, 0:1]
		yaw_vel_des = self.commands[:, 2:3]
		y_vel_des = yaw_vel_des * desired_stance_length / 2
		desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
		desired_ys_offset[:, 2:4] *= -1
		desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

		desired_ys_nom = desired_ys_nom + desired_ys_offset
		desired_xs_nom = desired_xs_nom + desired_xs_offset

		desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

		err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

		reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

		return reward
	
	def _reward_feet_clearance_cmd_linear(self):
		phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
		foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
		# target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
		target_height = torch.full((self.num_envs,), 0.08, device=self.device).unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
		rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
		return torch.sum(rew_foot_clearance, dim=1)

	def _reward_straight_heading(self):
		# Calculate the heading error relative to the desired straight direction
		target_heading = 0.0  # Straight heading (aligned with the world frame's positive x-axis)
		
		# Compute the current heading (extracted from base orientation using `projected_gravity` or quaternion)
		current_heading = torch.atan2(self.projected_gravity[:, 1], self.projected_gravity[:, 0])  # in radians
		heading_error = torch.abs(current_heading - target_heading) % (2 * torch.pi)
		heading_error = torch.min(heading_error, 2 * torch.pi - heading_error)
		
		# Penalize the heading error
		# return torch.exp(-10*heading_error)
		return torch.square(heading_error)  # Squared error for smoother gradients


	'''
	For visualizing the imitation foot points and debugging
	'''
	def clear_lines(self):
		self.gym.clear_lines(self.viewer)

	def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
		sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
		sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
		gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)

	def draw_line(self, start_point, end_point, color, env_id):
		gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.envs[env_id])