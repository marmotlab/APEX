from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
import pandas as pd
from legged_gym.utils.math import quat_apply_yaw
import yaml

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	gamma_decap = config["gamma"]
	k_decap = config["k"]
	path_to_imitation_data = config["path_to_imitation_data"]

#Read the imitation data
df_imit = pd.read_csv(path_to_imitation_data, parse_dates=False)

class Go1(LeggedRobot):
	def _compute_torques(self, actions):
		actions_scaled = actions * self.cfg.control.action_scale
		actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
		control_type = self.cfg.control.control_type

		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = self.df_imit.iloc[index_array,6:18].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)
		# Convert the array to a PyTorch tensor
		dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)

		if control_type == 'apex_torque':

			decap_factor = gamma_decap**(self.torque_ref_decay_factor/k_decap)
			# if decap_factor < 0.01:
			# 	print("DECAY has reached 0.01")
			torques = actions_scaled + decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
		
		elif control_type=="position":
			torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

		elif control_type == 'torque':
			torques = actions_scaled
		
		elif control_type == 'decap_position':
			torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel + gamma_decap**(self.torque_ref_decay_factor/k_decap)*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
			# torques = (self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
		
		elif control_type == 'velocity':
			torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt

		elif control_type == 'decap_velocity':
			decap_factor = gamma_decap**(self.torque_ref_decay_factor/k_decap)

			torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt + decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)

		else:
			raise ValueError(f"Unknown control type: {control_type}")
		return torch.clip(torques, -self.torque_limits, self.torque_limits)
	   
	def _reward_no_fly(self):
		contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
		single_contact = torch.sum(1.*contacts, dim=1)==2
		return 1.*single_contact
	
	def _reward_imitation_angles(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = df_imit.iloc[index_array,6:18].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)

		# Convert the array to a PyTorch tensor
		dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)
		dof_imit_error = torch.sum(torch.square(self.dof_pos - dof_imit_arr)*self.obs_scales.dof_imit, dim=1)  
		return torch.exp(-10*dof_imit_error)  
		# return dof_imit_error
	
	def _reward_imitation_height(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		height = df_imit.iloc[index_array,21].to_numpy()
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height = height.reshape(self.num_envs, )
		height = torch.from_numpy(height).float().to(self.device)
		# print("BASE_HEIGHT", base_height.shape, "HEIGHT", height.shape)
		# height_error = torch.sum(torch.abs(base_height - height), dim=1)    
		height_error = torch.square(base_height - height)
		return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)


	def _reward_imitation_angles_indiv_legs(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)

		dof_imit_arr_leg1 = df_imit.iloc[index_array,6:9].to_numpy()
		dof_imit_arr_leg2 = df_imit.iloc[index_array,9:12].to_numpy()
		dof_imit_arr_leg3 = df_imit.iloc[index_array,12:15].to_numpy()
		dof_imit_arr_leg4 = df_imit.iloc[index_array,15:18].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr_leg1 = dof_imit_arr_leg1.reshape(self.num_envs, 3)
		dof_imit_arr_leg2 = dof_imit_arr_leg2.reshape(self.num_envs, 3)
		dof_imit_arr_leg3 = dof_imit_arr_leg3.reshape(self.num_envs, 3)
		dof_imit_arr_leg4 = dof_imit_arr_leg4.reshape(self.num_envs, 3)

		# Convert the array to a PyTorch tensor
		dof_imit_arr_leg1 = torch.from_numpy(dof_imit_arr_leg1).float().to(self.device)
		dof_imit_arr_leg2 = torch.from_numpy(dof_imit_arr_leg2).float().to(self.device)
		dof_imit_arr_leg3 = torch.from_numpy(dof_imit_arr_leg3).float().to(self.device)
		dof_imit_arr_leg4 = torch.from_numpy(dof_imit_arr_leg4).float().to(self.device)

		dof_imit_error_leg1 = torch.sum(torch.square((self.dof_pos[:,0:3] - dof_imit_arr_leg1)*self.obs_scales.dof_imit), dim=1)    
		dof_imit_error_leg2 = torch.sum(torch.square((self.dof_pos[:,3:6] - dof_imit_arr_leg2)*self.obs_scales.dof_imit), dim=1)
		dof_imit_error_leg3 = torch.sum(torch.square((self.dof_pos[:,6:9] - dof_imit_arr_leg3)*self.obs_scales.dof_imit), dim=1)
		dof_imit_error_leg4 = torch.sum(torch.square((self.dof_pos[:,9:12] - dof_imit_arr_leg4)*self.obs_scales.dof_imit), dim=1)

		reward = torch.exp(-dof_imit_error_leg1/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg2/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg3/self.cfg.rewards.tracking_sigma) + torch.exp(-dof_imit_error_leg4/self.cfg.rewards.tracking_sigma) 
		return reward

	def _reward_imitation_lin_vel(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		lin_vel_imit_arr = df_imit.iloc[index_array,:3].to_numpy()

		lin_vel_imit_arr = lin_vel_imit_arr.reshape(self.num_envs, 3)
		lin_vel_imit_arr = torch.from_numpy(lin_vel_imit_arr).float().to(self.device)
		lin_imit_error = torch.sum(torch.abs(self.base_lin_vel - lin_vel_imit_arr), dim=1)    
		return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)
	
	def _reward_imitation_ang_vel(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		ang_vel_imit_arr = df_imit.iloc[index_array,3:6].to_numpy()
		# print(ang_vel_imit_arr.shape)
		ang_vel_imit_arr = ang_vel_imit_arr.reshape(self.num_envs, 3)
		ang_vel_imit_arr = torch.from_numpy(ang_vel_imit_arr).float().to(self.device)
		lin_imit_error = torch.sum(torch.abs(self.base_ang_vel - ang_vel_imit_arr), dim=1)
		return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)
	
	# def _reward_imitation_height(self):
	#     index_array = self.imitation_index.detach().cpu().numpy().astype(int)
	#     # print(index_array)
	#     # Retrieve the corresponding rows from df_imit using array indexing
	#     height = df_imit.iloc[index_array,21].to_numpy()
	#     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
	#     height = height.reshape(self.num_envs, )
	#     height = torch.from_numpy(height).float().to(self.device)
	#     # print("BASE_HEIGHT", base_height.shape, "HEIGHT", height.shape)
	#     # height_error = torch.sum(torch.abs(base_height - height), dim=1)    
	#     height_error = torch.square(base_height - height)
	#     return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)
	
	def _reward_imitation_height_penalty(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		height = df_imit.iloc[index_array,21].to_numpy()
		# print("HEIGHT REFERENCE", height)
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height = height.reshape(self.num_envs, )
		height = torch.from_numpy(height).float().to(self.device)
		return torch.square(base_height - height)
	
	
	def _reward_imitate_end_effector_pos(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_ref = df_imit.iloc[index_array,22:34].to_numpy()
		end_effector_ref = end_effector_ref.reshape(self.num_envs, 12)
		end_effector_ref = torch.from_numpy(end_effector_ref).float().to(self.device)

		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
		for i in range(4):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])
		footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 12)

		end_effector_error = torch.sum(torch.square(end_effector_ref - footsteps_in_body_frame), dim=1)
		# print("END_EFFECTOR_ERROR", end_effector_error[0])
		# print("END_EFFECTOR_REF", end_effector_ref[0])
		# print("INDEX_ARRAY", index_array[0])

		# cur_foot_pos_leg1 = footsteps_in_body_frame[0,:].detach().cpu().numpy() + self.base_pos[0,:].detach().cpu().numpy().repeat(4)
		# target_foot_pos_leg1 = end_effector_ref[0,:].detach().cpu().numpy() + self.base_pos[0,:].detach().cpu().numpy().repeat(4)
		# print("Current foot position: ", cur_foot_pos_leg1 ,target_foot_pos_leg1)
		# sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))

		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(),0, self.foot_positions[0,0,0].item()+0.1,
		#                                                   self.foot_positions[0,0,1].item()+0.1, 0], [0.1, 0.15, 0.5])

		self.gym.clear_lines(self.viewer) 

		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(),self.foot_positions[0,0,2].item(), 
		#                                     end_effector_ref[0,0].item()+ self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(), self.foot_positions[0,0,2].item()], [0.95, 0.5, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,1,0].item(), self.foot_positions[0,1,1].item(),self.foot_positions[0,1,2].item(),
		#                                     end_effector_ref[0,3].item() + self.foot_positions[0,1,0].item(), self.foot_positions[0,1,1].item(), self.foot_positions[0,1,2].item()], [0.95, 0.5, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,2,0].item(), self.foot_positions[0,2,1].item(),self.foot_positions[0,2,2].item(),
		#                                     end_effector_ref[0,6].item() + self.foot_positions[0,2,0].item(), self.foot_positions[0,2,1].item(), self.foot_positions[0,2,2].item()], [0.95, 0.5, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,3,0].item(), self.foot_positions[0,3,1].item(),self.foot_positions[0,3,2].item(),
		#                                     end_effector_ref[0,9].item() + self.foot_positions[0,3,0].item(), self.foot_positions[0,3,1].item(), self.foot_positions[0,3,2].item()], [0.95, 0.5, 0.5])

		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(),self.foot_positions[0,0,2].item(), 
		#                             self.foot_positions[0,0,0].item(), end_effector_ref[0,1].item() + self.foot_positions[0,0,1].item(), self.foot_positions[0,0,2].item()], [0.9, 0.95, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,1,0].item(), self.foot_positions[0,1,1].item(),self.foot_positions[0,1,2].item(),
		#                             self.foot_positions[0,1,0].item(), end_effector_ref[0,4].item() +  self.foot_positions[0,1,1].item(), self.foot_positions[0,1,2].item()], [0.9, 0.95, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,2,0].item(), self.foot_positions[0,2,1].item(),self.foot_positions[0,2,2].item(),
		#                             self.foot_positions[0,2,0].item(), end_effector_ref[0,7].item() + self.foot_positions[0,2,1].item(), self.foot_positions[0,2,2].item()], [0.9, 0.95, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,3,0].item(), self.foot_positions[0,3,1].item(),self.foot_positions[0,3,2].item(),
		#                             self.foot_positions[0,3,0].item(), end_effector_ref[0,10].item() + self.foot_positions[0,3,1].item(), self.foot_positions[0,3,2].item()], [0.9, 0.95, 0.5])
		# self.gym.clear_lines(self.viewer)

		return torch.exp(-40*end_effector_error)
	
	def _reward_imitate_base_end_effector_pos_world(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_ref_wf = df_imit.iloc[index_array,40:52].to_numpy()
		end_effector_ref_wf = end_effector_ref_wf.reshape(self.num_envs, 12)

		# default_base_pos = self.default_base_pos[:,0:3].cpu().numpy()
		# default_base_pos = default_base_pos.reshape(self.num_envs, 3)

		init_pos = self.default_base_pos.clone().detach().cpu().numpy()
		init_pos[:,2] = 0
		#Translate the reference for each robot by adding the (x,y) of the default base position
		init_pos_footsteps_translation = np.concatenate((init_pos,init_pos,init_pos,init_pos), axis=1)
		init_pos_footsteps_translation = init_pos_footsteps_translation.reshape(self.num_envs, 12)
		end_effector_ref_wf = end_effector_ref_wf + init_pos_footsteps_translation

		end_effector_ref_wf = torch.from_numpy(end_effector_ref_wf).float().to(self.device)
		# print("LINEAR_TRANSFORM", linear_transform)
		self.gym.clear_lines(self.viewer) 

		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		end_effector_error = torch.sum(torch.square(end_effector_ref_wf - cur_foot_pos), dim=1)

		#plot reference w.r.t foot pos in x axis
		for i in range(4):
			self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,i*3].item(), cur_foot_pos[0,1+3*i].item(),cur_foot_pos[0,2+3*i].item(),
															   end_effector_ref_wf[0,3*i].item(), cur_foot_pos[0,1+3*i].item(), cur_foot_pos[0,2+3*i].item()], [0.95, 0.5, 0.5])

		for i in range(4):
			self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,i*3].item(), cur_foot_pos[0,1+3*i].item(),cur_foot_pos[0,2+3*i].item(),
																cur_foot_pos[0,i*3].item(), end_effector_ref_wf[0,1+3*i].item(), cur_foot_pos[0,2+3*i].item()], [0.9, 0.95, 0.5])
			
		for i in range(4):
			self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,i*3].item(), cur_foot_pos[0,1+3*i].item(),cur_foot_pos[0,2+3*i].item(),
																cur_foot_pos[0,i*3].item(), cur_foot_pos[0,1+3*i].item(), end_effector_ref_wf[0,2+3*i].item()], [1.0, 1.0, 1.0])
		return torch.exp(-10*end_effector_error)
		
	def _reward_imitate_foot_height(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_z1_ref = df_imit.iloc[index_array,42].to_numpy()
		end_effector_z2_ref = df_imit.iloc[index_array,45].to_numpy()
		end_effector_z3_ref = df_imit.iloc[index_array,48].to_numpy()
		end_effector_z4_ref = df_imit.iloc[index_array,51].to_numpy()

		end_effector_z1_ref = end_effector_z1_ref.reshape(self.num_envs, 1)
		end_effector_z2_ref = end_effector_z2_ref.reshape(self.num_envs, 1)
		end_effector_z3_ref = end_effector_z3_ref.reshape(self.num_envs, 1)
		end_effector_z4_ref = end_effector_z4_ref.reshape(self.num_envs, 1)

		end_effector_z1_ref = torch.from_numpy(end_effector_z1_ref).float().to(self.device)
		end_effector_z2_ref = torch.from_numpy(end_effector_z2_ref).float().to(self.device)
		end_effector_z3_ref = torch.from_numpy(end_effector_z3_ref).float().to(self.device)
		end_effector_z4_ref = torch.from_numpy(end_effector_z4_ref).float().to(self.device)
		
		# cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		# print("CUR_FOOT_POS", cur_foot_pos)
		# cur_foot_pos = cur_footsteps_translated.reshape(self.num_envs, 12)
		
		cur_foot_pos = self.foot_positions.reshape(self.num_envs, 12)
		foot_height_error = torch.sum(torch.square(end_effector_z1_ref - cur_foot_pos[:,2].unsqueeze(1)), dim=1) \
			+ torch.sum(torch.square(end_effector_z2_ref - cur_foot_pos[:,5].unsqueeze(1)), dim=1) + torch.sum(torch.square(end_effector_z3_ref - cur_foot_pos[:,8].unsqueeze(1)), dim=1) \
				  + torch.sum(torch.square(end_effector_z4_ref - cur_foot_pos[:,11].unsqueeze(1)), dim=1)

		# self.gym.clear_lines(self.viewer) 
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(),self.foot_positions[0,0,2].item(), 
		#                                             self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(), end_effector_z1_ref[0,0].item()], [0.1, 0.15, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,3].item(), cur_foot_pos[0,4].item(),cur_foot_pos[0,5].item(),
		#                                             cur_foot_pos[0,3].item(), cur_foot_pos[0,4].item(), end_effector_z2_ref[0,0].item()], [0.1, 0.15, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,6].item(), cur_foot_pos[0,7].item(),cur_foot_pos[0,8].item(),
		#                                             cur_foot_pos[0,6].item(), cur_foot_pos[0,7].item(), end_effector_z3_ref[0,0].item()], [0.1, 0.15, 0.5])
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,9].item(), cur_foot_pos[0,10].item(),cur_foot_pos[0,11].item(),
		#                                             cur_foot_pos[0,9].item(), cur_foot_pos[0,10].item(), end_effector_z4_ref[0,0].item()], [0.1, 0.15, 0.5])
		# return torch.exp(-15*foot_height_error)
		return torch.exp(-40*foot_height_error)


	def _reward_imitate_base_pos(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		#Get the reference base position
		base_pos_ref = df_imit.iloc[index_array,34:36].to_numpy()
		base_pos_ref = base_pos_ref.reshape(self.num_envs, 2) 

		#Add the initial base position to convert to world frame of each robot
		base_pos_ref = torch.from_numpy(base_pos_ref).float().to(self.device) + self.default_base_pos[:,0:2]

		base_pos_error = torch.sum(torch.square(base_pos_ref - self.base_pos[:,0:2]), dim=1)
		# base_pos_ref = base_pos_ref.detach().cpu().numpy()
		# self.base_pos = self.base_pos.detach().cpu().numpy()
		# print("BASE_POS", self.base_pos[0,0].item())
		# self.gym.clear_lines(self.viewer)
		# self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.base_pos[0,0].item(), self.base_pos[0,1].item(),self.base_pos[0,2].item(), base_pos_ref[0,0].item(),
		#                                     base_pos_ref[0,1].item(), self.base_pos[0,2].item()], [0.1, 0.85, 0.1])
		# self.gym.add_lines(self.viewer, self.envs[0], 1, [0,0,0,1,1,1], [0.1, 0.85, 0.1])
		return torch.exp(-4*base_pos_error)


	def _reward_imitate_quat(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		quat_ref = df_imit.iloc[index_array,36:40].to_numpy()
		quat_ref = quat_ref.reshape(self.num_envs, 4)
		quat_ref = torch.from_numpy(quat_ref).float().to(self.device)

		quat_error = torch.sum(torch.square(quat_ref - self.base_quat), dim=1)
		return torch.exp(-20*quat_error)
	
	def _reward_imitate_quat_penalty(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		quat_ref = df_imit.iloc[index_array,36:40].to_numpy()
		quat_ref = quat_ref.reshape(self.num_envs, 4)
		quat_ref = torch.from_numpy(quat_ref).float().to(self.device)

		quat_error = torch.sum(torch.square(quat_ref - self.base_quat), dim=1)
		return quat_error


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
	