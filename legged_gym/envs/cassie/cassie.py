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

df_imit = pd.read_csv(path_to_imitation_data, parse_dates=False)

class Cassie(LeggedRobot):
	def _compute_torques(self, actions):
		actions_scaled = actions * self.cfg.control.action_scale
		actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
		control_type = self.cfg.control.control_type

		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = self.df_imit.iloc[index_array,0:12].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)
		# Convert the array to a PyTorch tensor
		dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)

		if control_type == 'apex_torque':
			decap_factor = gamma_decap**(self.torque_ref_decay_factor/k_decap)
			torques = actions_scaled + decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
		
		elif control_type=="position":
			torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

		elif control_type == 'torque':
			torques = actions_scaled
		
		elif control_type == 'decap_position':
			torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel + gamma_decap**(self.torque_ref_decay_factor/k_decap)*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
		
		else:
			raise ValueError(f"Unknown control type: {control_type}")
		return torch.clip(torques, -self.torque_limits, self.torque_limits)
	
	def _reward_no_fly(self):
		contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
		single_contact = torch.sum(1.*contacts, dim=1)==1
		return 1.*single_contact
	
	def _reward_imitation_angles(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = df_imit.iloc[index_array,0:12].to_numpy()
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
		height = df_imit.iloc[index_array,15].to_numpy()
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height = height.reshape(self.num_envs, )
		height = torch.from_numpy(height).float().to(self.device)
		# print("BASE_HEIGHT", base_height.shape, "HEIGHT", height.shape)
		# height_error = torch.sum(torch.abs(base_height - height), dim=1)    
		height_error = torch.square(base_height - height)
		return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)
	
	def _reward_imitation_height_penalty(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		height = df_imit.iloc[index_array,15].to_numpy()
		# print("HEIGHT REFERENCE", height)
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height = height.reshape(self.num_envs, )
		height = torch.from_numpy(height).float().to(self.device)
		return torch.square(base_height - height)
	
	def _reward_imitate_end_effector_pos(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_ref = df_imit.iloc[index_array,16:22].to_numpy()
		end_effector_ref = end_effector_ref.reshape(self.num_envs, 6 )
		end_effector_ref = torch.from_numpy(end_effector_ref).float().to(self.device)

		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 2, 3, device=self.device)
		for i in range(2):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])
		footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 6)

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
	
	def _reward_imitate_foot_height(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_z1_ref = df_imit.iloc[index_array,18].to_numpy()
		end_effector_z2_ref = df_imit.iloc[index_array,21].to_numpy()

		end_effector_z1_ref = end_effector_z1_ref.reshape(self.num_envs, 1)
		end_effector_z2_ref = end_effector_z2_ref.reshape(self.num_envs, 1)

		end_effector_z1_ref = torch.from_numpy(end_effector_z1_ref).float().to(self.device)
		end_effector_z2_ref = torch.from_numpy(end_effector_z2_ref).float().to(self.device)
		# print("END_EFFECTOR_Z1_REF", end_effector_z1_ref[0])
		# print("END_EFFECTOR_Z2_REF", end_effector_z2_ref[0])
		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		cur_foot_pos = cur_footsteps_translated.reshape(self.num_envs, 6)
		# print("CUR_FOOT_POS", cur_foot_pos[0])

		foot_height_error = torch.sum(torch.square(end_effector_z1_ref - cur_foot_pos[:,2].unsqueeze(1)), dim=1) + torch.sum(torch.square(end_effector_z2_ref - cur_foot_pos[:,5].unsqueeze(1)), dim=1) 
		return torch.exp(-40*foot_height_error)