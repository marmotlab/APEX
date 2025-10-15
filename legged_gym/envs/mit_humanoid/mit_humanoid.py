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

class Humanoid(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_imitation_angles(self):
        index_array = self.imitation_index.detach().cpu().numpy().astype(int)
        # Retrieve the corresponding rows from df_imit using array indexing
        dof_imit_arr = df_imit.iloc[index_array,0:10].to_numpy()
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
        height = df_imit.iloc[index_array,13].to_numpy()
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
        height = df_imit.iloc[index_array,13].to_numpy()
        # print("HEIGHT REFERENCE", height)
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height = height.reshape(self.num_envs, )
        height = torch.from_numpy(height).float().to(self.device)
        return torch.square(base_height - height)
    
    def _reward_imitate_end_effector_pos(self):
        index_array = self.imitation_index.detach().cpu().numpy().astype(int)
        end_effector_ref = df_imit.iloc[index_array,14:20].to_numpy()
        end_effector_ref = end_effector_ref.reshape(self.num_envs, self.num_actions )
        end_effector_ref = torch.from_numpy(end_effector_ref).float().to(self.device)

        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 6, 3, device=self.device)
        for i in range(6):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])
        footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, self.num_actions)

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
        end_effector_z1_ref = df_imit.iloc[index_array,16].to_numpy()
        end_effector_z2_ref = df_imit.iloc[index_array,19].to_numpy()
        # end_effector_z3_ref = df_imit.iloc[index_array,30].to_numpy()
        # end_effector_z4_ref = df_imit.iloc[index_array,33].to_numpy()
        # end_effector_z5_ref = df_imit.iloc[index_array,36].to_numpy()
        # end_effector_z6_ref = df_imit.iloc[index_array,39].to_numpy()

        end_effector_z1_ref = end_effector_z1_ref.reshape(self.num_envs, 1)
        end_effector_z2_ref = end_effector_z2_ref.reshape(self.num_envs, 1)
        # end_effector_z3_ref = end_effector_z3_ref.reshape(self.num_envs, 1)
        # end_effector_z4_ref = end_effector_z4_ref.reshape(self.num_envs, 1)
        # end_effector_z5_ref = end_effector_z5_ref.reshape(self.num_envs, 1)
        # end_effector_z6_ref = end_effector_z6_ref.reshape(self.num_envs, 1)

        end_effector_z1_ref = torch.from_numpy(end_effector_z1_ref).float().to(self.device)
        end_effector_z2_ref = torch.from_numpy(end_effector_z2_ref).float().to(self.device)
        # end_effector_z3_ref = torch.from_numpy(end_effector_z3_ref).float().to(self.device)
        # end_effector_z4_ref = torch.from_numpy(end_effector_z4_ref).float().to(self.device)
        # end_effector_z5_ref = torch.from_numpy(end_effector_z5_ref).float().to(self.device)
        # end_effector_z6_ref = torch.from_numpy(end_effector_z6_ref).float().to(self.device)

        cur_foot_pos = self.foot_positions.reshape(self.num_envs, self.num_actions)
        foot_height_error = torch.sum(torch.square(end_effector_z1_ref - cur_foot_pos[:,2].unsqueeze(1)), dim=1) \
            + torch.sum(torch.square(end_effector_z2_ref - cur_foot_pos[:,5].unsqueeze(1)), dim=1) 

        # self.gym.clear_lines(self.viewer) 
        # self.gym.add_lines(self.viewer, self.envs[0], 1 , [self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(),self.foot_positions[0,0,2].item(), 
        #                                             self.foot_positions[0,0,0].item(), self.foot_positions[0,0,1].item(), end_effector_z1_ref[0,0].item()], [0.1, 0.15, 0.5])
        # self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,3].item(), cur_foot_pos[0,4].item(),cur_foot_pos[0,5].item(),
        #                                             cur_foot_pos[0,3].item(), cur_foot_pos[0,4].item(), end_effector_z2_ref[0,0].item()], [0.1, 0.15, 0.5])
        # self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,6].item(), cur_foot_pos[0,7].item(),cur_foot_pos[0,8].item(),
        #                                             cur_foot_pos[0,6].item(), cur_foot_pos[0,7].item(), end_effector_z3_ref[0,0].item()], [0.1, 0.15, 0.5])
        # self.gym.add_lines(self.viewer, self.envs[0], 1 , [cur_foot_pos[0,9].item(), cur_foot_pos[0,10].item(),cur_foot_pos[0,11].item(),
                                                    # cur_foot_pos[0,9].item(), cur_foot_pos[0,10].item(), end_effector_z4_ref[0,0].item()], [0.1, 0.15, 0.5])
        # return torch.exp(-15*foot_height_error)
        return torch.exp(-40*foot_height_error)
    
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip
    