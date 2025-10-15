import os
import statistics
import time
from collections import deque

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
import yaml


from rsl_rl.algorithms import PPO
from rsl_rl.algorithms import MultiCriticPPO  # ensure you import the modified version
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticEncoder, ActorCritic, MultiCriticActorCritic

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        # Create the multi–critic actor; note that you can also pass a desired number of reward groups.
        actor_critic_class = eval(self.cfg["policy_class_name"])  # e.g. MultiCriticActorCritic
        actor_critic = actor_critic_class(self.env.num_obs,
                                          num_critic_obs,
                                          self.env.num_actions,
                                          **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # e.g. MultiCriticPPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # If your config includes number of reward groups, pass it to the rollout storage.
        num_reward_groups = self.cfg["critic_num"]
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs],
                              [self.env.num_actions],
                              num_reward_groups=num_reward_groups)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()    

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

            self.wbd_writer = wandb.init(project=self.cfg['experiment_name'],
                                         name=self.cfg['run_name'],
                                         config=self.cfg,
                                         save_code=True)
            
            artifact = wandb.Artifact('param_config', type='config')
            artifact.add_file('legged_gym/envs/param_config.yaml')
            self.wbd_writer.log_artifact(artifact)

            artifact = wandb.Artifact('robot_config_code', type='code')
            artifact.add_file('legged_gym/envs/go2/go2_config.py')
            self.wbd_writer.log_artifact(artifact)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        decapbuffer = deque(maxlen=100)
        
        # RMSE buffers for iteration-level logging
        rmse_buffers = {
            'joint_pos_rmse': deque(maxlen=1000),
            'imitation_height_rmse': deque(maxlen=1000),
            'end_effector_pos_rmse': deque(maxlen=1000),
            'quaternion_rmse': deque(maxlen=1000),
            'lin_vel_tracking_rmse': deque(maxlen=1000),
            'ang_vel_tracking_rmse': deque(maxlen=1000)
        }

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        decapbuffer = deque(maxlen=100)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Unpack the tuple: actions and the transition dictionary
                    actions, transition = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Process the environment step, storing the multi–reward transition.
                    self.alg.process_env_step(transition, rewards, dones, infos)

                    # For logging, aggregate rewards if there are multiple reward groups.
                    if rewards.dim() > 1 and rewards.shape[1] > 1:
                        reward_agg = rewards.sum(dim=-1)  # This yields a 1D tensor of shape [num_env]
                    else:
                        reward_agg = rewards
                    # Bookkeeping for episode rewards
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    
                    # Accumulate RMSE errors for iteration-level logging
                    if 'step' in infos:
                        for key, value in infos['step'].items():
                            if 'rmse' in key.lower():
                                # Map the long names to shorter buffer keys
                                if 'joint_pos' in key:
                                    rmse_buffers['joint_pos_rmse'].append(value)
                                elif 'height' in key:
                                    rmse_buffers['imitation_height_rmse'].append(value)
                                elif 'end_effector' in key:
                                    rmse_buffers['end_effector_pos_rmse'].append(value)
                                elif 'quaternion' in key:
                                    rmse_buffers['quaternion_rmse'].append(value)
                                elif 'lin_vel_tracking' in key:
                                    rmse_buffers['lin_vel_tracking_rmse'].append(value)
                                elif 'ang_vel_tracking' in key:
                                    rmse_buffers['ang_vel_tracking_rmse'].append(value)
                    
                    if 'decap_factor' in infos:
                            decapbuffer.append(infos['decap_factor'])
                    cur_reward_sum += reward_agg
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # After each training iteration, increment the global_training_iteration in the environment
            if hasattr(self.env, "global_training_iteration"):
                self.env.global_training_iteration += 1
            elif hasattr(self.env, "envs"):  # If using VecEnv wrapper
                for env in self.env.envs:
                    if hasattr(env, "global_training_iteration"):
                        env.global_training_iteration += 1
            if self.log_dir is not None:
                self.log(locals(), decapbuffer=decapbuffer, rmse_buffers=rmse_buffers)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35, decapbuffer=None, rmse_buffers=None):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                self.wbd_writer.log({'Episode/' + key: value}, step=locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        
        # Log RMSE errors at iteration level
        rmse_log_dict = {}
        if rmse_buffers is not None:
            for key, buffer in rmse_buffers.items():
                if len(buffer) > 0:
                    mean_rmse = statistics.mean(buffer)
                    self.writer.add_scalar(f'RMSE/{key}', mean_rmse, locs['it'])
                    rmse_log_dict[f'RMSE/{key}'] = mean_rmse
        
        # Wandb logging for losses and RMSE together
        wandb_log_dict = {
            'Loss/value_function': locs['mean_value_loss'],
            'Loss/surrogate': locs['mean_surrogate_loss'],
            'Loss/learning_rate': self.alg.learning_rate,
            'Policy/mean_noise_std': mean_std.item(),
            'Perf/total_fps': fps,
            'Perf/collection time': locs['collection_time'],
            'Perf/learning_time': locs['learn_time']
        }
        wandb_log_dict.update(rmse_log_dict)
        self.wbd_writer.log(wandb_log_dict, step=locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.wbd_writer.log({'Train/mean_reward': statistics.mean(locs['rewbuffer']),
                                 'Train/mean_episode_length': statistics.mean(locs['lenbuffer']),
                                 'Train/mean_reward/time': statistics.mean(locs['rewbuffer']),
                                 'Train/mean_episode_length/time': statistics.mean(locs['lenbuffer'])}, step=locs['it'])
        # --- Decap factor logging ---
        if decapbuffer is not None and len(decapbuffer) > 0:
            mean_decap = statistics.mean(decapbuffer)
            self.writer.add_scalar('Train/decap_factor_env0', mean_decap, locs['it'])
            self.wbd_writer.log({'Train/decap_factor_env0': mean_decap}, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
            if decapbuffer is not None and len(decapbuffer) > 0:
                log_string += f"{'Decap factor (env 0):':>{pad}} {mean_decap:.4f}\n"
            # Add RMSE to console output
            if rmse_buffers is not None:
                for key, buffer in rmse_buffers.items():
                    if len(buffer) > 0:
                        mean_rmse = statistics.mean(buffer)
                        display_name = key.replace('_', ' ').title()
                        log_string += f"{f'{display_name}:':>{pad}} {mean_rmse:.4f}\n"
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
