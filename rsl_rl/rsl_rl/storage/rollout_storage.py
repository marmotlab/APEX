import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories
import yaml
import os 

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	use_one_critic_ablation = config['use_one_critic_ablation']

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu',
                 num_reward_groups=1):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_reward_groups = num_reward_groups

        # Core tensors
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape,
                                                       device=self.device)
        else:
            self.privileged_observations = None
        # Create rewards with an extra dimension for reward groups
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_reward_groups, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO; note that if you have separate value predictions per reward group, 
        # then these tensors are extended similarly.
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # Assuming you now store value estimates for each reward group:
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_reward_groups, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_reward_groups, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # RNN-related tensors remain unchanged.
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition["observations"])
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition["critic_observations"])
        self.actions[self.step].copy_(transition["actions"])
        # For rewards, note that in multi–reward mode the rewards tensor may have extra dimensions.
        self.rewards[self.step].copy_(transition["rewards"].view(self.num_envs, -1))
        self.dones[self.step].copy_(transition["dones"].view(self.num_envs, 1))
        self.values[self.step].copy_(transition["values"])
        self.actions_log_prob[self.step].copy_(transition["actions_log_prob"].view(self.num_envs, 1))
        # If your transition dict includes these keys, update them:
        if "action_mean" in transition:
            self.mu[self.step].copy_(transition["action_mean"])
        if "action_sigma" in transition:
            self.sigma[self.step].copy_(transition["action_sigma"])
        self._save_hidden_states(transition.get("hidden_states", None))
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state to match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                                          for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                                          for i in range(len(hid_c))]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        Computes the returns and advantages for each reward group.
        Expects last_values to have shape (num_envs, num_reward_groups)
        and self.values to have shape (num_transitions, num_envs, num_reward_groups).
        """
        # Initialize advantage with zeros matching the reward groups
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # dones is (T, num_envs, 1) so we expand it to match reward groups
            next_not_terminal = 1.0 - self.dones[step].float().expand(-1, self.num_reward_groups)
            # Compute TD error per reward group
            delta = self.rewards[step] + next_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages for each reward group independently.
        # alpha = [0.5,0.5]
        # alpha = [0.7,0.3]
        if use_one_critic_ablation:
            alpha = [1.0,0.0]
        else:
            alpha = [0.5,0.5]


        # Initialize a buffer to accumulate the final multi-critic advantage.
        advantages_mc = torch.zeros(self.num_transitions_per_env, self.num_envs, device=self.device)

        # Compute and sum normalized advantages from each reward group.
        for g in range(self.num_reward_groups):
            adv_g = self.returns[:, :, g] - self.values[:, :, g]
            adv_mean_g = adv_g.mean()
            adv_std_g = adv_g.std() + 1e-8
            normalized_adv_g = (adv_g - adv_mean_g) / adv_std_g

            # Accumulate into the multi-critic advantage, scaled by alpha[g].
            advantages_mc += alpha[g] * normalized_adv_g

        # Finally store the aggregated advantage in self.advantages
        self.advantages = advantages_mc

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # You might also wish to compute separate statistics per reward group here.
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        # Note: values, returns, advantages now have extra dimension for reward groups.
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None)

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1,
                                                                                                                      0).contiguous()
                               for saved_hidden_states in self.saved_hidden_states_a]
                hid_c_batch = [saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1,
                                                                                                                      0).contiguous()
                               for saved_hidden_states in self.saved_hidden_states_c]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield (obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch,
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch),
                       masks_batch)

                first_traj = last_traj
