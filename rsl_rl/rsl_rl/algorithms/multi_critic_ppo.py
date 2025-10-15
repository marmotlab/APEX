import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, MultiCriticActorCritic  # in this context, use your multi–critic actor
from rsl_rl.storage import RolloutStorage

class MultiCriticPPO:
    def __init__(self,
                 actor_critic,  # should be an instance of MultiCriticActorCritic
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu'):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic.to(self.device)
        self.storage = None  # will be initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = None  # transition will be created in act()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_reward_groups=1):
        from rsl_rl.storage import RolloutStorage
        self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                      actor_obs_shape, critic_obs_shape, action_shape,
                                      device=self.device, num_reward_groups=num_reward_groups)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        actions = self.actor_critic.act(obs).detach()
        values = self.actor_critic.evaluate(critic_obs).detach()  # now returns shape (batch_size, num_reward_groups)
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        action_mean = self.actor_critic.action_mean.detach()
        action_sigma = self.actor_critic.action_std.detach()
        transition = {
            'observations': obs,
            'critic_observations': critic_obs,
            'actions': actions,
            'values': values,
            'actions_log_prob': actions_log_prob,
            'action_mean': action_mean,
            'action_sigma': action_sigma
        }
        return actions, transition

    def process_env_step(self, transition, rewards, dones, infos):
        # If rewards come in as a single dimension, consider unsqueezing to match (num_envs, num_reward_groups)
        # Otherwise, assume rewards are already multi-dimensional.
        transition['rewards'] = rewards.clone()
        transition['dones'] = dones
        self.storage.add_transitions(transition)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()  # shape (num_envs, num_reward_groups)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch_all = self.actor_critic.evaluate_all(critic_obs_batch, masks=masks_batch,
                                                             hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Compute the surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # if ratio.dim() == 1:
            #     ratio = ratio.unsqueeze(1)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Here target_values_batch and returns_batch have shape (batch_size, num_reward_groups)
            value_loss_total = 0
            num_heads = value_batch_all.shape[1]
            for i in range(num_heads):
                value_batch = value_batch_all[:, i:i + 1]  # shape: (batch_size, 1)
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch[:, i:i + 1] + (value_batch - target_values_batch[:, i:i + 1]).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch[:, i:i + 1]).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch[:, i:i + 1]).pow(2)
                    value_loss_i = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss_i = (returns_batch[:, i:i + 1] - value_batch).pow(2).mean()
                value_loss_total += value_loss_i
            value_loss = value_loss_total / num_heads

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
