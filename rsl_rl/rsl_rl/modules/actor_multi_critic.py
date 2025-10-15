import torch
import torch.nn as nn
from torch.distributions import Normal

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class MultiCriticActorCritic(nn.Module):
    """
    Actor-Critic network with one actor and multiple critic heads.
    Each critic head is intended to learn the value for one reward group.
    """
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_critics=2,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super(MultiCriticActorCritic, self).__init__()

        activation_fn = get_activation(activation)

        # ----- Actor network -----
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # ----- Multi-critic networks -----
        self.num_critics = num_critics
        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            critic_layers = []
            critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic_layers.append(activation_fn)
            for i in range(len(critic_hidden_dims)):
                if i == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                    critic_layers.append(activation_fn)
            self.critics.append(nn.Sequential(*critic_layers))

        # ----- Action noise -----
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # self.std = init_noise_std * torch.ones(num_actions, dtype=torch.float, device='cuda:0', requires_grad=False)
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """
        Returns the value predictions from each critic head.
        Expected output shape: (batch_size, num_critics) which should match the number of reward groups.
        """
        values = [critic(critic_observations) for critic in self.critics]
        values = torch.cat(values, dim=1)  # Do not aggregate (i.e. no mean)
        return values

    def evaluate_all(self, critic_observations, **kwargs):
        # This is equivalent to evaluate() in this design.
        values = [critic(critic_observations) for critic in self.critics]
        values = torch.cat(values, dim=1)
        return values

    def reset(self, dones=None):
        pass
