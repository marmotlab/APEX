from rsl_rl.modules.actor_critic import *


def build_mlp(input_dim, hidden_dims, activation, output_dim):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)
    return nn.Sequential(*layers)


class ActorCriticEncoder(ActorCritic):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        self.actor_encoder = build_mlp(mlp_input_dim_a - 68, actor_hidden_dims, activation, 30)
        self.actor = build_mlp(68 + 30, actor_hidden_dims, activation, num_actions)

        # Value function
        self.critic = build_mlp(mlp_input_dim_c, critic_hidden_dims, activation, 1)

        print(f"Actor Encoder MLP: {self.actor_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def update_distribution(self, observations):
        zt = self.actor_encoder(observations[:, 68:])
        zt = torch.cat((observations[:, :68], zt), dim=1)
        mean = self.actor(zt)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act_inference(self, observations):
        zt = self.actor_encoder(observations[:, 68:])
        zt = torch.cat((observations[:, :68], zt), dim=1)
        actions_mean = self.actor(zt)
        return actions_mean

