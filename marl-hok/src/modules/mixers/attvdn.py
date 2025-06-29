from re import M
import torch as th
import torch.nn as nn
import numpy as np


class ATTVDNMixer(nn.Module):
    def __init__(self, args):
        super(ATTVDNMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        single_agent_dim = self.input_dim // self.n_agents
        dq = 5

        self.embeding = nn.Embedding(6, dq)
        self.query_map = nn.Linear(single_agent_dim, dq)

        self.state_constant = th.nn.Sequential(
            nn.Linear(self.state_dim, self.n_agents),
            nn.ReLU(),
            nn.Linear(self.n_agents, 1)
        )

        self.state_weight = th.nn.Sequential(
            nn.Linear(self.state_dim, self.n_agents),
            nn.ReLU(),
            nn.Linear(self.n_agents, self.n_agents),
        )

        self.pos_embed = th.zeros(self.n_agents, dq)
        for t in range(self.n_agents):
            for i in range(0, dq, 2):
                self.pos_embed[t][i] = np.sin(t / (10000 ** (i / dq)))
                if i + 1 < dq:
                    self.pos_embed[t][i + 1] = np.cos(t / (10000 ** (i / dq)))

        self.dq = dq

    def forward(self, agent_qs, state, actions):
        # agent_qs: [B, T, N_agents]
        # State Shape [B, T, S]
        # Action Shape [B, T, N_agents, 1]
        transformed_state = state.view(state.size(0), state.size(1), self.n_agents, -1) # [B, T, N_agents, S]
        squeezed_actions = actions.squeeze(-1) # [B, T, N_agents]
        transformed_actions = self._actions_transformer(squeezed_actions) # [B, T, N_agents]

        query = self.query_map(transformed_state) + self.pos_embed[None, None, ...]  # [b, t, N_agents, dq]
        key = self.embeding(transformed_actions)# [b, t, N_agents, dq]

        matrix = key @ query.transpose(-1, -2) / np.sqrt(self.dq)  # [b, t, N_agents, N_agents]
        weights = th.softmax(matrix, dim=-1)
        weights = th.sum(weights, dim=-2) # [b, t, N_agents]

        weights = th.clamp(weights, min=0.25) # Ensure the weights are not too small

        # Record the weights for debugging
        # with open('attvdn_weights.txt', 'a') as f:
        #     f.write(str(weights[0][0].cpu().detach().numpy().tolist()) + '\n')

        agent_star = agent_qs * weights # [b, t, N_agents]
        return th.sum(agent_star, dim=2, keepdim=True)  # [b, t, 1]
    
    def _actions_transformer(self, actions):
        # actions: [B, T, N_agents]
        # return: [B, T, N_agents]
        # Turn actions range from [0, 1, \dots, 12]
        # to [0, 1, 2, 3, 4, 5]
        # [0, 1, 2, 3, 9, 10, 11, 12] -> 0
        # [4, 5, 6, 7, 8] -> [1, 2, 3, 4, 5]

        actions = actions.clone()
        conditions = (actions >= 4) & (actions <= 8)
        return th.where(conditions, actions - 3, 0)