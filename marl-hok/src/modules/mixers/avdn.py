import torch as th
import torch.nn as nn
import numpy as np


class AVDNMixer(nn.Module):
    def __init__(self, args):
        super(AVDNMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        self.embed_module = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.n_agents),
            nn.Softmax(dim=-1)
        )

    def forward(self, agent_qs, batch):
        weights = self.embed_module(batch) # weight [b, t, n_agents]
        agent_qs = weights * agent_qs * 5.0  # [b, t, n_agents]
        # Record the weights for debugging
        # with open('weights.txt', 'a') as f:
        #     f.write(str(weights[0][0].cpu().detach().numpy().tolist()) + '\n')
        return th.sum(agent_qs, dim=2, keepdim=True)