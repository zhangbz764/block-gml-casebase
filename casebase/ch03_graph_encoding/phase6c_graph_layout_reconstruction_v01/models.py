"""Two independent encoders, one decoder architecture; no layout bypass."""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class RelationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 64 output channels total; two heads concatenated, each width32.
        self.layer1 = GATConv(20, 32, heads=2, edge_dim=24, add_self_loops=True, fill_value=0.)
        self.layer2 = GATConv(64, 32, heads=2, edge_dim=24, add_self_loops=True, fill_value=0.)
        self.project = nn.Sequential(nn.Linear(64+30, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, node, block, batch, edge_index, edge_attributes):
        hidden = F.elu(self.layer1(node, edge_index, edge_attributes))
        hidden = F.elu(self.layer2(hidden, edge_index, edge_attributes))
        return self.project(torch.cat([global_mean_pool(hidden, batch, size=block.shape[0]), block], dim=-1))


class SetEncoder(nn.Module):
    def __init__(self, width=104):
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(20, width), nn.ELU(), nn.Linear(width, 64), nn.ELU())
        self.project = nn.Sequential(nn.Linear(64+30, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, node, block, batch):
        hidden = self.node_mlp(node)
        return self.project(torch.cat([global_mean_pool(hidden, batch, size=block.shape[0]), block], dim=-1))


class LayoutDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16*8*8, 16))
        self.queries = nn.Parameter(torch.randn(64, 16)*.02)
        self.head = nn.Sequential(nn.Linear(32+16+1+16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 7))

    def forward(self, z, boundary, counts):
        if z.shape[1] != 32 or boundary.shape[1:] != (32, 32): raise ValueError('Decoder expects z32 and boundary32x32')
        if bool(((counts < 1) | (counts > 64)).any()): raise ValueError('Invalid decoder count')
        condition = self.condition(boundary[:, None])
        common = torch.cat([z, condition, counts[:, None].to(z.dtype)/64], dim=-1)
        raw = self.head(torch.cat([common[:, None].expand(-1, 64, -1), self.queries[None].expand(z.shape[0], -1, -1)], dim=-1))
        short = F.softplus(raw[..., 3:4])+1e-6
        long = short+F.softplus(raw[..., 2:3])+1e-6
        axial = F.normalize(raw[..., 4:6]+raw.new_tensor([0., 1e-6]), dim=-1, eps=1e-8)
        output = torch.cat([raw[..., :2], long, short, axial, raw[..., 6:7]], dim=-1)
        mask = torch.arange(64, device=z.device)[None] < counts[:, None]
        return output, mask


class ReconstructionModel(nn.Module):
    def __init__(self, kind, set_width=104):
        super().__init__()
        if kind not in ('relation', 'set'): raise ValueError('Unknown model')
        self.kind = kind
        # Initialize decoder first: the same seed gives identical decoder starting weights for A/B.
        self.decoder = LayoutDecoder()
        self.encoder = RelationEncoder() if kind == 'relation' else SetEncoder(set_width)

    def forward(self, batch):
        if self.kind == 'relation':
            z = self.encoder(batch['node'], batch['block'], batch['batch'], batch['edge_index'], batch['edge'])
        else: z = self.encoder(batch['node'], batch['block'], batch['batch'])
        return self.decoder(z, batch['boundary'], batch['counts'])

    def parameter_counts(self):
        return {key: sum(p.numel() for p in module.parameters())
                for key, module in [('encoder', self.encoder), ('decoder', self.decoder), ('total', self)]}
