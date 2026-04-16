import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFeatureFusion(nn.Module):
    def __init__(self, nela_dim=87, style_dim=10, trace_dim=128, hidden_dim=128, fusion_method='gating'):
        super().__init__()
        self.fusion_method = fusion_method

        def block(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )

        self.nela_proj = block(nela_dim)
        self.style_proj = block(style_dim)
        self.trace_proj = block(trace_dim)

        if fusion_method == 'concat':
            self.post = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            self.out_dim = hidden_dim

        elif fusion_method == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            self.out_dim = hidden_dim

        elif fusion_method == 'attention':
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.out_dim = hidden_dim

        elif fusion_method == 'gating':
            self.gate = nn.Linear(hidden_dim * 3, hidden_dim)
            self.out_dim = hidden_dim

    def forward(self, nela, style, trace):
        n = self.nela_proj(nela)
        s = self.style_proj(style)
        t = self.trace_proj(trace)

        if self.fusion_method == 'concat':
            return self.post(torch.cat([n, s, t], dim=-1))

        elif self.fusion_method == 'mlp':
            return self.mlp(torch.cat([n, s, t], dim=-1))

        elif self.fusion_method == 'attention':
            seq = torch.stack([n, s, t], dim=1)
            attn_out, _ = self.attn(seq, seq, seq)
            return attn_out.mean(dim=1)

        elif self.fusion_method == 'gating':
            concat = torch.cat([n, s, t], dim=-1)
            g = torch.sigmoid(self.gate(concat))
            return g * n + (1 - g) * (s + t) / 2


# Test the module
if __name__ == '__main__':
    batch_size = 8
    
    # Mock data dimensions based on standard extractor outputs
    dim_nela = 87 
    dim_style = 10 
    dim_trace = 128
    
    mock_nela = torch.randn(batch_size, dim_nela)
    mock_style = torch.randn(batch_size, dim_style)
    mock_trace = torch.randn(batch_size, dim_trace)

    print("=== Testing Fusion Strategies ===\\n")
    strategies = ['concat', 'mlp', 'attention', 'gating']
    
    for strategy in strategies:
        fusion_module = MultiFeatureFusion(
            nela_dim=dim_nela, 
            style_dim=dim_style, 
            trace_dim=dim_trace, 
            hidden_dim=256, 
            fusion_method=strategy
        )
        
        output = fusion_module(mock_nela, mock_style, mock_trace)
        print(f"Strategy: {strategy:10} | Output Shape: {output.shape} | Output Dim: {fusion_module.out_dim}")
