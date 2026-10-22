import torch
import torch.nn.functional as F

from lewm.models.action_encoder import ActionEncoder

def make_encoder():
    return ActionEncoder(input_dim=10, smoothed_dim=6, emb_dim=16, mlp_scale=4)

def test_output_shape():
    out = make_encoder()(torch.randn(3, 5, 10))
    assert out.shape == (3, 5, 16)

def test_conv_k1_equals_per_timestep_linear():
    enc = make_encoder()
    x = torch.randn(3, 5, 10)
    weight = enc.patch_embed.weight.squeeze(-1)     # [smoothed, input]
    smoothed = F.linear(x, weight, enc.patch_embed.bias)
    assert torch.allclose(enc(x), enc.embed(smoothed), atol=1e-6)

def test_timesteps_do_not_mix():
    enc = make_encoder().eval()
    x = torch.randn(1, 5, 10)
    x2 = x.clone()
    x[:, 2] = torch.randn(1, 10)        # change only timestep 2
    out, out2, = enc(x), enc(x2)
    keep = [0, 1, 3, 4]
    assert torch.allclose(out[:, keep], out2[:, keep], atol=1e=6)
    assert not torch.allclose(out[:, 2], out2[:, 2])
