# tests/test_gpt.py

import torch
import pytest

from model.gpt import GPT, MLP, GPTConfig, Block, CausalSelfAttention

@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=128,
        context_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        bias=False
    )

def test_mlp_shape(small_config):
    mlp = MLP(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    y = mlp(x)
    assert y.shape == (2, 16, small_config.d_model)

def test_attention_shape(small_config):
    attn = CausalSelfAttention(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    y = attn(x)
    assert y.shape == (2, 16, small_config.d_model)

def test_block_shape(small_config):
    block = Block(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    y = block(x)
    assert y.shape == (2, 16, small_config.d_model)

def test_gpt_forward_pass(small_config):
    model = GPT(small_config)
    idx = torch.randint(0, small_config.vocab_size, (2, 16))
    logits, loss = model(idx)
    
    assert logits.shape == (2, 16, small_config.vocab_size)
    assert loss is None

def test_gpt_loss(small_config):
    model = GPT(small_config)
    idx = torch.randint(0, small_config.vocab_size, (2, 16))
    targets = torch.randint(0, small_config.vocab_size, (2, 16))
    logits, loss = model(idx, targets)
    
    assert loss is not None
    assert loss.item() > 0

def test_weight_tying(small_config):
    model = GPT(small_config)
    # Weights of wte and lm_head should be the same object
    assert model.transformer.wte.weight is model.lm_head.weight

def test_num_parameters(small_config):
    model = GPT(small_config)
    total_params = model.num_parameters()
    params_no_emb = model.num_parameters(exclude_embeddings=True)
    
    assert total_params > params_no_emb
    assert params_no_emb > 0

def test_context_length_error(small_config):
    model = GPT(small_config)
    # Trigger assertion by exceeding context length
    idx = torch.randint(0, small_config.vocab_size, (1, small_config.context_length + 1))
    with pytest.raises(AssertionError) as excinfo:
        model(idx)
    assert "exceeds context length" in str(excinfo.value)

def test_initial_loss_is_near_log_vocab(small_config):
    """At init, loss should be ≈ log(vocab_size) — random prediction baseline."""
    torch.manual_seed(42)
    model = GPT(small_config)
    idx = torch.randint(0, small_config.vocab_size, (4, 32))
    targets = torch.randint(0, small_config.vocab_size, (4, 32))
    _, loss = model(idx, targets)
    expected = torch.log(torch.tensor(float(small_config.vocab_size)))
    assert abs(loss.item() - expected.item()) < 0.5, (
        f"Initial loss {loss.item():.3f} too far from log(vocab)={expected.item():.3f} "
        f"— weight init is broken"
    )

def test_loss_decreases_after_one_step(small_config):
    torch.manual_seed(42)
    model = GPT(small_config)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, device_type="cpu"
    )
    idx = torch.randint(0, small_config.vocab_size, (4, 32))
    targets = torch.randint(0, small_config.vocab_size, (4, 32))
    
    _, loss_before = model(idx, targets)
    loss_before.backward()
    optimizer.step()
    optimizer.zero_grad()

    _, loss_after = model(idx, targets)
    assert loss_after.item() < loss_before.item(), (
        f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
    )