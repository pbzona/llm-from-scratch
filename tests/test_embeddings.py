"""Checks for token lookup and position embedding behavior."""

import torch


def test_toy_embedding_is_a_trainable_lookup():
    torch.manual_seed(123)
    layer = torch.nn.Embedding(6, 3)
    input_ids = torch.tensor([2, 3, 5, 1])
    embeddings = layer(input_ids)
    assert embeddings.shape == (4, 3)
    torch.testing.assert_close(embeddings, layer.weight[input_ids])
    torch.testing.assert_close(
        layer(torch.tensor([3])),
        torch.tensor([[-0.4015, 0.9666, -1.1481]]),
        atol=5e-5,
        rtol=0,
    )
    embeddings.sum().backward()
    assert layer.weight.grad is not None
    assert torch.equal(layer.weight.grad[0], torch.zeros(3))
    assert torch.equal(layer.weight.grad[3], torch.ones(3))


def test_position_embeddings_broadcast_over_batches():
    torch.manual_seed(123)
    inputs = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 1, 1, 1],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
            [4, 4, 4, 4],
            [5, 0, 5, 0],
            [1, 3, 5, 0],
            [0, 2, 4, 1],
        ]
    )
    token_layer = torch.nn.Embedding(6, 5)
    position_layer = torch.nn.Embedding(4, 5)
    tokens = token_layer(inputs)
    positions = position_layer(torch.arange(4))
    combined = tokens + positions
    assert tokens.shape == combined.shape == (8, 4, 5)
    assert positions.shape == (4, 5)
    for batch in range(8):
        torch.testing.assert_close(combined[batch], tokens[batch] + positions)

    repeated_ids = torch.tensor([[0, 0, 0, 0]])
    repeated = token_layer(repeated_ids)
    assert torch.equal(repeated[0, 0], repeated[0, 1])
    assert not torch.equal((repeated + positions)[0, 0], (repeated + positions)[0, 1])

    combined.sum().backward()
    assert token_layer.weight.grad is not None
    assert position_layer.weight.grad is not None
    torch.testing.assert_close(position_layer.weight.grad, torch.full((4, 5), 8.0))
