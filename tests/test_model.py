import pytest
import torch

from src.model import (
    MultiScaleTemporalBlock,
    TCNBranch,
    TCNMSEMARGAT,
    lead_graph,
)


CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


@pytest.mark.parametrize("samples", [256, 512, 1024])
def test_tcn_branch_returns_finite_embedding_and_single_attention_map(
    samples: int,
) -> None:
    branch = TCNBranch(CHANNELS, hidden=32, levels=2, dropout=0.0)

    representation = branch(torch.randn(2, len(CHANNELS), samples))
    attention = branch.last_temporal_attention

    assert representation.shape == (2, 32)
    assert torch.isfinite(representation).all()
    assert attention is not None
    assert attention.shape == (2, 1, samples)
    assert torch.isfinite(attention).all()
    assert torch.allclose(
        attention.sum(dim=-1), torch.ones(2, 1), atol=1e-4
    )


def test_temporal_block_uses_three_full_convolution_scales_and_batch_norm() -> None:
    block = MultiScaleTemporalBlock(hidden=32, dilation=4, dropout=0.0)

    branch_convolutions = [branch[0] for branch in block.branches]
    branch_normalizations = [
        module
        for branch in block.branches
        for module in branch
        if isinstance(module, torch.nn.BatchNorm1d)
    ]

    assert [layer.kernel_size for layer in branch_convolutions] == [
        (3,),
        (7,),
        (15,),
    ]
    assert all(layer.in_channels == 32 for layer in branch_convolutions)
    assert all(layer.out_channels == 16 for layer in branch_convolutions)
    assert all(layer.groups == 1 for layer in branch_convolutions)
    assert all(layer.dilation == (4,) for layer in branch_convolutions)
    assert len(branch_normalizations) == 3
    assert isinstance(block.fusion[1], torch.nn.BatchNorm1d)


def test_tcn_branch_uses_v2_dilation_schedule() -> None:
    branch = TCNBranch(CHANNELS, hidden=32, levels=5, dropout=0.0)

    dilations = [block.branches[0][0].dilation for block in branch.blocks]

    assert dilations == [(1,), (2,), (4,), (8,), (16,)]


def test_tcn_branch_uses_batch_norm_without_group_norm_or_local_graph() -> None:
    branch = TCNBranch(CHANNELS, hidden=32, levels=2, dropout=0.0)

    assert any(isinstance(module, torch.nn.BatchNorm1d) for module in branch.modules())
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in branch.modules())
    assert not any(name.startswith("local_") for name, _ in branch.named_modules())
    assert not hasattr(branch, "local_adjacency")


def test_temporal_block_uses_recommended_default_branch_dimension() -> None:
    block = MultiScaleTemporalBlock(hidden=128, dilation=1, dropout=0.0)

    assert all(branch[0].out_channels == 48 for branch in block.branches)
    assert block.fusion[0].in_channels == 144
    assert block.fusion[0].out_channels == 128


def test_tcn_branch_propagates_gradients_through_encoder_and_attention() -> None:
    branch = TCNBranch(CHANNELS, hidden=32, levels=2, dropout=0.0)

    branch(torch.randn(2, len(CHANNELS), 256)).sum().backward()

    assert branch.blocks[0].branches[0][0].weight.grad is not None
    assert branch.pooling.attention.weight.grad is not None


@pytest.mark.parametrize("samples", [256, 512, 1024])
def test_model_returns_two_logits_for_each_window_length(samples: int) -> None:
    model = TCNMSEMARGAT(
        CHANNELS, hidden=32, node_dim=16, tcn_levels=2, heads=4, dropout=0.0
    )
    inputs = torch.randn(2, 18, samples)

    logits = model(inputs)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_model_supports_one_backward_step() -> None:
    model = TCNMSEMARGAT(
        CHANNELS, hidden=32, node_dim=16, tcn_levels=2, heads=4, dropout=0.0
    )
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(2, 18, 256)), torch.tensor([0, 1])
    )

    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model_rejects_wrong_channel_count() -> None:
    model = TCNMSEMARGAT(CHANNELS, hidden=32, node_dim=16, tcn_levels=2, heads=4)

    with pytest.raises(ValueError, match="18 channels"):
        model(torch.randn(2, 17, 256))


def test_lead_graph_preserves_shared_electrode_orientation() -> None:
    adjacency, orientation = lead_graph(CHANNELS)

    assert adjacency[0, 1].item() == 1
    assert orientation[0, 1].item() == -1
    assert adjacency[0, 8].item() == 0
