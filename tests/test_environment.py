"""Environment smoke tests.

Not placeholders. These assert the things the design documents claim about
the environment, so that a mismatch fails here with a clear message rather
than surfacing forty minutes into a training run.
"""

import sys

import pytest
import torch

import minigpt


def test_python_is_at_least_3_11():
    """pyproject requires >=3.11. The system Python on the dev machine is 3.9."""
    assert sys.version_info >= (3, 11), f"got {sys.version_info}"


def test_torch_is_at_least_2_14():
    """The throughput numbers in DESIGN section 5.1 were measured on 2.14.0."""
    major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 14), f"got {torch.__version__}"


def test_package_imports():
    assert minigpt.__version__


def test_a_tensor_survives_a_round_trip():
    """Cheapest possible proof that torch is actually functional, not just importable."""
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.equal(x.T.T, x)
    assert x.sum().item() == pytest.approx(15.0)


def test_ops_the_model_depends_on_exist():
    """Guards the ops the day-one spike probed on MPS.

    CI runs on CPU, so this checks availability rather than device behaviour —
    a missing or renamed op fails here instead of inside the model.
    """
    q = torch.randn(1, 2, 8, 4)
    assert (
        torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True).shape == q.shape
    )
    assert torch.nn.functional.layer_norm(q, (4,)).shape == q.shape
    assert torch.nn.functional.gelu(q, approximate="tanh").shape == q.shape


def test_causal_mask_construction_is_lower_triangular():
    """The masking convention the model will rely on, pinned before the model exists.

    True means attendable. Position 0 sees only itself; the last position sees
    everything. Getting this inverted is the classic silent bug.
    """
    T = 4
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[-1].all()
    assert not mask[0, 1:].any()
