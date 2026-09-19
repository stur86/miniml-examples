"""Tests for the Mixture of Bigots tree.

The tree walks down its layers to find how likely a sample is to reach each
node.  :func:`leaf_path_masks` states the same thing in closed form, one row per
leaf, so it serves here as the reference the walk is checked against.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from miniml_examples.mob_tree import (
    MoBBinaryTree,
    leaf_path_masks,
    sigmoid_entropy_schedule,
)

DEPTHS = [1, 2, 3, 5]


def _tree(depth: int, n_in: int = 4, n_out: int = 3) -> MoBBinaryTree:
    tree = MoBBinaryTree(n_in, n_out, depth)
    tree.randomize(depth)
    return tree


def _inputs(n_samples: int = 20, n_in: int = 4) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.array(rng.normal(size=(n_samples, n_in)), dtype=jnp.float32)


def _mask_log_probs(split_logits: np.ndarray, layer: int) -> np.ndarray:
    """Compute a layer's log-probabilities the closed form way.

    A path contributes ``log(sigmoid(z)) = z - softplus(z)`` where it turns
    right, and ``log(sigmoid(-z)) = -softplus(z)`` where it turns left.
    """
    right_mask, path_mask = leaf_path_masks(layer)
    logits = split_logits[: 2**layer - 1]
    softplus = np.logaddexp(0.0, logits)

    return right_mask @ logits - path_mask @ softplus


@pytest.mark.parametrize("depth", DEPTHS)
def test_layer_walk_matches_masks(depth: int) -> None:
    tree = _tree(depth)
    X = _inputs()
    split_logits = tree._split_logits(X, tree._buffer)

    walked = tree._layer_log_probs(split_logits)
    assert len(walked) == depth

    for layer in range(1, depth + 1):
        expected = _mask_log_probs(np.array(split_logits, dtype=np.float64), layer)
        assert np.allclose(np.array(walked[layer - 1]), expected, atol=1e-5)


@pytest.mark.parametrize("depth", DEPTHS)
def test_layer_probs_are_normalized(depth: int) -> None:
    tree = _tree(depth)
    X = _inputs()

    for layer in range(1, depth + 1):
        probs = np.array(tree.layer_probs(X, layer))
        assert probs.shape == (len(X), 2**layer)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("depth", DEPTHS)
def test_layers_aggregate_onto_the_one_above(depth: int) -> None:
    """The two children of a node hold, together, what the node holds."""
    tree = _tree(depth)
    X = _inputs()

    for layer in range(2, depth + 1):
        above = np.array(tree.layer_probs(X, layer - 1))
        below = np.array(tree.layer_probs(X, layer))
        assert np.allclose(below[:, 0::2] + below[:, 1::2], above, atol=1e-5)


def test_layer_walk_stops_where_asked() -> None:
    tree = _tree(4)
    split_logits = tree._split_logits(_inputs(), tree._buffer)

    assert len(tree._layer_log_probs(split_logits, down_to=2)) == 2


@pytest.mark.parametrize("depth", DEPTHS)
def test_hard_leaves_follow_the_preferred_branch(depth: int) -> None:
    """The walked path must be the one a plain loop over the nodes takes."""
    tree = _tree(depth)
    X = _inputs()
    split_weights = np.array(tree._split_weights())
    split_biases = np.array(tree._split_biases())

    expected = []
    for x in np.array(X):
        node = 0
        for _ in range(depth):
            node = 2 * node + 1 + int(split_weights[node] @ x + split_biases[node] > 0)
        expected.append(node - (tree.num_leaves - 1))

    assert np.array_equal(np.array(tree._hard_leaves(X)), np.array(expected))


@pytest.mark.parametrize("depth", DEPTHS)
def test_predict_tree_returns_the_reached_leaf(depth: int) -> None:
    tree = _tree(depth)
    X = _inputs()

    leaf_log_p = np.array(tree.predict_tree(X))
    assert leaf_log_p.shape == (len(X), tree.n_out)
    assert np.allclose(np.exp(leaf_log_p).sum(axis=1), 1.0, atol=1e-5)

    leaves = np.array(tree._hard_leaves(X))
    logits = np.array(tree._leaf_weights())
    expected = logits[leaves] - np.log(np.exp(logits[leaves]).sum(axis=1, keepdims=True))
    assert np.allclose(leaf_log_p, expected, atol=1e-5)


@pytest.mark.parametrize("depth", DEPTHS)
def test_biases_shift_the_splits(depth: int) -> None:
    """A large enough bias must send every sample the same way."""
    tree = _tree(depth)
    X = _inputs()

    params = tree.get_params()
    assert params["_split_biases.v"].shape == (2**depth - 1,)

    before = np.array(tree.layer_probs(X, 1))
    params["_split_biases.v"] = jnp.full((2**depth - 1,), 1e3)
    tree.set_params(params)
    after = np.array(tree.layer_probs(X, 1))

    assert not np.allclose(before, after)
    # Everyone now takes the right branch out of the root
    assert np.allclose(after[:, 1], 1.0, atol=1e-5)
    assert np.array_equal(
        np.array(tree._hard_leaves(X)), np.full(len(X), 2**depth - 1)
    )


def test_entropy_loss_only_in_training_mode() -> None:
    from miniml import PredictMode
    from miniml.model import PredictKernelOutput

    tree = _tree(3)
    X = _inputs()

    # No weights set: nothing to add, in either mode
    for mode in PredictMode:
        assert not isinstance(
            tree._predict_kernel(X, tree._buffer, mode=mode), PredictKernelOutput
        )

    tree.set_entropy_weights([1.0, 1.0, 1.0])
    trained = tree._predict_kernel(X, tree._buffer, mode=PredictMode.TRAINING)
    inferred = tree._predict_kernel(X, tree._buffer, mode=PredictMode.INFERENCE)

    assert isinstance(trained, PredictKernelOutput)
    assert float(trained.activity_loss) > 0.0
    assert not isinstance(inferred, PredictKernelOutput)
    assert np.allclose(np.array(trained.y_pred), np.array(inferred), atol=1e-6)


def test_entropy_loss_sums_the_weighted_layers() -> None:
    from miniml import PredictMode

    tree = _tree(3)
    X = _inputs()
    weights = [0.0, 0.5, 2.0]
    tree.set_entropy_weights(weights)

    expected = 0.0
    for layer, weight in enumerate(weights, start=1):
        probs = np.array(tree.layer_probs(X, layer))
        expected += weight * -np.sum(probs * np.log(probs))

    loss = tree._predict_kernel(X, tree._buffer, mode=PredictMode.TRAINING).activity_loss
    assert np.isclose(float(loss), expected, rtol=1e-4)


def test_entropy_weights_are_validated() -> None:
    tree = _tree(3)

    assert np.array_equal(tree.entropy_weights, np.zeros(3))
    with pytest.raises(ValueError):
        tree.set_entropy_weights([1.0, 1.0])
    with pytest.raises(ValueError):
        tree.set_entropy_weights(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        tree.set_entropy_weights([1.0, np.nan, 1.0])

    # A failed call must leave the weights as they were
    assert np.array_equal(tree.entropy_weights, np.zeros(3))


def test_entropy_weights_getter_is_a_copy() -> None:
    tree = _tree(3)
    tree.entropy_weights[0] = 1.0

    assert tree.entropy_weights[0] == 0.0


def test_depth_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MoBBinaryTree(4, 3, 0)
    with pytest.raises(ValueError):
        leaf_path_masks(0)


def test_layer_out_of_range() -> None:
    tree = _tree(3)
    X = _inputs()

    for layer in (0, 4):
        with pytest.raises(ValueError):
            tree.layer_probs(X, layer)


def test_unbound_model_is_refused() -> None:
    from miniml import MiniMLError

    tree = MoBBinaryTree(4, 3, 3)
    X = _inputs()

    with pytest.raises(MiniMLError):
        tree.layer_probs(X)
    with pytest.raises(MiniMLError):
        tree.predict_tree(X)


def test_sigmoid_schedule_front_walks_down() -> None:
    schedule = sigmoid_entropy_schedule(3, scale=8.0)
    assert schedule.shape == (5, 3)

    # Every layer is squeezed harder as the stages pass
    assert np.all(np.diff(schedule, axis=0) > 0)
    # And at any stage, the layers near the root are squeezed the hardest
    assert np.all(np.diff(schedule, axis=1) < 0)
    assert np.all((schedule >= 0) & (schedule <= 8.0))

    with pytest.raises(ValueError):
        sigmoid_entropy_schedule(0)
    with pytest.raises(ValueError):
        sigmoid_entropy_schedule(3, n_stages=0)
