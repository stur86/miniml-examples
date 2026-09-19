"""Mixture of Bigots (MoB): a differentiable binary decision tree.

The model is a full binary tree of fixed depth, trained end to end by gradient
descent.

* Each internal node holds a weight vector $w_n$.  For an input $x$ it sends the
  sample to its right child with probability $\\sigma(w_n \\cdot x)$ and to its
  left child with probability $1-\\sigma(w_n \\cdot x)$, where $\\sigma$ is the
  logistic function.  The split is therefore soft: every sample reaches every
  leaf with some probability.
* Each leaf holds a fixed set of class logits, which do not depend on the input.
  A leaf is thus a *bigot*: an expert with one opinion that it gives no matter
  what it is asked.  The tree is a mixture of experts in which the gating
  network is the tree and the experts are the bigots, hence "Mixture of Bigots".

The probability that a sample reaches leaf $l$ is the product of the branch
probabilities along the root-to-leaf path:

$$
p(l|x) = \\prod_{n \\in \\mathrm{path}(l)} \\sigma(s_{n,l}\\, w_n \\cdot x)
$$

with $s_{n,l}=+1$ if the path turns right at node $n$ and $-1$ if it turns left.
The prediction is the mixture of the leaf distributions weighted by those path
probabilities:

$$
p(c|x) = \\sum_l p(l|x)\\, p(c|l)
$$

Everything is computed in log space, so the two products above become sums over
path nodes and a `logsumexp` over leaves.  The two sums are expressed as matrix
products with the constant masks built by :func:`leaf_path_masks`, which lets
the whole tree be evaluated in a few dense operations with no Python recursion.

The model outputs log-probabilities and is trained with
:class:`miniml.loss.CrossEntropyLogLoss`.

Reference: Frosst & Hinton, *Distilling a Neural Network Into a Soft Decision
Tree* (2017), https://arxiv.org/abs/1711.09784
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JXArray
from miniml import MiniMLModel, MiniMLParam, PredictMode
from miniml.loss import CrossEntropyLogLoss, LNormRegularization

__all__ = ["leaf_path_masks", "MoBBinaryTree"]


def leaf_path_masks(depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the constant masks that map node quantities onto leaves.

    Internal nodes are numbered in level order, as in a heap: the root is 0 and
    the children of node $n$ are $2n+1$ (left) and $2n+2$ (right).  Leaves are
    numbered left to right, so that the bits of a leaf index, most significant
    first, are the sequence of branch choices that reach it.

    Args:
        depth (int): Depth of the tree.  A tree of depth ``d`` has ``2**d``
            leaves and ``2**d - 1`` internal nodes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two ``(2**depth, 2**depth - 1)`` float
        arrays.  ``right_mask[l, n]`` is 1 if the path to leaf ``l`` goes
        through node ``n`` and turns right there.  ``path_mask[l, n]`` is 1 if
        the path to leaf ``l`` goes through node ``n`` at all, in either
        direction.

    Raises:
        ValueError: If ``depth`` is smaller than 1.
    """
    if depth < 1:
        raise ValueError("Tree depth must be at least 1")

    n_leaves = 2**depth
    n_nodes = n_leaves - 1

    right_mask = np.zeros((n_leaves, n_nodes))
    path_mask = np.zeros((n_leaves, n_nodes))

    for leaf in range(n_leaves):
        node = 0
        for level in range(depth):
            goes_right = (leaf >> (depth - 1 - level)) & 1
            path_mask[leaf, node] = 1.0
            right_mask[leaf, node] = goes_right
            node = 2 * node + 1 + goes_right

    return right_mask, path_mask


class MoBBinaryTree(MiniMLModel):
    """A Mixture of Bigots binary decision tree classifier.

    The model has two parameters:

    * ``_split_weights``, of shape ``(2**depth - 1, n_in)``: one weight vector
      per internal node.  It carries an L1 regularization loss, which pushes
      the splits to use few input features each, as a hard decision tree does.
    * ``_leaf_weights``, of shape ``(2**depth, n_out)``: the class logits of
      each leaf.  It carries a weaker L2 regularization loss, which keeps the
      leaf distributions from saturating early in the fit.

    Both are visible under those names in ``get_params()`` and ``set_params()``,
    as ``"_split_weights.v"`` and ``"_leaf_weights.v"``.

    Note that the splits have no bias term, so every split hyperplane passes
    through the origin.  Center the inputs, or append a constant feature, if
    that matters for the data at hand.

    Example:
        >>> tree = MoBBinaryTree(n_in=4, n_out=3, depth=3)
        >>> tree.randomize(5)
        >>> tree.fit(X_train, y_train).success
        True
        >>> log_p = tree.predict(X_test)  # log-probabilities, (n_samples, 3)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        depth: int,
        split_reg_scale: float = 1.0,
        leaf_reg_scale: float = 1e-2,
    ) -> None:
        """Construct a Mixture of Bigots tree.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output classes.
            depth (int): Depth of the tree.  The number of parameters grows as
                ``2**depth``, so keep it small.
            split_reg_scale (float, optional): Scale of the L1 regularization
                loss on the node weights. Defaults to 1.0.
            leaf_reg_scale (float, optional): Scale of the L2 regularization
                loss on the leaf logits. Defaults to 1e-2.

        Raises:
            ValueError: If ``depth`` is smaller than 1.
        """
        self._depth = depth
        self._n_in = n_in
        self._n_out = n_out
        self._num_leaves = 2**depth

        right_mask, path_mask = leaf_path_masks(depth)

        self._split_weights = MiniMLParam(
            shape=(self._num_leaves - 1, n_in),
            reg_loss=LNormRegularization(1),
            reg_scale=split_reg_scale,
        )
        self._leaf_weights = MiniMLParam(
            shape=(self._num_leaves, n_out),
            reg_loss=LNormRegularization(2),
            reg_scale=leaf_reg_scale,
        )

        super().__init__(loss=CrossEntropyLogLoss())

        self._right_mask = jnp.array(right_mask, dtype=self._dtype)
        self._path_mask = jnp.array(path_mask, dtype=self._dtype)

    @property
    def depth(self) -> int:
        """The depth of the tree."""
        return self._depth

    @property
    def num_leaves(self) -> int:
        """The number of leaves, namely of bigots, in the tree."""
        return self._num_leaves

    @property
    def n_in(self) -> int:
        """The number of input features."""
        return self._n_in

    @property
    def n_out(self) -> int:
        """The number of output classes."""
        return self._n_out

    def _leaf_log_probs(self, X: JXArray, buffer: JXArray) -> JXArray:
        """Compute the log-probability that each sample reaches each leaf.

        Args:
            X (JXArray): Input data, of shape ``(n_samples, n_in)``.
            buffer (JXArray): Parameter buffer.

        Returns:
            JXArray: Log-probabilities, of shape ``(n_leaves, n_samples)``.
            Each column sums (in probability) to one.
        """
        # Logit of the right branch at each node, for each sample
        split_logits = self._split_weights(buffer) @ X.T

        # log(sigmoid(z)) = z - softplus(z) for a right turn, and
        # log(sigmoid(-z)) = -softplus(z) for a left one.  Summing over the
        # nodes of each path gives both cases at once: add z on right turns
        # only, subtract softplus(z) on every node of the path.
        return self._right_mask @ split_logits - self._path_mask @ jax.nn.softplus(
            split_logits
        )

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs: Any,
    ) -> JXArray:
        """Predict the class log-probabilities of each sample.

        Args:
            X (JXArray): Input data, of shape ``(n_samples, n_in)``.
            buffer (JXArray): Parameter buffer.
            rng_key (JXArray, optional): Unused; the model is deterministic.
            mode (PredictMode, optional): Unused; training and inference are
                identical.
            **predict_kwargs: Unused.

        Returns:
            JXArray: Class log-probabilities, of shape ``(n_samples, n_out)``.
        """
        # (n_leaves, n_samples)
        leaf_log_p = self._leaf_log_probs(X, buffer)
        # (n_leaves, n_out)
        class_log_p = jax.nn.log_softmax(self._leaf_weights(buffer), axis=-1)

        # log(sum_leaves p(leaf|x) * p(class|leaf)), summed over the leaf axis
        return jax.scipy.special.logsumexp(
            leaf_log_p.T[:, :, None] + class_log_p[None, :, :], axis=1
        )
