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
The same masks, built for a shallower tree, give the distribution over the nodes
of any intermediate layer.

The model outputs log-probabilities and is trained with
:class:`miniml.loss.CrossEntropyLogLoss`.  It also supports an optional
entropy activity loss, described in
:meth:`MoBBinaryTree.set_entropy_weights`, which squeezes the soft tree towards
a hard one.

Reference: Frosst & Hinton, *Distilling a Neural Network Into a Soft Decision
Tree* (2017), https://arxiv.org/abs/1711.09784
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JXArray
from miniml import MiniMLError, MiniMLModel, MiniMLParam, PredictMode
from miniml.loss import CrossEntropyLogLoss, LNormRegularization
from numpy.typing import ArrayLike

__all__ = ["leaf_path_masks", "sigmoid_entropy_schedule", "MoBBinaryTree"]


def leaf_path_masks(depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the constant masks that map node quantities onto leaves.

    Internal nodes are numbered in level order, as in a heap: the root is 0 and
    the children of node $n$ are $2n+1$ (left) and $2n+2$ (right).  Leaves are
    numbered left to right, so that the bits of a leaf index, most significant
    first, are the sequence of branch choices that reach it.

    Because of that numbering, the masks of a tree of depth ``d`` also describe
    layer ``d`` of any deeper tree: its first ``2**d - 1`` nodes are exactly the
    nodes above that layer.

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


def sigmoid_entropy_schedule(
    depth: int,
    n_stages: int | None = None,
    scale: float = 1.0,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Build a schedule of entropy weights that squeezes the tree layer by
    layer, from the root down.

    Stage $s$ gives layer $l$ the weight

    $$
    \\lambda_l(s) = \\frac{w}{1+\\exp\\left[-k\\,(c_s-l)\\right]}
    $$

    a logistic step in the layer index, centered on a front $c_s$ that advances
    one layer at a time.  Layers above the front are squeezed, layers below it
    are still free to stay soft, and the sharpness $k$ sets how abrupt the
    border between the two is.  The first stage leaves the whole tree nearly
    free, and by the last one the front has passed every layer.

    The schedule is meant to be walked one stage per fit::

        for weights in sigmoid_entropy_schedule(tree.depth):
            tree.set_entropy_weights(weights)
            tree.fit(X, y)

    Args:
        depth (int): Depth of the tree the schedule is for.
        n_stages (int, optional): Number of stages, namely of fits.  Defaults
            to ``depth + 2``, which moves the front by one layer per stage.
        scale (float, optional): The weight a fully squeezed layer tends to.
            It must be large enough for the first stage to move the objective,
            or that fit stops before it takes a step, as
            :meth:`MoBBinaryTree.set_entropy_weights` describes. Defaults to
            1.0.
        sharpness (float, optional): Steepness of the front, in inverse layers.
            Large values squeeze one layer at a time, small ones squeeze the
            whole tree at once, more and more strongly. Defaults to 2.0.

    Returns:
        np.ndarray: The weights, of shape ``(n_stages, depth)``.  Each row is
        one argument for :meth:`MoBBinaryTree.set_entropy_weights`.

    Raises:
        ValueError: If ``depth`` or ``n_stages`` is smaller than 1.
    """
    if depth < 1:
        raise ValueError("Tree depth must be at least 1")
    if n_stages is None:
        n_stages = depth + 2
    elif n_stages < 1:
        raise ValueError("A schedule must have at least one stage")

    # The front starts above the first layer and ends below the last one, so
    # that the tree is soft at the start and squeezed throughout at the end
    fronts = np.linspace(0.0, depth + 1.0, n_stages)
    layers = np.arange(1, depth + 1)

    return scale / (1.0 + np.exp(-sharpness * (fronts[:, None] - layers[None, :])))


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

    Layers are numbered by the number of splits above them: layer 1 holds the
    two children of the root and layer ``depth`` holds the leaves.  The
    per-layer entropy weights set by :meth:`set_entropy_weights` are indexed in
    the same order.

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
        if depth < 1:
            raise ValueError("Tree depth must be at least 1")

        self._depth = depth
        self._n_in = n_in
        self._n_out = n_out
        self._num_leaves = 2**depth

        # One pair of masks per layer; layer l is at index l-1
        layer_masks = [leaf_path_masks(layer) for layer in range(1, depth + 1)]

        # No entropy loss until the weights are set
        self._entropy_weights = np.zeros(depth)

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

        self._layer_masks = [
            (
                jnp.array(right_mask, dtype=self._dtype),
                jnp.array(path_mask, dtype=self._dtype),
            )
            for right_mask, path_mask in layer_masks
        ]

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

    @property
    def entropy_weights(self) -> np.ndarray:
        """A copy of the per-layer entropy weights of the activity loss."""
        return self._entropy_weights.copy()

    def set_entropy_weights(self, weights: ArrayLike | Sequence[float]) -> None:
        """Set the per-layer weights of the entropy activity loss.

        During training only, the model adds an activity loss to the objective:

        $$
        \\mathcal{L}_a = \\sum_{l=1}^{d} \\lambda_l
            \\sum_{x} H\\left[p(\\cdot|x, l)\\right]
        $$

        where $p(\\cdot|x, l)$ is the distribution of sample $x$ over the nodes
        of layer $l$ and $H$ is its entropy in nats.  The entropy is summed over
        samples, as the cross-entropy loss is, so the balance between the two
        does not depend on the size of the training set.  MiniML scales the
        whole term by the ``active_reg_lambda`` argument of ``fit()``.

        A positive weight penalizes a sample that spreads over many nodes of
        that layer, and so squeezes the tree towards a hard one, in which each
        sample follows a single path.  Weights are zero by default, and the
        activity loss is then not computed at all.

        Setting the weights between fits is how a training schedule is built:
        fit with soft splits first, so that the gradients are informative, then
        raise the weights layer by layer and fit again to disentangle the
        paths, starting from the layers near the root.  Note that squeezing a
        layer also squeezes every layer above it, since the path to a node runs
        through them.

        Take steps large enough to move the objective: a fit that restarts from
        a converged point with a barely changed objective can stop before it
        takes a single step, since the buffer is single precision, and reports
        a line search failure.

        The weights are a property of the training run, not of the model, so
        they are not stored by ``save()``.

        Args:
            weights (ArrayLike | Sequence[float]): One weight per layer, of
                length ``depth``, ordered from the layer below the root to the
                leaves.

        Raises:
            ValueError: If the weights are not a one-dimensional array of
                length ``depth``, or if any of them is not finite.
        """
        weights = np.asarray(weights, dtype=float)

        if weights.shape != (self._depth,):
            raise ValueError(
                f"Entropy weights must have shape ({self._depth},), one per "
                f"layer, but have shape {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Entropy weights must all be finite")

        self._entropy_weights = weights

    def layer_probs(self, X: JXArray, layer: int | None = None) -> JXArray:
        """Compute the probability that each sample reaches each node of a
        layer.

        Meant for inspecting a fitted tree: how sharp these distributions are
        tells how close the tree is to a hard one.

        Args:
            X (JXArray): Input data, of shape ``(n_samples, n_in)``.
            layer (int, optional): The layer to compute the distribution of,
                from 1 (the children of the root) to ``depth``.  Defaults to
                the leaves.

        Returns:
            JXArray: Probabilities, of shape ``(n_samples, 2**layer)``, each
            row summing to one.

        Raises:
            MiniMLError: If the model is not bound to a parameter buffer.
            ValueError: If ``layer`` is out of range.
        """
        if not self.bound:
            raise MiniMLError("Model parameters have not been bound to buffers")

        if layer is None:
            layer = self._depth
        elif not 1 <= layer <= self._depth:
            raise ValueError(
                f"Layer must be between 1 and the depth {self._depth}, not {layer}"
            )

        split_logits = self._split_logits(X, self._buffer)

        return jnp.exp(self._layer_log_probs(split_logits, layer)).T

    def _split_logits(self, X: JXArray, buffer: JXArray) -> JXArray:
        """Compute the logit of the right branch at each node, per sample.

        Args:
            X (JXArray): Input data, of shape ``(n_samples, n_in)``.
            buffer (JXArray): Parameter buffer.

        Returns:
            JXArray: Logits, of shape ``(n_nodes, n_samples)``.
        """
        return self._split_weights(buffer) @ X.T

    def _layer_log_probs(self, split_logits: JXArray, layer: int) -> JXArray:
        """Compute the log-probability that each sample reaches each node of a
        layer.

        Args:
            split_logits (JXArray): Node logits, as returned by
                :meth:`_split_logits`.
            layer (int): The layer to compute the distribution of, from 1 (the
                children of the root) to ``depth`` (the leaves).

        Returns:
            JXArray: Log-probabilities, of shape ``(2**layer, n_samples)``.
            Each column sums (in probability) to one.
        """
        right_mask, path_mask = self._layer_masks[layer - 1]

        # Only the nodes above this layer play a part, and the heap numbering
        # puts them first
        logits = split_logits[: 2**layer - 1]

        # log(sigmoid(z)) = z - softplus(z) for a right turn, and
        # log(sigmoid(-z)) = -softplus(z) for a left one.  Summing over the
        # nodes of each path gives both cases at once: add z on right turns
        # only, subtract softplus(z) on every node of the path.
        return right_mask @ logits - path_mask @ jax.nn.softplus(logits)

    def _entropy_loss(self, split_logits: JXArray) -> JXArray | None:
        """Compute the weighted sum of the layer entropies.

        Layers whose weight is zero are skipped, so a model with no weights set
        costs nothing.

        Args:
            split_logits (JXArray): Node logits, as returned by
                :meth:`_split_logits`.

        Returns:
            JXArray | None: The activity loss, or None if every weight is zero.
        """
        loss: JXArray | None = None

        for layer, weight in enumerate(self._entropy_weights, start=1):
            if weight == 0:
                continue
            log_p = self._layer_log_probs(split_logits, layer)
            # Entropy in nats, summed over samples
            entropy = -jnp.sum(jnp.exp(log_p) * log_p)
            term = weight * entropy
            loss = term if loss is None else loss + term

        return loss

    def _predict_kernel(
        self,
        X: JXArray,
        buffer: JXArray,
        rng_key: JXArray | None = None,
        mode: PredictMode = PredictMode.INFERENCE,
        **predict_kwargs: Any,
    ) -> Any:
        """Predict the class log-probabilities of each sample.

        Args:
            X (JXArray): Input data, of shape ``(n_samples, n_in)``.
            buffer (JXArray): Parameter buffer.
            rng_key (JXArray, optional): Unused; the model is deterministic.
            mode (PredictMode, optional): Prediction mode.  The entropy
                activity loss is computed in training mode only.
            **predict_kwargs: Unused.

        Returns:
            Any: Class log-probabilities of shape ``(n_samples, n_out)``, on
            their own or wrapped in a ``PredictKernelOutput`` together with the
            entropy activity loss.
        """
        split_logits = self._split_logits(X, buffer)

        # (n_leaves, n_samples)
        leaf_log_p = self._layer_log_probs(split_logits, self._depth)
        # (n_leaves, n_out)
        class_log_p = jax.nn.log_softmax(self._leaf_weights(buffer), axis=-1)

        # log(sum_leaves p(leaf|x) * p(class|leaf)), summed over the leaf axis
        y_pred = jax.scipy.special.logsumexp(
            leaf_log_p.T[:, :, None] + class_log_p[None, :, :], axis=1
        )

        activity_loss = (
            self._entropy_loss(split_logits)
            if mode == PredictMode.TRAINING
            else None
        )

        return self._with_activity_loss(y_pred, activity_loss)
