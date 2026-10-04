import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Mixture of Bigots tree",
    css_file="../../theme/notebook.css",
)


@app.cell
def _():
    import logging

    # JAX warns when it sees a GPU but has no CUDA support; the CPU is fine here
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from miniml_examples.dataset import sklearn_classification_data
    from miniml_examples.mob_tree import MoBBinaryTree, sigmoid_entropy_schedule
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    return (
        LinearSegmentedColormap,
        MoBBinaryTree,
        datasets,
        mo,
        np,
        plt,
        sigmoid_entropy_schedule,
        sklearn_classification_data,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Mixture of Bigots tree

    A *soft* binary decision tree, trained by gradient descent on the Iris
    dataset. Companion to the blog post
    [*Decision Trees With Gradient Descent*](https://stur86.github.io/s-plus-plus/posts/decision-trees-with-gradient-descent/).

    Each passing sample merely has a *probability* of going left or right at
    each node $O_i$. By convention the logit of going left is $0$ and the
    logit of going right is $r_i = \mathbf{v}_i\cdot\mathbf{x}+b_i$, so

    $$
    P_i(\mathrm{left}) = \frac{1}{1+e^{r_i}} \qquad
    P_i(\mathrm{right}) = \frac{e^{r_i}}{1+e^{r_i}}
    $$

    Probabilities multiply along a path, so each leaf $L_j$ is reached with
    probability

    $$
    P(L_j) = \prod_{i \in \mathrm{path}(j)} \frac{e^{z_i}}{1+e^{r_i}}
    $$

    where $z_i$ is $0$ or $r_i$ depending on which way the path goes. Each
    leaf holds a "bigot": a constant class distribution that it gives no
    matter what it is asked. The output is the mixture of the bigots,
    weighted by the leaf probabilities, which makes this a "Mixture of
    Bigots" (Frosst & Hinton, [*Distilling a Neural Network Into a Soft
    Decision Tree*](https://arxiv.org/abs/1711.09784)). Here we use depth
    $D = 3$, so eight bigots for three classes.
    """)
    return


@app.cell
def _(datasets, np, sklearn_classification_data, train_test_split):
    iris = datasets.load_iris()
    class_names = list(iris.target_names)
    X, y = sklearn_classification_data(iris)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 80% train, 20% test
        shuffle=True,  # Shuffles/scrambles the dataset (True by default)
        random_state=42,  # Seed for reproducible randomness
        stratify=y,  # Preserves class balance across splits
    )
    # Integer labels of the test set, for plotting and scoring
    test_labels = np.argmax(y_test, axis=1)
    return X_test, X_train, class_names, test_labels, y_train


@app.cell
def _(LinearSegmentedColormap, np, plt):
    # Dark figures on a transparent background, to sit on the dark page theme.
    # Brighter means more probable.
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "axes.edgecolor": "#3a4757",
        "xtick.color": "#9aa7b4",
        "ytick.color": "#9aa7b4",
        "axes.titlecolor": "#e6edf3",
    })
    CMAP = LinearSegmentedColormap.from_list("p", ["#16202b", "#1f5fd6", "#8fd0ff"])
    EDGE = "#5b6b7d"
    TEXT = "#e6edf3"

    def ink(value):
        """Text colour that reads well on CMAP(value)."""
        return "#0e141b" if value > 0.7 else TEXT

    def draw_tree(ax, layers, leaf_labels=None, title=None):
        """Draw a horizontal tree, each node coloured by the probability of
        reaching it.

        Args:
            ax: Matplotlib axes.
            layers: One array of node probabilities per layer, root first.
            leaf_labels: Optional text to write to the right of each leaf.
            title: Optional axes title.
        """
        depth = len(layers) - 1
        pos = {
            (d, i): (d, 1 - (i + 0.5) / 2**d)
            for d in range(depth + 1)
            for i in range(2**d)
        }
        for d in range(depth):
            for i in range(2**d):
                for child in (2 * i, 2 * i + 1):
                    flow = layers[d + 1][child]
                    (x0, y0), (x1, y1) = pos[d, i], pos[d + 1, child]
                    ax.plot(
                        [x0, x1], [y0, y1], color=CMAP(0.2 + 0.8 * flow),
                        lw=0.6 + 6 * flow, solid_capstyle="round", zorder=1,
                    )
        for (d, i), (x, y) in pos.items():
            p = layers[d][i]
            leaf = d == depth
            ax.scatter(
                x, y, s=420 if leaf else 480, marker="s" if leaf else "o",
                c=[CMAP(p)], edgecolors=EDGE, linewidths=1.0, zorder=2,
            )
            ax.text(
                x, y, f"{p:.2f}", ha="center", va="center", fontsize=7,
                color=ink(p), zorder=3,
            )
            if leaf and leaf_labels is not None:
                ax.text(x + 0.25, y, leaf_labels[i], ha="left", va="center", fontsize=8, color=TEXT)
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlim(-0.3, depth + (1.1 if leaf_labels is not None else 0.3))
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

    def entropy(p):
        """Mean entropy, in nats, of the rows of a probability matrix."""
        return float(np.mean(-np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)))

    return CMAP, draw_tree, entropy, ink


@app.cell
def _(MoBBinaryTree, X_test, X_train, mo, np, test_labels, y_train):
    mobt = MoBBinaryTree(n_in=4, n_out=3, depth=3)
    mobt.randomize(5)
    with mo.status.spinner(title="Fitting the soft tree..."):
        soft_fit_ok = mobt.fit(X_train, y_train).success

    # Snapshot of the soft fit, taken before the squeezing below changes the model
    soft_accuracy = float(np.mean(np.argmax(mobt.predict(X_test), axis=1) == test_labels))
    return mobt, soft_accuracy, soft_fit_ok


@app.cell
def _(mo, soft_accuracy, soft_fit_ok):
    mo.md(f"""
    ## The soft fit

    We first train the tree with plain cross-entropy loss and no squeezing.
    The fit {"converged" if soft_fit_ok else "**did not converge**"}, and the
    soft tree classifies **{soft_accuracy:.2%}** of the test set correctly.
    The bigots it learned are below: the bigot matrix $B$ has one column per
    leaf, and each column is the class distribution of that leaf.
    """)
    return


@app.cell
def _(CMAP, class_names, ink, mobt, np, plt):
    def _bigot_matrix():
        _logits = np.asarray(mobt.get_params()["_leaf_weights.v"])
        _p = np.exp(_logits - _logits.max(axis=1, keepdims=True))
        # (n_classes, n_leaves): one column per bigot
        return (_p / _p.sum(axis=1, keepdims=True)).T

    soft_bigots = _bigot_matrix()

    _fig, _ax = plt.subplots(figsize=(7, 2.4))
    _ax.imshow(soft_bigots, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for (_c, _l), _v in np.ndenumerate(soft_bigots):
        _ax.text(_l, _c, f"{_v:.2f}", ha="center", va="center", fontsize=8,
                 color=ink(_v))
    _ax.set_xticks(range(mobt.num_leaves), [f"$L_{_l}$" for _l in range(mobt.num_leaves)])
    _ax.set_yticks(range(len(class_names)), class_names)
    _ax.set_title("Bigot matrix $B$ after the soft fit", fontsize=11)
    _fig.tight_layout()
    _ax
    return (soft_bigots,)


@app.cell
def _(mo):
    mo.md(r"""
    ## How to grow a real tree

    The fit above most likely relies on soft mixing between *all* the leaves.
    To use the model *as a tree*, each sample must flow through a single
    path. Call $\mathbf{p}_d$ the probability distribution of a sample over
    the $2^d$ nodes of layer $d$; in a "classic" tree it is a vector of
    zeroes with a single entry set to one, namely a very low entropy
    distribution. So we add an entropy regularization term to the loss:

    $$
    \mathcal{L} = -\sum_i y_i\log(\hat{y}_i) - \sum_{d=1}^D \lambda_d \sum_{j=1}^{2^d} p_d^{(j)}\log(p_d^{(j)})
    $$

    with a per-layer regularization strength $\lambda_d$. Imposing this
    right away would risk "freezing" the model in an ineffectual shape, so
    we "squeeze" the tree gradually instead. The schedule starts with all
    $\lambda$ at zero (the fit above), then increases them from the top
    down, all the way to the leaves: a logistic front that moves one layer
    per stage, with one fit per stage.
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    class_names,
    entropy,
    mo,
    mobt,
    np,
    sigmoid_entropy_schedule,
    soft_bigots,
    test_labels,
    y_train,
):
    def _snapshot(label, lambdas, bigots):
        """State of the tree on the test set at one point of the schedule."""
        _layers = [np.array(mobt.layer_probs(X_test, _d)) for _d in range(1, mobt.depth + 1)]
        _pred = np.argmax(mobt.predict(X_test), axis=1)
        return {
            "label": label,
            "lambdas": np.asarray(lambdas),
            "layers": _layers,
            "entropies": [entropy(_p) for _p in _layers],
            "predictions": _pred,
            "accuracy": float(np.mean(_pred == test_labels)),
            "leaf_classes": [class_names[_c] for _c in np.argmax(bigots, axis=0)],
        }

    def _bigots():
        _logits = np.asarray(mobt.get_params()["_leaf_weights.v"])
        _p = np.exp(_logits - _logits.max(axis=1, keepdims=True))
        return (_p / _p.sum(axis=1, keepdims=True)).T

    # Depends on soft_bigots, so that it runs after the soft fit has been read
    history = [_snapshot("soft fit", np.zeros(mobt.depth), soft_bigots)]

    # One fit per stage, each starting from the fit of the stage before
    _schedule = sigmoid_entropy_schedule(mobt.depth, scale=32.0)
    for _stage, _weights in enumerate(
        mo.status.progress_bar(
            _schedule, title="Squeezing the tree", subtitle="one fit per stage",
            show_eta=True, show_rate=False,
        ),
        start=1,
    ):
        mobt.set_entropy_weights(_weights)
        mobt.fit(X_train, y_train)
        history.append(_snapshot(f"stage {_stage}", _weights, _bigots()))
    return (history,)


@app.cell
def _(CMAP, history, mo, np, plt):
    _stages = np.arange(len(history))
    _labels = [_h["label"] for _h in history]
    _lambdas = np.array([_h["lambdas"] for _h in history])
    _entropies = np.array([_h["entropies"] for _h in history])
    _depth = _lambdas.shape[1]
    _colors = [CMAP(0.35 + 0.6 * _d / max(_depth - 1, 1)) for _d in range(_depth)]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for _d in range(_depth):
        _ax1.plot(_stages, _lambdas[:, _d], "o-", color=_colors[_d], label=f"$\\lambda_{_d + 1}$")
        _ax2.plot(_stages, _entropies[:, _d], "o-", color=_colors[_d], label=f"layer {_d + 1}")
    _ax1.set_title("Entropy weights $\\lambda_d$", fontsize=11)
    _ax2.set_title("Mean entropy of $\\mathbf{p}_d$ on the test set (nats)", fontsize=11)
    for _ax in (_ax1, _ax2):
        _ax.set_xticks(_stages, _labels, rotation=30, ha="right", fontsize=8)
        _ax.legend(frameon=False, fontsize=8)
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {_h['label']} | {' / '.join(f'{_e:.4f}' for _e in _h['entropies'])} | {_h['accuracy']:.2%} |"
        for _h in history
    )
    mo.vstack([
        _ax1,
        mo.md(
            "Entropy goes down layer by layer, from the root to the leaves. "
            + (
                f"Test accuracy stays at {history[0]['accuracy']:.2%} throughout."
                if history[0]["accuracy"] == history[-1]["accuracy"]
                else f"Test accuracy goes from {history[0]['accuracy']:.2%} after the "
                f"soft fit to {history[-1]['accuracy']:.2%} after the last stage."
            )
            + "\n\n"
            "| | entropy per layer | accuracy |\n|---|---|---|\n" + _rows
        ),
    ])
    return


@app.cell
def _(class_names, draw_tree, entropy, history, mo, np, plt, test_labels):
    import base64 as _base64
    import io as _io
    import json as _json

    # The test samples the soft fit is least sure about, namely with the highest
    # entropy over the leaves, among those the squeezed tree classifies correctly
    _soft_leaves = history[0]["layers"][-1]
    _correct = history[-1]["predictions"] == test_labels
    _order = np.argsort([-entropy(_row[None]) for _row in _soft_leaves])
    _picked = [_i for _i in _order if _correct[_i]][:2]
    # Plus the least sure setosa, for comparison with an easy class
    _setosa = class_names.index("setosa")
    _picked.append(next(_i for _i in _order if _correct[_i] and test_labels[_i] == _setosa))

    def _render(h):
        """One stage of the schedule as a PNG data URL."""
        _fig, _axes = plt.subplots(1, len(_picked), figsize=(4.2 * len(_picked), 3.6))
        for _ax, _i in zip(_axes, _picked):
            _layers = [np.ones(1)] + [_p[_i] for _p in h["layers"]]
            draw_tree(
                _ax, _layers, leaf_labels=h["leaf_classes"],
                title=f"sample {_i} ({class_names[test_labels[_i]]}), "
                f"H = {entropy(h['layers'][-1][_i][None]):.2f}",
            )
        _fig.suptitle(h["label"], fontsize=12)
        _fig.tight_layout()
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=110)
        plt.close(_fig)
        return "data:image/png;base64," + _base64.b64encode(_buf.getvalue()).decode()

    # Every stage is drawn up front, so the slider needs no kernel and keeps
    # working in a static HTML export
    _frames = [_render(_h) for _h in history]
    _labels = [_h["label"] for _h in history]

    _slider = f"""
    <style>
      body {{ margin: 0; background: transparent; color: #e6edf3;
              font-family: system-ui, sans-serif; font-size: 14px; }}
      .row {{ display: flex; align-items: center; gap: 12px; margin: 4px 0 8px; }}
      input[type=range] {{ flex: 0 1 320px; accent-color: #4eb4da; }}
      #label {{ font-variant-numeric: tabular-nums; color: #4eb4da; font-weight: 600; }}
      img {{ width: 100%; height: auto; display: block; }}
    </style>
    <div class="row">
      <label for="stage">Stage</label>
      <input id="stage" type="range" min="0" max="{len(_frames) - 1}" value="0" step="1">
      <span id="label">{_labels[0]}</span>
    </div>
    <img id="frame" src="{_frames[0]}" alt="Tree at the current stage">
    <script>
      const frames = {_json.dumps(_frames)};
      const labels = {_json.dumps(_labels)};
      const slider = document.getElementById("stage");
      slider.addEventListener("input", () => {{
        document.getElementById("frame").src = frames[slider.value];
        document.getElementById("label").textContent = labels[slider.value];
      }});
    </script>
    """

    mo.vstack([
        mo.md(
            "Move the slider to watch the schedule squeeze the tree. Each tree "
            "follows one test sample that the squeezed tree gets right. The first "
            "two are the ones the soft fit is least sure about, namely those with "
            "the highest entropy $H$ over the leaves; the third is the setosa "
            "sample with the highest $H$. Node colour is the probability of "
            "reaching that node, and the text next to each leaf is the class its "
            "bigot favours."
        ),
        mo.iframe(_slider, height="400px"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inference as a proper tree

    Once the tree is squeezed, we can do inference by going through the
    nodes along a single path, following the more likely branch at each
    node. This makes inference $\mathcal{O}(D)$ instead of
    $\mathcal{O}(2^D)$. The two predictions agree as long as the squeezing
    worked.
    """)
    return


@app.cell
def _(CMAP, X_test, class_names, history, ink, mo, mobt, np, plt, test_labels):
    # history is listed so that this runs after the squeezing
    _final = history[-1]
    _soft = np.argmax(mobt.predict(X_test), axis=1)
    _hard = np.argmax(mobt.predict_tree(X_test), axis=1)

    _n = len(class_names)
    _confusion = np.zeros((_n, _n), dtype=int)
    for _t, _p in zip(test_labels, _hard):
        _confusion[_t, _p] += 1

    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.imshow(_confusion, cmap=CMAP)
    for (_i, _j), _v in np.ndenumerate(_confusion):
        _ax.text(_j, _i, str(_v), ha="center", va="center",
                 color=ink(_v / _confusion.max()))
    _ax.set_xticks(range(_n), class_names)
    _ax.set_yticks(range(_n), class_names)
    _ax.set_xlabel("single-path prediction")
    _ax.set_ylabel("true class")
    _fig.tight_layout()

    _summary = (
        f"After {_final['label']}, the two predictions agree on "
        f"**{np.mean(_soft == _hard):.2%}** of the test set.\n\n"
        "| | accuracy |\n"
        "|---|---|\n"
        f"| mixture of every leaf, $\\mathcal{{O}}(2^D)$ | {np.mean(_soft == test_labels):.2%} |\n"
        f"| single path, $\\mathcal{{O}}(D)$ | {np.mean(_hard == test_labels):.2%} |\n\n"
        "Confusion matrix of the single-path predictions:"
    )
    mo.vstack([mo.md(_summary), _ax])
    return


if __name__ == "__main__":
    app.run()
