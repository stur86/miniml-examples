import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from miniml_examples.dataset import sklearn_classification_data
    from miniml_examples.mob_tree import MoBBinaryTree, sigmoid_entropy_schedule
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    return (
        MoBBinaryTree,
        datasets,
        mo,
        np,
        sigmoid_entropy_schedule,
        sklearn_classification_data,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Mixture of Bigots tree

        A soft binary decision tree, fitted on the Iris dataset. Each internal
        node splits the samples with a logistic gate, and each leaf holds one
        fixed class distribution (a "bigot"). The prediction is the mixture of
        the leaf distributions, weighted by the probability of reaching each
        leaf.
        """
    )
    return


@app.cell
def _(datasets, sklearn_classification_data, train_test_split):
    X, y = sklearn_classification_data(datasets.load_iris())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 80% train, 20% test
        shuffle=True,  # Shuffles/scrambles the dataset (True by default)
        random_state=42,  # Seed for reproducible randomness
        stratify=y,  # Preserves class balance across splits
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(MoBBinaryTree, X_train, y_train):
    mobt = MoBBinaryTree(n_in=4, n_out=3, depth=3)
    mobt.randomize(5)

    print(f"Fit, success: {mobt.fit(X_train, y_train).success}")
    # Class logits of each of the eight leaves
    print(mobt.get_params()["_leaf_weights.v"])
    return (mobt,)


@app.cell
def _(X_test, mobt, np, y_test):
    print(
        f"Accuracy: {np.mean(np.argmax(mobt.predict(X_test), axis=1) == np.argmax(y_test, axis=1)):.2%}"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Squeezing the tree

        The splits above are soft: a sample reaches several leaves at once. The
        entropy activity loss penalizes that. Its weights, one per layer, can be
        changed between fits, so the tree can be hardened one stage at a time.

        The schedule below is a logistic front that walks down the tree: it
        squeezes the layer below the root first, then the next one, and leaves
        the layers below the front free to stay soft in the meantime.
        """
    )
    return


@app.cell
def _(X_test, X_train, mobt, np, sigmoid_entropy_schedule, y_test, y_train):
    def _layer_entropies() -> np.ndarray:
        # Mean entropy of the distribution of a test sample over each layer
        _h = []
        for _layer in range(1, mobt.depth + 1):
            _p = np.array(mobt.layer_probs(X_test, _layer))
            _h.append(-np.sum(_p * np.log(np.clip(_p, 1e-12, None)), axis=1).mean())
        return np.array(_h)

    def _report(stage: str) -> None:
        _acc = np.mean(
            np.argmax(mobt.predict(X_test), axis=1) == np.argmax(y_test, axis=1)
        )
        print(
            f"{stage:>10}: entropy per layer {np.round(_layer_entropies(), 4)}"
            f", accuracy {_acc:.2%}"
        )

    _report("soft")

    # One fit per stage, each starting from the fit of the stage before
    _schedule = sigmoid_entropy_schedule(mobt.depth, scale=8.0)
    for _stage, _weights in enumerate(_schedule):
        mobt.set_entropy_weights(_weights)
        mobt.fit(X_train, y_train)
        _report(f"stage {_stage}")
    return


if __name__ == "__main__":
    app.run()
