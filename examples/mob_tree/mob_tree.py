import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    from miniml_examples.mob_tree import MoBBinaryTree
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    return MoBBinaryTree, datasets, jnp, mo, np, train_test_split


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
def _(datasets, jnp, np, train_test_split):
    iris = datasets.load_iris()

    X = jnp.array(iris.data)
    # One-hot encoding of the targets
    y = jnp.zeros((len(X), 3))
    y = y.at[np.arange(len(X)), iris.target].set(1)

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


if __name__ == "__main__":
    app.run()
