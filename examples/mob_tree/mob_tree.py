import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import marimo as mo
    import numpy as np
    from miniml import MiniMLModel, MiniMLParam, PredictMode
    from jax import Array as JxArray
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.debug as jdebug
    from miniml.loss import CrossEntropyLogLoss, LNormRegularization
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    return (
        CrossEntropyLogLoss,
        LNormRegularization,
        MiniMLModel,
        MiniMLParam,
        PredictMode,
        datasets,
        jax,
        jnp,
        np,
        train_test_split,
    )


@app.cell
def _(np):
    def make_next_step_path_matrix(m: np.ndarray):
        n_leaves = m.shape[0]
        n_leaves_new = 2*n_leaves
        n_nodes_new = n_leaves
        new_branches = np.zeros((n_leaves_new, n_nodes_new))
        I = np.eye(n_leaves)
        new_branches[1::2,:] = I
        return np.concatenate([np.repeat(m, 2, axis=0), new_branches], axis=1)

    def make_next_step_norm_matrix(m: np.ndarray):
        n_paths = m.shape[0]
        return np.concatenate([np.repeat(m, 2, axis=0), np.eye(2*n_paths)], axis=1)    

    def make_all_path_matrices(n: int):
        a0 = np.array([0, 1])[:,None]
        all = [a0]
        for i in range(1, n):
            all.append(make_next_step_path_matrix(all[-1]))
        return all

    def make_all_norm_matrices(n: int):
        a0 = np.array([1])[:,None]
        all = [a0]
        for i in range(1, n):
            all.append(make_next_step_norm_matrix(all[-1]))
        return all

    return make_all_norm_matrices, make_all_path_matrices


@app.cell
def _(make_all_path_matrices, np):
    make_all_path_matrices(2)[1]@np.random.random((3,2))
    return


@app.cell
def _(make_all_norm_matrices):
    make_all_norm_matrices(2)[1]
    return


@app.cell
def _(
    CrossEntropyLogLoss,
    LNormRegularization,
    MiniMLModel,
    MiniMLParam,
    PredictMode,
    jax,
    jnp,
    make_all_norm_matrices,
    make_all_path_matrices,
):
    class MoBBinaryTree(MiniMLModel):

        def __init__(self, n_in: int, n_out: int, depth: int):
            # Tree weights
            self._num_leaves = 2**depth
            self._tree_weights = MiniMLParam(shape=(self._num_leaves-1, n_in), reg_loss=LNormRegularization(1))
            self._mob_weights = MiniMLParam(shape=(self._num_leaves, n_out),
    reg_loss=LNormRegularization(2), reg_scale=1e-2)

            self._path_matrices = [jnp.array(a) for a in make_all_path_matrices(depth)]
            self._norm_matrices = [jnp.array(a) for a in make_all_norm_matrices(depth)]
        
            super().__init__(loss=CrossEntropyLogLoss())

        def _predict_kernel(self, X, buffer, rng_key=None, mode=PredictMode.INFERENCE):
            node_pass_logits = self._tree_weights(buffer)@X.T

            # tree_logits = self._path_matrices[-1]@node_pass_logits
            # # Normalization is based on each individual pass factor
            # node_norms = 1.0+jnp.exp(node_pass_logits)
            # tree_norms = jnp.repeat(jnp.exp(self._norm_matrices[-1]@jnp.log(node_norms)), 2, axis=0)
            # tree_p = jnp.exp(tree_logits)
            # tree_p /= tree_norms
            # mob_logits = self._mob_weights(buffer)
            # mob_p = jnp.exp(mob_logits-np.mean(mob_logits, axis=1, keepdims=True))
            # mob_p /= jnp.sum(mob_p, axis=1, keepdims=True)
            # return jnp.log(jnp.sum((tree_p.T)[:,:,None]*mob_p[None,:,:], axis=1))

            node_logits = self._tree_weights(buffer) @ X.T
    
            # log(tree probability)
            tree_logits = self._path_matrices[-1] @ node_logits
            tree_log_norms = self._norm_matrices[-1] @ jax.nn.softplus(node_logits)
            tree_log_norms = jnp.repeat(tree_log_norms, 2, axis=0)
    
            tree_log_p = tree_logits - tree_log_norms
    
            # log(MoB probabilities)
            mob_logits = self._mob_weights(buffer)
            mob_log_p = mob_logits - jax.scipy.special.logsumexp(
                mob_logits, axis=1, keepdims=True
            )
    
            # log(sum(tree_p * mob_p))
            return jax.scipy.special.logsumexp(
                tree_log_p.T[:, :, None] + mob_log_p[None, :, :],
                axis=1,
            )

    return (MoBBinaryTree,)


@app.cell
def _(MoBBinaryTree):
    mobt = MoBBinaryTree(4, 3, 3)
    return (mobt,)


@app.cell
def _(datasets):
    iris = datasets.load_iris()
    return (iris,)


@app.cell
def _(iris, jnp, mobt, np, train_test_split):
    X = jnp.array(iris.data)
    y = jnp.zeros((len(X), 3))
    y = y.at[np.arange(len(X)), iris.target].set(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,       # 80% train, 20% test
        shuffle=True,        # Shuffles/scrambles the dataset (True by default)
        random_state=42,     # Seed for reproducible randomness
        stratify=y           # Preserves class balance across splits
    )


    mobt.randomize(5)

    # _mob_init = np.zeros((mobt._num_leaves, 3))
    # for _i in range(3):
    #     _mob_init[_i::3,_i] = 1.0
    # mobt.set_params({
    #     "_mob_weights.v": jnp.array(_mob_init)
    # })
    print(f"Fit, success: {mobt.fit(X_train, y_train).success}")
    print(mobt.get_params()["_mob_weights.v"])
    return X_test, y_test


@app.cell
def _(X_test, mobt, np, y_test):
    print(f"Accuracy: {np.mean(np.argmax(mobt.predict(X_test), axis=1) == np.argmax(y_test, axis=1)):.2%}")
    return


if __name__ == "__main__":
    app.run()
