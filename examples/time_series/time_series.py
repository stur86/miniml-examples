import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from erlenmeyer import Species, Reaction, ReactionSystem, GillespieSimulator, sample_trajectory
    import matplotlib.pyplot as plt
    from miniml_examples.dataset import TimeSeriesDataset
    from scipy.stats import zscore

    return (
        GillespieSimulator,
        Reaction,
        ReactionSystem,
        Species,
        TimeSeriesDataset,
        mo,
        np,
        plt,
        sample_trajectory,
        zscore,
    )


@app.cell
def _(Reaction, ReactionSystem, Species):
    # Generate synthetic data: Lotka-Volterra w Gillespie
    # Species: Herbivore and Carnivore
    H, C, _W = Species('H'), Species('C'), Species('W')
    # Reaction
    lv = ReactionSystem([H, C, _W])
    lv.add_reaction(Reaction(H, 2 * H, 2.0))
    lv.add_reaction(Reaction(H + C, 2 * C, 0.01))
    lv.add_reaction(Reaction(C, _W, 1.0))
    return C, H, lv


@app.cell
def _(C, GillespieSimulator, H, lv, mo):
    gill_sim = GillespieSimulator(lv)
    start_pop = {H: 100, C: 50}
    t_end = 40.0
    dataset = {
        'gill_trajectories': [],
        'sampled_trajectories': []
    }

    _seed = 0
    gill_target_n = 100
    with mo.status.progress_bar(total=gill_target_n) as _bar:
        while len(dataset['gill_trajectories']) < gill_target_n:
            _traj = gill_sim.run(start_pop, seed=_seed, t_end=t_end, max_steps=100000)
            if _traj.times[-1] >= t_end:
                # Successful
                dataset['gill_trajectories'].append(_traj)
                _bar.update()
            _seed += 1
    return dataset, gill_target_n, t_end


@app.cell
def _(dataset, plt):
    _f, _a = plt.subplots()

    _sample_traj = dataset['gill_trajectories'][20]
    _a.plot(_sample_traj.times, _sample_traj.values[:,:2])
    _a.legend(_sample_traj.species)
    return


@app.cell
def _(dataset, gill_target_n, mo, np, sample_trajectory, t_end):
    # Now the sampling
    samples_per_traj = 10
    sample_t = np.linspace(0, t_end, 100)
    dataset["sampled_trajectories"] = []
    with mo.status.progress_bar(total=gill_target_n*samples_per_traj) as _bar:
        for _traj in dataset["gill_trajectories"]:
            for _seed in range(samples_per_traj):
                sampled_pops = sample_trajectory(_traj, volume_fraction=0.01, with_replacement=False, rng=_seed, volume=20, times=sample_t)
                dataset["sampled_trajectories"].append(sampled_pops)
                _bar.update()
    dataset["sampled_trajectories"] = np.array(dataset["sampled_trajectories"])
    return (sample_t,)


@app.cell
def _(dataset, np, plt, sample_t):
    _f, _a = plt.subplots()

    _mean = np.mean(dataset["sampled_trajectories"][:,:,:2], axis=0)
    _std = np.std(dataset["sampled_trajectories"][:,:,:2], axis=0)
    _a.plot(sample_t, _mean)
    _a.fill_between(sample_t, (_mean-_std)[:,0], (_mean+_std)[:,0], alpha=0.2)
    _a.fill_between(sample_t, (_mean-_std)[:,1], (_mean+_std)[:,1], alpha=0.2)
    return


@app.cell
def _(TimeSeriesDataset, dataset):
    tsdset = TimeSeriesDataset(4, 1, dataset["sampled_trajectories"][:,:,:2], target_feature=1)
    return (tsdset,)


@app.cell
def _(dataset, np, plt, sample_t, tsdset, zscore):
    _X = zscore(tsdset.x)
    _X = np.concat([_X, np.ones((len(_X), 1))], axis=1)
    _y = tsdset.y

    linreg_c = np.linalg.solve(_X.T@_X+0.1*np.eye(_X.shape[1]), _X.T@_y)

    _f, _a = plt.subplots()

    _test_idx = 50
    _ttest = dataset["sampled_trajectories"][_test_idx,:,:2]
    _a.plot(sample_t, _ttest)
    _a.plot(sample_t[tsdset._idim:], _X[(100-tsdset._idim)*_test_idx:(100-tsdset._idim)*(_test_idx+1)]@linreg_c)
    return (linreg_c,)


@app.cell
def _(linreg_c):
    linreg_c
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
