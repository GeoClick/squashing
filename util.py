import numpy as np
import matplotlib.pyplot as plt


def plot_gp(X, m, C, training_points=None, fig=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    if fig is None:
        fig = plt.figure()

    X = np.squeeze(X)
    m = np.squeeze(m)

    fig.fill_between(X,
                     m - 1.96 * np.sqrt(np.diag(C)),  # 95% confidence interval of a Gaussian is
                     m + 1.96 * np.sqrt(np.diag(C)),  # .. contained within mean +/- 1.96*s.d.
                     alpha=0.5)
    # Plot GP mean and initial training points
    fig.plot(X, m, "-", label="Mean GP fit")
    fig.legend()

    fig.set_xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        fig.plot(X_, Y_, "kx", mew=2)
        fig.legend(labels=["GP fit", "sample points"])
