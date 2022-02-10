import copy
from dataclasses import dataclass

import numpy as np
from numpy.linalg import slogdet
from scipy.special import loggamma, multigammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class niw_sufficient_statistics:
    """NIW sufficient statistics
    Args:
        X (np.ndarray): array of data, dimensions [N, D]
    """

    N: int
    D: int
    sum: np.ndarray
    S: np.float32

    def __init__(self, X: np.ndarray) -> None:
        """Initialize and calculate sufficient statistics for NIW.
        Args:
            X (np.ndarray): array of data, dimensions [N, D]
        """

        self.N, self.D = X.shape
        self.sum = np.array(X.sum(0))

        X = np.array(X, dtype=np.float64)
        S = X.T @ X
        self.S = 0.5 * (S + S.T)


@dataclass
class niw_hyperparams:
    """Class to hold the parameters for the normal-inverse-Wishart distribution"""

    k: np.float32
    v: np.float32
    mu: np.ndarray
    psi: np.ndarray

    def update_posterior(self, suff_stat: niw_sufficient_statistics):
        """updated the NIW parameters posterior parameters
        Args:
            suff_stat (niw_sufficient_statistics):
        """
        # updated params (equations 2.43, 2.44)
        if suff_stat.N == 0:
            return self
        k = self.k + suff_stat.N
        v = self.v + suff_stat.N
        mu = (self.mu * self.k + suff_stat.sum) / k

        psi = (
            self.v * self.psi
            + self.k * np.outer(self.mu, self.mu)
            - k * np.outer(mu, mu)
            + suff_stat.S
        ) / v

        self.k = k
        self.v = v
        self.mu = mu
        self.psi = (psi + psi.T) / 2


def log_marginal_likelihood(
    niw_prior: niw_hyperparams,
    niw_posterior: niw_hyperparams,
    suff_stats: niw_sufficient_statistics,
) -> float:
    """Calculates the marginal log likelihood.
    Args:
        niw_prior (niw_hyperparams): the NIW global prior
        niw_posterior (niw_hyperparams): the updated corresponding posterior
        suff_stats (niw_sufficient_statistics): the corresponing suff. stats.
    Returns:
        float: marginal log likelihood
    """

    D = suff_stats.D
    logpi = np.log(np.pi)

    ll = (
        -suff_stats.N * D * 0.5 * logpi
        + multigammaln(niw_posterior.v / 2, D)
        - multigammaln(niw_prior.v / 2, D)
        + (niw_prior.v / 2) * (D * np.log(niw_prior.v) + slogdet(niw_prior.psi)[1])
        - (niw_posterior.v / 2)
        * (D * np.log(niw_posterior.v) + slogdet(niw_posterior.psi)[1])
        + (D / 2) * (np.log(niw_prior.k / niw_posterior.k))
    )

    return ll


def log_hasting_ratio(
    X: np.ndarray, Z: np.ndarray, niw_prior: niw_hyperparams, alpha: float
) -> float:
    """Calculates the log Hasting Ratio
    Args:
        X (np.ndarray): data array, shaped: [N, D]
        Z (np.ndarray): binary {0,1} split assignment array, shaped: [N]
        niw_prior (niw_hyperparams): the global NIW prior
        alpha (float): Dirichlet alpha paramater
    Returns:
        float: log Hasting Ratio
    """

    niw_post = copy.copy(niw_prior)
    niw_post_r = copy.copy(niw_prior)
    niw_post_l = copy.copy(niw_prior)

    # calculate sufficient statistics:
    suff_stats_cp = niw_sufficient_statistics(X)
    suff_stats_cpr = niw_sufficient_statistics(X[Z == 0])
    suff_stats_cpl = niw_sufficient_statistics(X[Z == 1])

    if suff_stats_cpr.N == 0 or suff_stats_cpl.N == 0:
        return -1000

    # update niw posteriors params:
    niw_post.update_posterior(suff_stats_cp)
    niw_post_r.update_posterior(suff_stats_cpr)
    niw_post_l.update_posterior(suff_stats_cpl)

    # calculate marginal log likelihood:
    log_likelihood = log_marginal_likelihood(niw_prior, niw_post, suff_stats_cp)
    log_likelihood_r = log_marginal_likelihood(niw_prior, niw_post_r, suff_stats_cpr)
    log_likelihood_l = log_marginal_likelihood(niw_prior, niw_post_l, suff_stats_cpl)

    log_HR = (
        np.log(alpha)
        + loggamma(suff_stats_cpr.N)
        + log_likelihood_r
        + loggamma(suff_stats_cpl.N)
        + log_likelihood_l
        - loggamma(suff_stats_cp.N)
        - log_likelihood
    )

    return log_HR


if __name__ == "__main__":
    D = 4
    N = 1000
    X = np.random.normal(size=(N, D))

    Z = np.array(np.random.rand(N) > 0.5, dtype=np.int8)
    # Z = np.ones(shape=N)

    suff_stat = niw_sufficient_statistics(X[Z == 0])
    print("Suff stats:")
    print(suff_stat)

    k = 0.1
    v = 10.0
    mu = np.random.normal(size=(1, D))
    psi = np.random.normal(size=(D, D))
    psi = 0.5 * (psi + psi.T)

    niw_params = niw_hyperparams(k, v, mu, psi)
    print("NIW params:")
    print(niw_params)

    niw_params_post = copy.copy(niw_params)

    niw_params_post.update_posterior(suff_stat)
    print("Posterior update:")
    print(niw_params_post)

    print("log_marginal_likelihood calculation:")
    ll = log_marginal_likelihood(niw_params, niw_params_post, suff_stat)
    print(ll)

    lhr = log_hasting_ratio(X, Z, niw_params, 10)
    print(f"Log Hasting Ratio calc: {lhr}")
    print(f"Hasting Ratio calc: {np.exp(lhr)}")


def calc_hastings_ratio_gt(X, Y, niw_prior, alpha):

    bs = len(X)

    hastings_mat = np.zeros(bs)

    for i in range(bs):
        # pos = Y[i] == 0
        # if 1 < pos.sum() < N - 1:
        x = X[i]
        y = Y[i]
        H = log_hasting_ratio(x, y, niw_prior, alpha)
        hastings_mat[i] = H

    return hastings_mat


def calc_hastings_ratio_kmeans(X, niw_prior, alpha):
    bs = len(X)

    hast_mat = np.zeros(bs)

    for i in range(bs):
        x = X[i]
        y_kmeans = KMeans(n_clusters=2).fit_predict(x)
        H = log_hasting_ratio(x, y_kmeans, niw_prior, alpha)
        hast_mat[i] = H

    return hast_mat


def calc_hastings_ratio_random(X, niw_prior, alpha):
    bs, N, D = X.shape
    hast_mat = np.zeros(bs)

    for i in range(bs):
        x = X[i]
        y_rand = np.random.choice(2, size=N)
        H = log_hasting_ratio(x, y_rand, niw_prior, alpha)
        hast_mat[i] = H

    return hast_mat


def calc_hastings_ratio_ratio(HR_pred, HR_gt) -> float:
    """calculates the mean Hastings Ratio between 2 sets
    Args:
        HR_pred ([type]): [description]
        HR_gt ([type]): [description]
    Returns:
        [type]: [description]
    """

    assert HR_gt.shape == HR_pred.shape

    N = HR_gt.shape[0]
    hast_mat = np.zeros(N)

    for i in range(N):
        # hast_mat[i] = np.exp(HR_pred[i] - HR_gt[i])
        hast_mat[i] = HR_pred[i] / HR_gt[i]

    ratio_mean = float(hast_mat.mean())
    return ratio_mean


def eignesplit(X: np.ndarray) -> np.ndarray:
    # X_ = StandardScaler().fit_transform(X)
    X_1d = PCA(n_components=1).fit_transform(X)

    p_10 = np.percentile(X_1d, 10)
    p_90 = np.percentile(X_1d, 90)

    init_centers = np.array([p_10, p_90]).reshape(-1, 1)

    Z = KMeans(n_clusters=2, init=init_centers, n_init=1).fit_predict(X_1d)

    return Z


def calc_hastings_ratio_eigensplit(X, niw_prior, alpha):
    bs, N, D = X.shape
    hast_mat = np.zeros(bs)

    for i in range(bs):
        x = X[i]
        y_pred = eignesplit(x)
        H = log_hasting_ratio(x, y_pred, niw_prior, alpha)
        hast_mat[i] = H

    return hast_mat