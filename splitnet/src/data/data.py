import numpy as np
import seaborn
import matplotlib.pyplot as plt

from numpy.random import dirichlet
from scipy.stats import invwishart, multivariate_normal

from src.utils.metrics import log_hasting_ratio, niw_hyperparams
from progressbar import ProgressBar


def sample_niw(psi: np.ndarray, nu: int, mu_0: np.ndarray, kappa: float):
    """
    samples from Normal Inverse Wishart distrbution with params:
    - psi: inverse scale matrix, must be a SPD, shape: [D,D]
    - mu_0: location prior, shape: [1, D]
    - nu: degrees of freedom, must be nu > D -1
    - kappa: kappabda,

    returns mu, Sigma
    """
    # sample covariance matrix:
    D = psi.shape[0]
    assert nu > D - 1

    # sample covariance matrix:
    cov = invwishart(df=int(nu), scale=psi).rvs()

    # sample mu:
    cov_ = (cov / kappa) + np.eye(D) * 1e-8
    mu = multivariate_normal(mean=mu_0, cov=cov_).rvs()
    return mu, cov


def sample_iw(psi: np.ndarray, nu: int):
    """
    samples from Inverse Wishart distrbution with params:
    - psi: inverse scale matrix, must be a SPD, shape: [D,D]
    - mu_0: location prior, shape: [1, D]
    - nu: degrees of freedom, must be nu > D -1
    - kappa: kappabda,

    returns mu, Sigma
    """
    # sample covariance matrix:
    D = psi.shape[0]
    assert nu > D - 1

    # sample covariance matrix:
    cov = invwishart(df=nu, scale=psi).rvs()

    # set mu to zero:
    mu = np.zeros(D)
    # mu = multivariate_normal(mean=np.zeros(D), cov=cov / kappa).rvs()
    return mu, cov


def generate_sample_pair(
    alpha,
    psi,
    nu,
    mu_0,
    kappa,
    N=1000,
):

    n1, n2 = np.round(dirichlet(alpha=alpha, size=1) * N).flatten()
    n1, n2 = int(n1), int(n2)

    mu_1, cov_1 = sample_niw(psi=psi, nu=nu, mu_0=mu_0, kappa=kappa)
    mu_2, cov_2 = sample_niw(psi=psi, nu=nu, mu_0=mu_0, kappa=kappa)
    # mu_1, cov_1 = sample_iw(psi=psi, nu=nu)
    # mu_2, cov_2 = sample_iw(psi=psi, nu=nu)

    samples_1 = multivariate_normal(mean=mu_1, cov=cov_1).rvs(n1)
    samples_2 = multivariate_normal(mean=mu_2, cov=cov_2).rvs(n2)

    X = np.vstack([samples_1, samples_2])
    y = np.hstack([np.zeros(n1), np.ones(n2)])

    return X, y


def genereate_splitnet_dataset(
    niw_params: niw_hyperparams,
    alpha: int,
    dp_alpha: int,
    num_points: int,
    dataset_size: int,
    hr_threshold: float = 0.01,
    use_hr_threshold: bool = True,
    multi_k: bool = False,
    K_max: int = 10,
):
    """create a synthetic dataset for splitnet training and evaluation
    shape: [dataset_size, D, N]

    Args:
        D (int): [description]
        dataset_size (int): [description]

    Returns:
        [type]: [description]
    """
    # dirichlet params:
    alpha = np.ones(2) * alpha

    # NIW params:
    k = niw_params.k
    v = niw_params.v
    mu_0 = niw_params.mu
    psi = niw_params.psi

    X = []
    Y = []

    i = 0

    pbar = ProgressBar(maxval=dataset_size).start()
    while i < dataset_size:
        if multi_k:
            x, y = generate_sample_multi_pair(
                alpha=alpha,
                psi=psi,
                nu=v,
                mu_0=mu_0,
                kappa=k,
                N=num_points,
                K_max=K_max,
            )
        else:
            x, y = generate_sample_pair(
                alpha=alpha,
                psi=psi,
                nu=v,
                mu_0=mu_0,
                kappa=k,
                N=num_points,
            )
        if use_hr_threshold:
            if log_hasting_ratio(x, y, niw_params, dp_alpha) > np.log(hr_threshold):
                X.append(x)
                Y.append(y)
                i = i + 1
                pbar.update(i)
        else:
            X.append(x)
            Y.append(y)
            i = i + 1

    pbar.finish()
    X = np.stack(X, axis=0)
    Y = np.vstack(Y)

    print(f"Generated data, shape: {X.shape}")
    print(f"Generated labels, shape: {Y.shape}")

    return X, Y


def generate_sample_multi_pair(
    alpha,
    psi,
    nu,
    mu_0,
    kappa,
    K_max,
    N=1000,
):

    n1, n2 = np.round(dirichlet(alpha=alpha, size=1) * N).flatten()
    n1, n2 = int(n1), int(n2)

    # sample initial locations
    mu_1, cov_1 = sample_niw(psi=psi * 10, nu=nu, mu_0=mu_0, kappa=kappa)
    mu_2, cov_2 = sample_niw(psi=psi * 10, nu=nu, mu_0=mu_0, kappa=kappa)

    # sample how many sub-clusters in each
    k1 = int(np.random.choice(np.arange(1, K_max), 1))
    k2 = int(np.random.choice(np.arange(1, K_max), 1))

    # distribute 10% between all subcluster to ensure minimal # of points:
    pct = 0.10
    k1_sub_ns = np.ones(shape=k1) * int(n1 * pct / k1)
    k2_sub_ns = np.ones(shape=k2) * int(n2 * pct / k2)

    # sample how many points in each sub-cluster
    k1_sub_ns += np.floor(
        dirichlet(alpha=[1] * k1, size=1) * (n1 - k1_sub_ns.sum())
    ).flatten()
    k2_sub_ns += np.floor(
        dirichlet(alpha=[1] * k2, size=1) * (n2 - k2_sub_ns.sum())
    ).flatten()

    if k1_sub_ns.sum() != n1 or k2_sub_ns.sum() != n2:
        k1_sub_ns[k1_sub_ns.argmin()] += np.abs(k1_sub_ns.sum() - n1)
        k2_sub_ns[k2_sub_ns.argmin()] += np.abs(k2_sub_ns.sum() - n2)

    samples_1 = []
    samples_2 = []

    y = []
    i_ = 0

    for i in range(k1):
        m1, c1 = sample_niw(psi=psi, nu=nu, mu_0=mu_1, kappa=kappa / 10)
        points = multivariate_normal(mean=m1, cov=c1).rvs(int(k1_sub_ns[i]))
        samples_1.append(points)
        y.append(np.ones(int(k1_sub_ns[i]) * i))
        i_ = i

    for j in range(k2):
        m2, c2 = sample_niw(psi=psi, nu=nu, mu_0=mu_2, kappa=kappa / 10)
        points = multivariate_normal(mean=m2, cov=c2).rvs(int(k2_sub_ns[j]))
        samples_2.append(points)
        y.append(np.ones(int(k2_sub_ns[j]) * (j + i_)))

    X = np.vstack([np.concatenate(samples_1), np.concatenate(samples_2)])
    Y_k = np.hstack(y)
    Y = np.hstack([np.zeros(n1), np.ones(n2)])

    return X, Y, Y_k


def genereate_splitnet_dataset_multi(
    niw_params: niw_hyperparams,
    alpha: int,
    dp_alpha: int,
    num_points: int,
    dataset_size: int,
    hr_threshold: float = 0.01,
    K_max: int = 10,
):
    """create a synthetic dataset for splitnet training and evaluation
    shape: [dataset_size, D, N]

    Args:
        D (int): [description]
        dataset_size (int): [description]

    Returns:
        [type]: [description]
    """
    # dirichlet params:
    alpha = np.ones(2) * alpha

    # NIW params:
    k = niw_params.k
    v = niw_params.v
    mu_0 = niw_params.mu
    psi = niw_params.psi

    X = []
    Y = []

    i = 0
    while i < dataset_size:
        x, y, y_k = generate_sample_multi_pair(
            alpha=alpha,
            psi=psi,
            nu=v,
            mu_0=mu_0,
            kappa=k,
            K_max=K_max,
            N=num_points,
        )
        if log_hasting_ratio(x, y, niw_params, dp_alpha) > np.log(hr_threshold):
            X.append(x)
            Y.append(y_k)
            Y.append(y)
            i = i + 1

    X = np.stack(X, axis=0)
    Y = np.vstack(Y)

    print(f"Generated data, shape: {X.shape}")
    print(f"Generated labels, shape: {Y.shape}")

    return X, Y