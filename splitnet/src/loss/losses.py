import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import lgamma, mvlgamma
from torch.linalg import slogdet
from dataclasses import dataclass


@dataclass
class batch_niw_sufficient_stats:
    """NIW sufficient statistics

    Args:
        X (torch.Tensor): array of data, dimensions [bs, N, D]
    """

    bs: int
    D: int
    N: torch.Tensor
    sum: torch.Tensor
    S: torch.Tensor

    def __init__(self, X: torch.Tensor, Z: torch.Tensor = None) -> None:
        """Initialize and calculate sufficient statistics for NIW.

        Args:
            X (torch.Tensor): array of data, dimensions [bs, N, D]
            X (torch.Tensor): array of subscluster prob, dimensions [bs, N, 1]
        """
        if hasattr(X, "device"):
            self.device = X.device
        else:
            self.device = "cpu"

        if Z is None:
            self.bs = X.shape[0]
            self.D = X.shape[-1]
            self.N = (X != 0).sum(1)[:, 0]

            self.sum = X.sum(1)
            self.S = torch.bmm(X.transpose(2, 1), X)
        else:
            self.bs = X.shape[0]
            self.D = X.shape[-1]
            self.N = Z.sum(1)[:, 0]

            self.sum = torch.multiply(X, Z).sum(1)
            self.S = torch.bmm(torch.multiply(X, Z).transpose(2, 1), X)


@dataclass
class niw_hyperparams:
    """Class to hold the parameters for the Normal-Inverse-Wishart distribution"""

    def __init__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        mu: torch.Tensor,
        psi: torch.Tensor,
    ):

        if not isinstance(k, torch.Tensor):
            self.k = torch.tensor(k)  # , requires_grad=True)
            self.v = torch.tensor(v)  # , requires_grad=True)
            self.mu = torch.tensor(mu)  # , requires_grad=True)
            self.psi = torch.tensor(psi)  # , requires_grad=True)
        else:
            self.k = k.clone()
            self.v = v.clone()
            self.mu = mu.clone()
            self.psi = psi.clone()

    def update_posterior(self, suff_stat: batch_niw_sufficient_stats):
        """updated the NIW parameters posterior parameters

        Args:
            suff_stat (niw_sufficient_statistics):
        """
        # move all tensors to device
        device = suff_stat.device
        self.k = self.k.to(device)
        self.v = self.v.to(device)
        self.mu = self.mu.to(device)
        self.psi = self.psi.to(device)

        # TODO: DO WE NEED THIS?
        # if suff_stat.N == 0:
        # return self

        k = self.k + suff_stat.N
        v = self.v + suff_stat.N
        mu = torch.div((self.mu * self.k + suff_stat.sum), k.unsqueeze(1))

        psi = (
            self.v * self.psi.repeat(suff_stat.bs, 1, 1)
            + self.k * torch.matmul(self.mu.T, self.mu).repeat(suff_stat.bs, 1, 1)
            - k.unsqueeze(1).unsqueeze(1)
            * torch.bmm(mu.unsqueeze(-1), mu.unsqueeze(-1).transpose(2, 1))
            + suff_stat.S
        ) / v.unsqueeze(1).unsqueeze(1)

        self.k = k
        self.v = v
        self.mu = mu
        self.psi = psi


def compute_data_covs_soft_assignment(logits, codes, K, mus, prior_name="NIW"):
    # compute the data covs in soft assignment
    if prior_name == "NIW":
        covs = []
        n_k = logits.sum(axis=0)
        for k in range(K):
            if len(logits) == 0 or len(codes) == 0:
                # happens when finding subcovs of empty clusters
                cov_k = torch.eye(mus.shape[1]) * 0.0001
            else:
                cov_k = torch.matmul(
                    (logits[:, k] * (codes - mus[k].repeat(len(codes), 1)).T),
                    (codes - mus[k].repeat(len(codes), 1)),
                )
                cov_k = cov_k / n_k[k]
            covs.append(cov_k)
        if torch.isnan(torch.stack(covs)).any():
            print("stop for debugging")
        return torch.stack(covs)
    elif prior_name == "NIG":
        return compute_data_sigma_sq_soft_assignment(
            logits=logits, codes=codes, K=K, mus=mus
        )


def batch_log_marginal_likelihood(
    niw_prior: niw_hyperparams,
    niw_posterior: niw_hyperparams,
    suff_stats: batch_niw_sufficient_stats,
) -> float:
    """Calculates the marginal log likelihood.

    Args:
        niw_prior (niw_hyperparams): the NIW global prior
        niw_posterior (niw_hyperparams): the updated corresponding posterior
        suff_stats (niw_sufficient_statistics): the corresponing suff. stats.

    Returns:
        float: marginal log likelihood per dataset, shape: [bs,]
    """

    D = suff_stats.D

    logpi = torch.log(torch.tensor(np.pi))

    ll = (
        -suff_stats.N * D * 0.5 * logpi
        + mvlgamma(niw_posterior.v / 2, D)
        - mvlgamma(niw_prior.v / 2, D)
        + (niw_prior.v / 2) * (D * torch.log(niw_prior.v) + slogdet(niw_prior.psi)[1])
        - (niw_posterior.v / 2)
        * (D * torch.log(niw_posterior.v) + slogdet(niw_posterior.psi)[1])
        + (D / 2) * (torch.log(niw_prior.k / niw_posterior.k))
    )

    return ll


def batch_log_hasting_ratio(
    X: torch.Tensor, Z: torch.Tensor, niw_prior: niw_hyperparams, alpha: float
) -> torch.Tensor:
    """Calculates the log Hasting Ratio

    Args:
        X (torch.Tensor): data array, shaped: [bs, N, D]
        Z (torch.Tensor): binary {0,1} split assignment array, shaped: [bs, N]
        niw_prior (niw_hyperparams): the global NIW prior
        alpha (float): Dirichlet alpha paramater

    Returns:
        float: log Hasting Ratio
    """

    niw_post = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)
    niw_post_r = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)
    niw_post_l = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)

    # calculate sufficient statistics:
    suff_stats_cp = batch_niw_sufficient_stats(X)
    # suff_stats_cpr = batch_niw_sufficient_stats(torch.mul(X, Z.unsqueeze(-1)))
    # suff_stats_cpl = batch_niw_sufficient_stats(torch.mul(X, (1 - Z).unsqueeze(-1)))
    suff_stats_cpr = batch_niw_sufficient_stats(X, Z.unsqueeze(-1))
    suff_stats_cpl = batch_niw_sufficient_stats(X, (1 - Z).unsqueeze(-1))

    # TEMPORARY FIX TO AVOID ZERO's:
    # suff_stats_cpr.N[suff_stats_cpr.N == 0] = +1
    # suff_stats_cpl.N[suff_stats_cpr.N == 0] = -1

    # suff_stats_cpl.N[suff_stats_cpl.N == 0] = +1
    # suff_stats_cpr.N[suff_stats_cpl.N == 0] = -1

    # update niw posteriors params:
    niw_post.update_posterior(suff_stats_cp)
    niw_post_r.update_posterior(suff_stats_cpr)
    niw_post_l.update_posterior(suff_stats_cpl)

    # calculate marginal log likelihood:
    log_likelihood_r = batch_log_marginal_likelihood(
        niw_prior, niw_post_r, suff_stats_cpr
    )
    log_likelihood_l = batch_log_marginal_likelihood(
        niw_prior, niw_post_l, suff_stats_cpl
    )

    log_HR = (
        torch.log(torch.tensor(alpha))
        + lgamma(suff_stats_cpr.N)
        + log_likelihood_r
        + lgamma(suff_stats_cpl.N)
        + log_likelihood_l
    )

    return log_HR.requires_grad_(True)
    # return log_HR.clone().requires_grad_(True)
    # return log_HR.clone().detach().requires_grad_(True)
    # return log_HR


class HastingsRatioLoss(nn.Module):
    def __init__(self, niw_prior: niw_hyperparams, alpha: int):
        super().__init__()
        self.niw_prior = niw_hyperparams(
            niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi
        )
        self.alpha = alpha

    def forward(self, x, pred_labels):
        """[summary]

        Args:
            x ([type]): [description]
            pred_labels ([type]): prediction probabilities per subcluster

        Returns:
            [type]: [description]
        """
        # X = x.clone().detach()
        log_hr = batch_log_hasting_ratio(x, pred_labels, self.niw_prior, self.alpha)
        loss = -log_hr.mean()
        return loss


class SetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """[summary]

        Args:
            preds ([type]): label prediction, shape: [B, N]
            labels ([type]): labels, shape: [B, N]
        """
        K = 2
        bs, N = preds.shape[:-1]

        labels_oh = F.one_hot(labels.long(), num_classes=K)
        assert (bs, N, K) == labels_oh.shape

        # Duplicate predicted logits K=2 times, resulting in (bs, N, K)
        preds_K = preds.repeat(1, 1, K)

        # take BCE for each point, shape: bs, N, 2
        bce_batch_loss = F.binary_cross_entropy_with_logits(
            input=preds_K, target=labels_oh.float(), reduction="none"
        )

        # take mean on the number of points (N) per dataset
        bce_batch_loss = bce_batch_loss.mean(1)

        # take min on the predicted cluster per dataset
        bce_batch_loss, idx = bce_batch_loss.min(1)

        # WHY THIS?
        # bcent[oh_labels.sum(1)==0] = float('inf')
        # bidx = bcent != float('inf')
        # bcent = bcent[bidx].mean()

        bce_batch_loss = bce_batch_loss.mean()
        return bce_batch_loss


if __name__ == "__main__":
    bs = 4
    D = 2
    N = 1000
    X = np.random.normal(size=(bs, N, D))

    Z = np.array(np.random.rand(bs, N) > 0.5, dtype=np.int8)
    X = torch.from_numpy(X)
    Z = torch.from_numpy(Z)

    k = 0.1
    v = 10.0
    mu = np.random.normal(size=(1, D))
    psi = np.random.normal(size=(D, D))
    psi = 0.5 * (psi + psi.T)

    niw_prior = niw_hyperparams(k, v, mu, psi)

    # # print("NIW params:")
    # # print(niw_prior)
    # # print(niw_prior.mu)
    # # print(niw_prior.psi)

    # niw_post = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)
    # niw_post_r = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)
    # niw_post_l = niw_hyperparams(niw_prior.k, niw_prior.v, niw_prior.mu, niw_prior.psi)

    # # calculate sufficient statistics:
    # suff_stats_cp = batch_niw_sufficient_stats(X)
    # suff_stats_cpr = batch_niw_sufficient_stats(torch.mul(X, Z.unsqueeze(-1)))
    # suff_stats_cpl = batch_niw_sufficient_stats(torch.mul(X, (1 - Z).unsqueeze(-1)))

    # print("Suff stats:")
    # print(suff_stats_cp)
    # print("Suff stats R:")
    # print(suff_stats_cpl)
    # print("Suff stats L:")
    # print(suff_stats_cpr)

    # niw_post.update_posterior(suff_stats_cp)
    # niw_post_r.update_posterior(suff_stats_cpr)
    # niw_post_l.update_posterior(suff_stats_cpl)

    # print("Suff stats:")
    # print(niw_post)
    # print("Suff stats R:")
    # print(niw_post_r)
    # print("Suff stats L:")
    # print(niw_post_l)

    # # niw_params_post = copy.copy(niw_params)

    # # niw_params_post.update_posterior(suff_stat)

    # log_likelihood_r = batch_log_marginal_likelihood(
    #     niw_prior, niw_post_r, suff_stats_cpr
    # )
    # log_likelihood_l = batch_log_marginal_likelihood(
    #     niw_prior, niw_post_l, suff_stats_cpl
    # )

    # # print("LLR: ", log_likelihood_r)
    # # print("LLL: ", log_likelihood_l)

    model = nn.Sequential(nn.Linear(D, N), nn.ReLU(), nn.Linear(N, 1))

    # X = np.random.normal(size=(bs, N))
    # X = torch.from_numpy(X)
    X = torch.randn(bs, N, D)

    output = model(X)
    output = (torch.sigmoid(output).round()).squeeze()

    loss = batch_log_hasting_ratio(X, output, niw_prior, 10)
    # loss = my_loss_fn(output, target)
    loss.backward()
    print(model[0].weight.grad)

    # print(f"Log Hasting Ratio calc: {lhr}")
    # print(f"Hasting Ratio calc: {torch.exp(lhr)}")
