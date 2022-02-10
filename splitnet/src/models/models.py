from pathlib import Path

import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from src.loss.losses import HastingsRatioLoss, SetLoss
from src.models.modules import ISAB, MAB, PMA, SAB, PMAv2
from src.utils.metrics import (
    calc_hastings_ratio_gt,
    calc_hastings_ratio_kmeans,
    niw_hyperparams,
)
from torchmetrics.functional import accuracy, fbeta


def plot_predictions(
    X: np.ndarray, labels: np.ndarray, current_epoch: int, num_ax: int = 3
):
    """[summary]

    Args:
        X (np.ndarray): Data array, shape: [bs, D, N]
        y_pred (np.ndarray): labels. sjape: [bs, N]
        current_epoch (int): current epoch number
        num_ax (int, optional): number of plots per axis. Defaults to 3.

    Returns:
        [type]: [description]
    """
    fig = plt.figure(figsize=(3 * num_ax, 3 * num_ax), facecolor="white")

    for i in range(num_ax ** 2):
        splot = plt.subplot(num_ax, num_ax, i + 1)
        # targets = y_pred.argmax(-1)
        plt.scatter(
            X[i, 0, labels[i] == 0],
            X[i, 1, labels[i] == 0],
            c="b",
            alpha=0.3,
            s=20,
        )
        plt.scatter(
            X[i, 0, labels[i] == 1],
            X[i, 1, labels[i] == 1],
            c="r",
            alpha=0.3,
            s=20,
        )

        # plt.legend()
        plt.title(f"Pred for epoch {current_epoch}")
        plt.axis("tight")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def plot_hastings_histograms(hist_dict, niw_prior, alpha, current_epoch):

    # fig = plt.figure()
    sns_fig = sns.displot(hist_dict, kind="kde")

    plt.title(f"Hastings Ratio Histogram | epoch: {current_epoch}")
    plt.xlabel("Log Hastings Ratio")
    handles = [
        mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
    ] * 2

    labels = []
    labels.append(
        f"NIW Prior: \n kappa={niw_prior.k} \n nu={niw_prior.v} \n D={niw_prior.psi.shape[0]}"
    )
    labels.append(f"alpha={alpha}")

    plt.legend(
        handles,
        labels,
        loc="best",
        fontsize="small",
        fancybox=True,
        framealpha=0.7,
        handlelength=0,
        handletextpad=0,
    )

    plt.show()
    return sns_fig.fig


class SplitTransformer(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        dim_input: int = 2,
        pma_seeds: int = 1,
        num_inds: int = 16,
        dim_hidden: int = 32,
        num_heads: int = 2,
        num_encoders: int = 2,
        num_decoders: int = 2,
        ln: bool = True,
    ):
        super(SplitTransformer, self).__init__()

        self.config = cfg

        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            *[ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)]
            * (num_encoders - 1),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            *[PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln)]
            * num_decoders,
            nn.Dropout(),
            nn.Linear(dim_hidden, 1),
        )

        # init prior params, maybe temp here:
        self.alpha = self.config.datamodule.cfg.data_props.train.dp_alpha
        kappa = self.config.datamodule.cfg.data_props.train.kappa
        nu = self.config.datamodule.cfg.data_props.train.nu
        psi = np.eye(self.config.datamodule.cfg.data_props.train.dimension)
        mu = np.zeros(self.config.datamodule.cfg.data_props.train.dimension)

        self.niw_prior = niw_hyperparams(k=kappa, v=nu, psi=psi, mu=mu)

        self.log_hr_loss = HastingsRatioLoss(self.niw_prior, self.alpha)

    def on_train_start(self) -> None:
        super().on_train_start()
        if "DummyLogger" not in str(self.logger.__class__):
            for callback in self.trainer.callbacks:
                if "ModelCheckpoint" in str(callback.__class__):
                    log_dir = callback.dirpath + f"/{self.logger.experiment_id}"
                    callback.dirpath = log_dir
                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(DictConfig(self.config), f=log_dir + "/config.yaml")

    def forward(self, X):
        Z = self.enc(X)
        # return self.dec(Z).squeeze()
        return self.dec(Z)

    def sample_batch(self, batch):
        N_max = batch[0].shape[1]
        N = int(np.random.uniform(low=1000, high=N_max, size=1))
        indices = np.random.choice(np.arange(N_max), size=N, replace=False)

        batch[0] = batch[0][:, indices]
        batch[1] = batch[1][:, indices]
        batch[2] = batch[2][:, indices]

        return batch

    def calc_metrics(self, preds, targets, tag):
        metrics_dict = {}

        # # shape back to [B, N, 2]:
        # preds = preds.reshape(*self.batch_shape[:2], 2)

        # For BCE:
        preds = preds.reshape(self.batch_shape[:2])
        targets = targets.reshape(self.batch_shape[:-1])

        accs = []
        nmis = []
        aris = []

        inv_targets = -1 * targets + 1

        # For BCE:
        sm_preds = torch.sigmoid(preds).round()

        for i in range(self.batch_shape[0]):
            nmi = nmi_score(
                targets[i].detach().cpu().numpy(),
                sm_preds[i].detach().cpu().numpy(),
            )

            ari = ari_score(
                targets[i].detach().cpu().numpy(), sm_preds[i].detach().cpu().numpy()
            )

            acc = accuracy(sm_preds[i], targets[i])
            inv_acc = accuracy(sm_preds[i], inv_targets[i])
            acc = torch.max(acc, inv_acc)

            accs.append(acc.detach().cpu().numpy())
            nmis.append(nmi)
            aris.append(ari)

        acc = np.mean(accs)
        ari = np.mean(aris)
        nmi = np.mean(nmis)

        metrics_dict[f"{tag}/nmi"] = nmi
        metrics_dict[f"{tag}/ARI"] = ari
        metrics_dict[f"{tag}/acc"] = acc

        return metrics_dict

    def step(self, batch):
        points, _, target = batch

        self.batch_shape = points.shape
        bsz = points.shape[0]

        logits = self.forward(points)
        loss = SetLoss().forward(logits, target)

        pred = logits.view(bsz, -1)
        pred_label = torch.sigmoid(pred)
        # ll_loss = self.log_hr_loss(points, pred_label)
        ll_loss = 0

        target = target.view(-1, 1).long().squeeze()
        return loss, pred, target, ll_loss

    def training_step(self, batch, batch_idx):

        if self.config.datamodule.cfg.resample_batch:
            batch = self.sample_batch(batch)

        loss, preds, targets, ll_loss = self.step(batch)

        metrics_dict = self.calc_metrics(preds, targets, "train")

        self.log_dict(metrics_dict, on_step=False, on_epoch=True)

        if self.config.loss.HR_loss:
            gamma = (
                np.linspace(0, 1, self.config.training.max_epochs)[self.current_epoch]
                * self.config.loss.gamma
            )
            # log_hr_loss = self.log_hr_loss(batch[0], preds)
            self.log("train/loss_hr", ll_loss)
            loss = loss + gamma * ll_loss
            # loss = ll_loss

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        # return ll_loss
        return loss

    def validation_step(self, batch, batch_idx):

        # loss, preds, targets = self.step(batch)
        loss, preds, targets, ll_loss = self.step(batch)

        metrics_dict = self.calc_metrics(preds, targets, "valid")

        self.log_dict(metrics_dict, on_step=False, on_epoch=True)

        if self.config.loss.HR_loss:
            gamma = (
                np.linspace(0, 1, self.config.training.max_epochs)[self.current_epoch]
                * self.config.loss.gamma
            )
            self.log("valid/loss_hr", ll_loss)

            loss = loss + gamma * ll_loss
            loss = ll_loss

        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:

        dim = self.config.datamodule.cfg.data_props.train.dimension

        if (self.current_epoch % 3) == 1:
            self.eval()
            num_ax = 2

            X, X_unorm, y = iter(self.val_dataloader()).__next__()
            X = X.to(self.device)[: num_ax ** 2]
            y = y.cpu().numpy()[: num_ax ** 2]

            with torch.no_grad():
                y_pred = torch.sigmoid(self.forward(X)).cpu().numpy().round().squeeze()

            if dim == 2:
                fig = plot_predictions(
                    X_unorm.transpose(2, 1).cpu().numpy(),
                    y_pred,
                    self.current_epoch,
                    num_ax=num_ax,
                )
            else:
                pca = PCA(n_components=2)
                X_ = X_unorm.cpu().numpy()
                bs, N, D = X_.shape

                x_pca = np.zeros((bs, N, 2))
                for i in range(bs):
                    x_pca[i] = pca.fit_transform(X_[i])
                print("SHAPE:")
                print(x_pca.shape)
                fig = plot_predictions(
                    # x_pca,
                    x_pca.transpose(0, 2, 1),
                    y_pred,
                    self.current_epoch,
                    num_ax=num_ax,
                )

            self.logger.experiment.log_image(
                "Predictions", fig, description=f"for epoch: {self.current_epoch}"
            )

            plt.close()

            batch = iter(self.val_dataloader()).__next__()
            X = batch[0].to(self.device)

            with torch.no_grad():
                y_pred = torch.sigmoid(self.forward(X)).cpu().numpy().round().squeeze()

            X = batch[1].cpu().numpy()
            Y = batch[2].cpu().numpy()
            model_hr = calc_hastings_ratio_gt(X, y_pred, self.niw_prior, self.alpha)

            res_dict = {}
            res_dict["ground truth"] = calc_hastings_ratio_gt(
                X, Y, self.niw_prior, self.alpha
            )
            res_dict["kmeans"] = calc_hastings_ratio_kmeans(
                X, self.niw_prior, self.alpha
            )

            res_dict["model"] = model_hr

            fig = plot_hastings_histograms(
                res_dict,
                self.niw_prior,
                self.alpha,
                self.current_epoch,
            )

            self.logger.experiment.log_image(
                "Log - Hastings Ratio Histograms",
                fig,
                description=f"for epoch: {self.current_epoch}",
            )

            plt.close()

    def test_step(self, batch, batch_idx, dataloader_idx):
        dim = self.config.datamodule.cfg.data_props.train.dimension
        tag = self.test_dataloader()[dataloader_idx].dataset.test_tag

        # loss, preds, targets = self.step(batch)
        loss, preds, targets, ll_loss = self.step(batch)
        if batch_idx == 0:
            X_, X_unorm, _ = batch

            with torch.no_grad():
                y_pred = torch.sigmoid(self.forward(X_)).cpu().numpy().round().squeeze()

            if dim == 2:
                fig = plot_predictions(
                    X_unorm.transpose(2, 1).cpu().numpy(), y_pred, 0, num_ax=4
                )
            else:
                pca = PCA(n_components=2)
                X_ = X_unorm.cpu().numpy()
                # X_ = X_unorm.transpose(2, 1).cpu().numpy()
                bs, N, D = X_.shape

                x_pca = np.zeros((bs, N, 2))
                for i in range(bs):
                    x_pca[i] = pca.fit_transform(X_[i])

                fig = plot_predictions(
                    x_pca.transpose(0, 2, 1),
                    # x_pca,
                    y_pred,
                    self.current_epoch,
                    num_ax=4,
                )

            self.logger.experiment.log_image(
                f"Predictions | Test set: {tag}", fig, description=f"testset"
            )

            plt.close()

            X = batch[1].cpu().numpy()
            Y = batch[2].cpu().numpy()
            model_hr = calc_hastings_ratio_gt(X, y_pred, self.niw_prior, self.alpha)

            res_dict = {}
            res_dict["ground truth"] = calc_hastings_ratio_gt(
                X, Y, self.niw_prior, self.alpha
            )
            res_dict["kmeans"] = calc_hastings_ratio_kmeans(
                X, self.niw_prior, self.alpha
            )

            res_dict["model"] = model_hr

            fig = plot_hastings_histograms(
                res_dict,
                self.niw_prior,
                self.alpha,
                0,
            )

            self.logger.experiment.log_image(
                f"Log-HR Histograms |  Test set: {tag}",
                fig,
                description=f"for epoch: {self.current_epoch}",
            )

            plt.close()

        metrics_dict = self.calc_metrics(preds, targets, f"test:{tag}")

        self.log(f"test:{tag}/loss", loss)
        self.log_dict(metrics_dict)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


class SplitTransformerV1(SplitTransformer):
    def __init__(
        self,
        cfg: DictConfig,
        dim_input: int = 1,
        pma_seeds: int = 1,
        class_num: int = 2,
        num_inds: int = 16,
        dim_hidden: int = 32,
        num_heads: int = 4,
        ln: bool = True,
    ):
        super(SplitTransformerV1, self).__init__(cfg)
        self.config = cfg

        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln),
            PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln),
            PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln),
            PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln),
            nn.Linear(dim_hidden, class_num),
        )

        # init prior params, maybe temp here:
        self.alpha = self.config.datamodule.cfg.data_props.train.dp_alpha
        kappa = self.config.datamodule.cfg.data_props.train.kappa
        nu = self.config.datamodule.cfg.data_props.train.nu
        psi = np.eye(self.config.datamodule.cfg.data_props.train.dimension)
        mu = np.zeros(self.config.datamodule.cfg.data_props.train.dimension)

        self.niw_prior = niw_hyperparams(k=kappa, v=nu, psi=psi, mu=mu)

    def forward(self, X):
        Z = self.enc(X)
        return self.dec(Z)


class SplitTransformerV2(SplitTransformer):
    def __init__(
        self,
        cfg: DictConfig,
        dim_input: int = 1,
        pma_seeds: int = 1,
        class_num: int = 2,
        num_inds: int = 16,
        dim_hidden: int = 32,
        num_heads: int = 4,
        ln: bool = True,
        num_encoders: int = 2,
        num_decoders: int = 2,
    ):
        super(SplitTransformerV2, self).__init__(cfg)
        self.config = cfg

        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            *[ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)]
            * (num_encoders - 1),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMAv2(dim_hidden, dim_hidden, num_heads, pma_seeds, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, 1),
        )

        # init prior params, maybe temp here:
        self.alpha = self.config.datamodule.cfg.data_props.train.dp_alpha
        kappa = self.config.datamodule.cfg.data_props.train.kappa
        nu = self.config.datamodule.cfg.data_props.train.nu
        psi = np.eye(self.config.datamodule.cfg.data_props.train.dimension)
        mu = np.zeros(self.config.datamodule.cfg.data_props.train.dimension)

        self.niw_prior = niw_hyperparams(k=kappa, v=nu, psi=psi, mu=mu)

    def forward(self, X):
        Z = self.enc(X)
        return self.dec(Z)
