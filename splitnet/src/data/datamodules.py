from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader

from src.data.data import genereate_splitnet_dataset
from src.data.datasets import SplitDataset
from src.utils.metrics import niw_hyperparams


class SyntheticSplitData(pl.LightningDataModule):
    """
    To define a DataModule define 5 methods:
    - prepare_data (how to download(), tokenize, etc…)
    - setup (how to split, etc…)
    - train_dataloader
    - val_dataloader(s)
    - test_dataloader(s)

    Args:
        pl ([type]): [description]
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg

        # for carriculum learning
        if cfg.curriculum_schedule:
            self.init_params = self.config.data_props.train
            curriculum_params = self.config.data_props.train.curriculum_params

            self.sample_size_train = int(
                curriculum_params.update_every
                / curriculum_params.total_epochs
                * cfg.train_samples
            )

            self.sample_size_valid = int(
                curriculum_params.update_every
                / curriculum_params.total_epochs
                * cfg.valid_samples
            )

            init_alpha = self.init_params.data_gen_alpha
            init_kappa = self.init_params.kappa
            init_nu = self.init_params.nu

            final_alpha = curriculum_params.final_data_gen_alpha
            final_kappa = curriculum_params.final_kappa
            final_nu = curriculum_params.final_nu

            update_every = self.config.data_props.train.curriculum_params.update_every
            self.alphas = np.linspace(
                init_alpha, final_alpha, curriculum_params.total_epochs // update_every
            )
            self.kappas = np.linspace(
                init_kappa, final_kappa, curriculum_params.total_epochs // update_every
            )
            self.nus = np.linspace(
                init_nu, final_nu, curriculum_params.total_epochs // update_every
            )

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be
        done only from a single GPU in distributed settings.
        - download
        - tokenize
        - etc…
        """

        alpha = self.config.data_props.train.data_gen_alpha
        dp_alpha = self.config.data_props.train.dp_alpha
        kappa = self.config.data_props.train.kappa
        nu = self.config.data_props.train.nu
        psi = np.eye(self.config.data_props.train.dimension)
        mu = np.zeros(self.config.data_props.train.dimension)

        niw_prior = niw_hyperparams(k=kappa, v=nu, psi=psi, mu=mu)
        print("Generating Train Data")
        X, Y = genereate_splitnet_dataset(
            num_points=self.config.data_props.train.num_points,
            niw_params=niw_prior,
            alpha=alpha,
            dp_alpha=dp_alpha,
            dataset_size=self.config.train_samples,
            hr_threshold=self.config.data_props.train.hr_threshold,
            multi_k=self.config.multi_k,
            K_max=self.config.K_max,
        )
        self.train_ds = SplitDataset(X, Y, permute=True)

        print("Generating Validation Data")
        X, Y = genereate_splitnet_dataset(
            num_points=self.config.data_props.train.num_points,
            niw_params=niw_prior,
            alpha=alpha,
            dp_alpha=dp_alpha,
            dataset_size=self.config.valid_samples,
            hr_threshold=self.config.data_props.train.hr_threshold,
            use_hr_threshold=self.config.use_hr_threshold,
            multi_k=self.config.multi_k,
            K_max=self.config.K_max,
        )

        self.valid_ds = SplitDataset(X, Y, permute=False)

        self.test_tags = []
        self.test_ds = []
        for i, test_set in enumerate(self.config.data_props.test):

            alpha = test_set.data_gen_alpha
            dp_alpha = test_set.dp_alpha
            kappa = test_set.kappa
            nu = test_set.nu
            psi = np.eye(test_set.dimension)
            mu = np.zeros(test_set.dimension)

            test_niw_prior = niw_hyperparams(k=kappa, v=nu, psi=psi, mu=mu)

            print(f"Generating Test Data #-{i}")
            X, Y = genereate_splitnet_dataset(
                num_points=test_set.num_points,
                niw_params=test_niw_prior,
                alpha=alpha,
                dp_alpha=dp_alpha,
                dataset_size=self.config.test_samples,
                use_hr_threshold=self.config.use_hr_threshold,
                multi_k=self.config.multi_k,
                K_max=self.config.K_max,
            )
            # self.test_tags.append(test_set.set_tag)
            dataset = SplitDataset(X, Y)
            dataset.test_tag = test_set.set_tag

            self.test_ds.append(dataset)

    def setup(self, stage: Optional[str]):
        """There are also data operations you might want to perform on every GPU.
        Use setup to do things like:

        - count number of classes
        - build vocabulary
        - perform train/val/test splits
        - apply transforms (defined explicitly in your datamodule or assigned in init)
        - etc…
        """
        # Assign Train/val split(s) for use in Dataloaders
        # if stage == "fit" or stage is None:

        # Assign Test split(s) for use in Dataloaders
        # if stage == "test" or stage is None:
        pass

    def update_train_data(self, index) -> None:

        print(f"Generating new data with index: {index}")

        current_alpha = int(self.alphas[index])
        current_kappa = self.kappas[index]
        current_nu = int(self.nus[index])

        # fixed params:
        dp_alpha = self.init_params.dp_alpha
        psi = np.eye(self.init_params.dimension)
        mu = np.zeros(self.init_params.dimension)

        new_niw_prior = niw_hyperparams(k=current_kappa, v=current_nu, psi=psi, mu=mu)

        newX, newY = genereate_splitnet_dataset(
            num_points=self.config.data_props.train.num_points,
            niw_params=new_niw_prior,
            alpha=current_alpha,
            dp_alpha=dp_alpha,
            dataset_size=self.sample_size_train,
            hr_threshold=self.init_params.hr_threshold,
            use_hr_threshold=self.config.use_hr_threshold,
            multi_k=self.config.multi_k,
            K_max=self.config.K_max,
        )

        # choose and update train dataset
        a, b = index * self.sample_size_train, (index + 1) * self.sample_size_train
        self.train_ds.X[a:b] = newX
        self.train_ds.y[a:b] = newY

        # For validation data:
        update_every = self.config.valid_samples // self.sample_size_valid
        newX, newY = genereate_splitnet_dataset(
            num_points=self.config.data_props.train.num_points,
            niw_params=new_niw_prior,
            alpha=current_alpha,
            dp_alpha=dp_alpha,
            dataset_size=self.sample_size_valid,
            hr_threshold=self.init_params.hr_threshold,
            use_hr_threshold=self.config.use_hr_threshold,
            multi_k=self.config.multi_k,
            K_max=self.config.K_max,
        )

        # choose and update validation dataset
        self.valid_ds.X[index::update_every] = newX
        self.valid_ds.y[index::update_every] = newY

    def train_dataloader(self, *args, **kwargs) -> DataLoader:

        if self.config.curriculum_schedule:
            update_every = self.config.data_props.train.curriculum_params.update_every
            if (self.trainer.current_epoch) % update_every == 0:
                self.update_train_data(self.trainer.current_epoch // update_every)

        return DataLoader(self.train_ds, shuffle=True, **self.config.dataloaders)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_ds, shuffle=False, **self.config.dataloaders)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        test_dls = []
        for dataset in self.test_ds:
            test_dls.append(
                DataLoader(dataset, shuffle=False, **self.config.dataloaders)
            )

        return test_dls
