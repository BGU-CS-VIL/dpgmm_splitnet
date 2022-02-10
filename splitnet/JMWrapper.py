import torch

from src.models.models import (
    SplitTransformerV1,
    SplitTransformer,
    SplitTransformerV2,
)
import numpy as np

from hydra import compose, initialize

from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
import omegaconf

from pathlib import Path


class JuliaModelWrapper:
    def __init__(self, ckpt_path, gpu: bool = False) -> None:
        ckpt_path = Path(ckpt_path)
        config_dir = str(ckpt_path.parent)

        cfg = omegaconf.OmegaConf.load(config_dir + "/config.yaml")

        print(
            f"Running inference with the following config:\n{OmegaConf.to_yaml(cfg.model)}"
        )

        print("Loading model...")
        self.model_name = cfg.model._target_
        self.D = cfg.datamodule.cfg.data_props.train.dimension
        if gpu:
            self.device_ = "cuda"
        else:
            self.device_ = "cpu"
        if "SplitTransformerV1" in self.model_name:
            self.model = SplitTransformerV1.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                cfg=cfg,
                **cfg.model,
                device=self.device_,
            )
        elif "SplitTransformerV2" in self.model_name:
            self.model = SplitTransformerV2.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                cfg=cfg,
                **cfg.model,
                device=self.device_,
            )
        elif "SplitTransformer" in self.model_name:
            self.model = SplitTransformer.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                cfg=cfg,
                **cfg.model,
                device=self.device_,
            )
        self.model = self.model.to(self.device_)

        self.model.eval()

    def test(self):
        x = torch.rand(1, 10, 1000).to(self.device_)
        print("Output Shape:")
        print(self.model.forward(x)[0].shape)

    def infer(self, x):
        Dx = x.shape[0]
        x = np.array(x).transpose()

        x = StandardScaler().fit_transform(x)
        x = torch.tensor(x).unsqueeze(0).to(self.device_)

        if Dx < self.D:
            x = x.repeat(1, 1, int(np.ceil(self.D - Dx) + 1))
            x = x[:, :, : self.D]

        assert x.shape[2] == self.D
        with torch.no_grad():
            y = torch.sigmoid(self.model.forward(x)).squeeze(0).cpu().numpy()
        return y
