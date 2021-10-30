from pathlib import Path

import wandb
import hydra
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.optim
import torchvision.transforms as T
from hydra.utils import get_original_cwd
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def create_transform(image_size=224, training=True):
    tf = [T.Resize((image_size,) * 2)]
    if training:
        tf.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )

    tf.extend(
        [
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return T.Compose(tf)


class PetDataset(Dataset):
    def __init__(self, image_path, labels=None, transform=None):
        assert len(image_path) == len(labels)
        self.image_path = image_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = read_image(self.image_path.iloc[index])
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels.iloc[index]
        return image


def create_dataset(df, transform_cfg, training):
    return PetDataset(
        df['path'],
        df['Pawpularity'],
        transform=create_transform(training=training, **transform_cfg),
    )


class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, datamodule_cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['train_df', 'val_df'])
        self.cfg = datamodule_cfg
        self.train_df = train_df
        self.val_df = val_df

    def train_dataloader(self):
        return DataLoader(
            create_dataset(self.train_df, self.cfg.transform, training=True),
            shuffle=True,
            drop_last=True,
            **self.cfg.dataloader,
        )

    def val_dataloader(self):
        return DataLoader(
            create_dataset(self.val_df, self.cfg.transform, training=False),
            shuffle=False,
            **self.cfg.dataloader,
        )


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.backbone = timm.create_model(**self.cfg.backbone)
        self.fc = nn.Sequential(
            nn.Dropout(self.cfg.fc_dropout), nn.Linear(self.backbone.num_features, self.cfg.output_dim)
        )
        self.criterion = hydra.utils.instantiate(cfg.loss)

    def forward(self, x):
        output = self.backbone(x)
        return self.fc(output)

    def shared_step(self, batch, prefix=''):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(1).float() / 100
        loss = self.criterion(pred, y)
        self.log(f'{prefix}loss', loss)
        return {'loss': loss, 'pred': 100*pred.detach(), 'label': 100*y.detach()}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train_')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val_')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.backbone(batch)

    def shared_epoch_end(self, outputs, prefix=''):
        pred = torch.cat([out['pred'] for out in outputs])
        label = torch.cat([out['label'] for out in outputs])
        rmse = torch.sqrt(((label - pred) ** 2).mean())
        self.log(f'{prefix}rmse', rmse)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'train_')

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'val_')

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        if 'lr_scheduler' not in self.cfg:
            return optimizer
        lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer=optimizer)
        return [optimizer], [lr_scheduler]


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    seed_everything(cfg.general.seed)
    cwd = Path(get_original_cwd())
    df = pd.read_csv(cfg.data.data_path)
    df['path'] = df['Id'].map(
        lambda i: str(cwd / Path(cfg.data.data_path).parent / f'train/{i}.jpg')
    )
    skf = StratifiedKFold(cfg.data.n_splits, shuffle=True, random_state=cfg.general.seed)
    for i, (train_idx, val_idx) in enumerate(skf.split(df.index, df['Pawpularity'])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = DataModule(train_df, val_df, cfg.datamodule)
        model = Model(cfg.model)
        early_stopping = EarlyStopping(monitor='val_rmse')
        lr_monitor = LearningRateMonitor()
        model_checkpoint = ModelCheckpoint(
            filename='best', monitor='val_rmse', save_top_k=1, mode='min'
        )
        logger = WandbLogger(cfg.wandb.name + f'_{i}', project=cfg.wandb.project, log_model='all', group=cfg.wandb.name+'_cv')
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[lr_monitor, early_stopping, model_checkpoint],
            **cfg.trainer,
        )
        trainer.fit(model, datamodule=datamodule)
        hydra_artifact = wandb.Artifact('hydra', 'setting file')
        hydra_artifact.add_dir(Path.cwd())


if __name__ == '__main__':
    main()