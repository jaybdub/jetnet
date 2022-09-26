from jetnet.classification import (
    ClassificationDataset,
    ClassificationModel,
    Classification
)
from jetnet.utils import make_parent_dir
import os
import torch
import torch.optim
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models
from torch.utils.data import DataLoader
from progressbar import ETA, Bar, ProgressBar, Timer
from typing import Optional, Tuple, Literal
from pydantic import BaseModel
from jetnet.torch.module import TorchModule
from jetnet.torch.optimizer import TorchOptimizer


class TorchClassificationModule(TorchModule):
    
    def update_classification_head_size(self, module, num_labels: int):
        pass


class TorchvisionClassificationModule(TorchClassificationModule):

    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenet_v2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201"
    ]
    pretrained: bool = False

    def build(self):
        return getattr(torchvision.models, self.name)(pretrained=self.pretrained)
    
    def update_classification_head_size(self, module, size: int):
        if self.name in ["resnet18", "resnet34"]:
            module.fc = torch.nn.Linear(512, size)
        elif self.name in ["resnet50", "resnet101", 'resnet152']:
            module.fc = torch.nn.Linear(512 * 4, size)
        else:
            raise NotImplementedError
        return module


class _TorchClassificationDataset:
    def __init__(self, dataset: ClassificationDataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        image, classification = self._dataset[index]
        image = self._transform(image)
        return (image, classification.index)


class _TorchClassificationModelImpl:

    def __init__(self, module, labels, transform, device):
        self._module = module.eval().to(device)
        self._labels = labels
        self._transform = transform
        self._device = device

    def get_labels(self):
        return self._labels

    def __call__(self, x):
        with torch.no_grad():
            x = self._transform(x).to(self._device)[None, ...]
            y = self._module(x)
            y = torch.softmax(y, dim=-1).cpu()
            idx = int(torch.argmax(y[0]))
            label = self.get_labels()[idx]
            score = y[0, idx]
            return Classification(index=idx, label=label, score=score)
        

class TorchClassificationModel(BaseModel):

    module: BaseModel
    init_weights_path: Optional[str] = None
    weights_path: Optional[str] = None
    dataset: BaseModel
    optimizer: TorchOptimizer
    epochs: int
    batch_size: int = 1
    shuffle: bool = False
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 1
    
    def _build_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        ])

    def get_labels(self):
        return self.dataset.get_labels()

    def load(self):
        module = self.module.build()
        module = self.module.update_classification_head_size(module, len(self.get_labels()))
        module.load_state_dict(torch.load(self.weights_path))
        device = torch.device("cuda")
        return _TorchClassificationModelImpl(module, self.get_labels(), self._build_transform(), device)

    def train(self):

        module = self.module.build()
        module = self.module.update_classification_head_size(module, len(self.get_labels()))

        if self.init_weights_path is not None:
            module.load_state_dict(torch.load(self.init_weights_path))

        dataset = _TorchClassificationDataset(self.dataset.build(), self._build_transform())
        optimizer = self.optimizer.build(module.parameters())
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        device = torch.device("cuda")

        module = module.train().to(device)

        net_loss = 0.0
        num_correct = 0
        total_num = 0
        for epoch in range(self.epochs):

            pbar = ProgressBar(
                maxval=len(loader),
                widgets=[f"Epoch {epoch}: [", Timer(), "] ", Bar(), " (", ETA(), ") "],
            )

            pbar.start()
            for batch_idx, (image, target) in enumerate(iter(loader)):

                image = image.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                output = module(image)

                loss = F.cross_entropy(output, target)

                loss.backward()
                optimizer.step()

                net_loss += float(loss)
                pbar.update(batch_idx)
                num_correct += int(torch.count_nonzero(torch.argmax(output, dim=-1) == target))
                total_num += len(target)

            net_loss /= len(loader)
            accuracy = 100.0 * num_correct / total_num
            print(f"\nEpoch {epoch}: Loss: {net_loss}, Accuracy: {accuracy}\n")
            pbar.finish()

            if (epoch % self.checkpoint_interval == 0) and self.checkpoint_dir is not None:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                make_parent_dir(checkpoint_path)
                torch.save({
                    "epoch": epoch,
                    "module_state": module.state_dict(), 
                    "optimizer_state": optimizer.state_dict()
                }, checkpoint_path)
        
        if self.weights_path is not None:
            make_parent_dir(self.weights_path)
            torch.save(module.state_dict(), self.weights_path)
        
        device = torch.device("cuda")

        return _TorchClassificationModelImpl(module, self.get_labels(), self._build_transform(), device)

    def build(self):
        if self.weights_path is not None and os.path.exists(self.weights_path):
            return self.load()
        else:
            return self.train()

