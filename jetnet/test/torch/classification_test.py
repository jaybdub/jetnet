from jetnet.torch.classification import TorchvisionClassificationModule, TorchClassificationModel
from jetnet.torch.optimizer import TorchOptimizerAdam
from jetnet.cifar10 import *


RESNET18_MODULE_PRETRAINED = TorchvisionClassificationModule(name='resnet18', pretrained=True)

ADAM_DEFAULT = TorchOptimizerAdam()

RESNET18_ADAM_CIFAR10 = TorchClassificationModel(
    module=RESNET18_MODULE_PRETRAINED,
    dataset=CIFAR10_TRAIN,
    optimizer=ADAM_DEFAULT,
    weights_path="data/torch/test_resnet18_cifar_train.pth",
    epochs=10,
    batch_size=256
)

def test_resnet18_cifar_train():
    model = RESNET18_ADAM_CIFAR10.build()
    

if __name__ == '__main__':
    test_resnet18_cifar_train()