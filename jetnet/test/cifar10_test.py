import pytest
from jetnet.cifar10 import CIFAR10_TEST, CIFAR10_TRAIN


def test_cifar10_test():

    dataset = CIFAR10_TEST.build()

    assert len(dataset) == 10000


def test_cifar10_train():

    dataset = CIFAR10_TRAIN.build()

    assert len(dataset) == 50000
    