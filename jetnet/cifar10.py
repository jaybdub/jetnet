import numpy as np
import tempfile
import PIL.Image
import os
import shutil
import subprocess
from pydantic import BaseModel
from jetnet.classification import ClassificationDataset, Classification
from jetnet.utils import download, make_parent_dir
from jetnet.dataset import DatasetFilter

def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

CIFAR10_TARFILE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


class _CIFAR10:
    
    def __init__(self, folder: str, train: bool = False):
        all_labels = []
        all_images = []
        if train:
            for batch in [1,2,3,4,5]:
                path = os.path.join(folder, f"data_batch_{batch}")
                data = _unpickle(path)
                all_labels += data[b'labels']
                all_images.append(
                    data[b'data']
                )
        else:
            path = os.path.join(folder, "test_batch")
            data = _unpickle(path)
            all_labels += data[b'labels']
            all_images.append(
                data[b'data']
            )

        images = np.concatenate(all_images, axis=0)
        images = np.reshape(images, (images.shape[0], 3, 32, 32))
        images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)))
        self._images = images
        self._labels = np.array(all_labels)

    def get_labels(self):
        return CIFAR10_LABELS

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index):
        image = PIL.Image.fromarray(self._images[index])
        label_idx = int(self._labels[index])
        return image, Classification(index=label_idx, label=self.get_labels()[label_idx])


class CIFAR10(BaseModel):

    path: str = "data/cifar10/cifar-10-batches-py"
    train: bool = False
    url: str = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    def get_labels(self):
        return CIFAR10_LABELS
        
    def build(self):

        if not os.path.exists(self.path):
            tmp = tempfile.mkdtemp()
            tar_file = os.path.join(tmp, 'cifar-10-python.tar.gz')
            download(self.url, tar_file)
            subprocess.call(['tar', '-xvf', tar_file], cwd=tmp)
            tmp_path = os.path.join(tmp, 'cifar-10-batches-py')
            make_parent_dir(self.path)
            shutil.move(tmp_path, self.path)

        return _CIFAR10(self.path, self.train)


CIFAR10_TRAIN = CIFAR10(train=True)
CIFAR10_TEST = CIFAR10(train=False)
CIFAR10_TRAIN_IMAGES = DatasetFilter(dataset=CIFAR10_TRAIN, expr="lambda x: x[0]")
CIFAR10_TEST_IMAGES = DatasetFilter(dataset=CIFAR10_TEST, expr="lambda x: x[0]")