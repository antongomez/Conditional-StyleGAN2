from torch import tensor
from torch.utils import data
from torchvision import transforms

from torchvision import datasets
import torch.nn.functional as F


def cycle(iterable):
    """
    Transform an iterable into a generator.
    """
    while True:
        for i in iterable:
            yield i

class DatasetManager():
    """
    The DatasetManager object is used to manage train, validation and test sets.
    """

    def __init__(self, folder, train=True, val_size=None, download=False):

        if train == False and val_size is not None:
            raise ValueError("Validation size is only available for train set.")
        
        if val_size is not None and val_size <= 0:
            raise ValueError("Validation size must be a positive integer.")

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root=folder, train=train, transform=transform, download=download)
        classes = dataset.classes

        if train:
            self.data_test = None
            if val_size is not None:
                data_train, data_val = data.random_split(dataset, [len(dataset) - val_size, val_size])
                self.data_train = Dataset(data_train, classes)
                self.data_val = Dataset(data_val, classes)
            else:
                self.data_train = Dataset(dataset, classes)
                self.data_val = None
        else:
            self.data_train = None
            self.data_val = None
            self.data_test = Dataset(dataset, classes)

    def get_train_set(self):
        if self.data_train is None:
            raise ValueError("Train set was not initialized.")
        return self.data_train
    
    def get_validation_set(self):
        if self.data_val is None:
            raise ValueError("Validation set was not initialized.")
        return self.data_val
    
    def get_test_set(self):
        if self.data_test is None:
            raise ValueError("Test set was not initialized.")
        return self.data_test
            

class Dataset(data.Dataset):
    """
    The Dataset object is used to read files from a given folder and generate both the labels and the tensor.
    """

    def __init__(self, data, labels):
        """
        Initialize the Dataset.

        :param folder: the path to the folder containing either pictures or subfolder with pictures.
        :type folder: str
        """
        super().__init__()
        
        self.data = data
        self.labels = labels
        self.label_dim = len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label_index = self.data[index]
        # Add padding to make the image 32x32
        image_padded = F.pad(image, (2, 2, 2, 2), value=-1)
        # Encode label as one hot encoding vector
        label = F.one_hot(tensor(label_index), num_classes=self.label_dim).float().cuda()
        return image_padded, label
