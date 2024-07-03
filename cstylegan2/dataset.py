import math, random, struct

import numpy as np

from torch import tensor, FloatTensor
from torch.utils import data
from torchvision import transforms
from sklearn import preprocessing

from torchvision import datasets
import torch.nn.functional as F

from config import TRAIN_SIZE, VAL_SIZE

# SAMPLES = 0.15  # porcentaje (0.15) o numero de muestras (200)
AUM = 0  # aumentado: 0-sin_aumentado, 1-con_aumentado

DATASETS = [
    "oitaven_river.raw",
    "xesta_basin.raw",
    "eiras_dam.raw",
    "ermidas_creek.raw",
    "ferreiras_river.raw",
    "das_mestas_river.raw",
    "mera_river.raw",
    "ulla_river.raw",
    "pavia_university.raw",
]
GTS = [
    "oitaven_river.pgm",
    "xesta_basin.pgm",
    "eiras_dam.pgm",
    "ermidas_creek.pgm",
    "ferreiras_river.pgm",
    "das_mestas_river.pgm",
    "mera_river.pgm",
    "ulla_river.pgm",
    "pavia_university_gt.pgm",
]
SEGS = [
    "seg_oitaven_wp.raw",
    "seg_xesta_wp.raw",
    "seg_eiras_wp.raw",
    "seg_ermidas_wp.raw",
    "seg_ferreiras_wp.raw",
    "seg_mestas_wp.raw",
    "seg_mera_wp.raw",
    "seg_ulla_wp.raw",
    "pavia_university_seg.raw",
]
CENTERS = [
    "seg_oitaven_wp_centers.raw",
    "seg_xesta_wp_centers.raw",
    "seg_eiras_wp_centers.raw",
    "seg_ermidas_wp_centers.raw",
    "seg_ferreiras_wp_centers.raw",
    "seg_mestas_wp_centers.raw",
    "seg_mera_wp_centers.raw",
    "seg_ulla_wp_centers.raw",
    "pavia_university_seg_centers.raw",
]


def read_raw(fichero):
    (B, H, V) = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=B * H * V, offset=3 * 4, dtype=np.int32)
    print("* Read dataset:", fichero)
    print("  B:", B, "H:", H, "V:", V)
    print("  Read:", len(datos))

    # normalize data to [-1, 1]
    datos = datos.astype(np.float64)
    preprocessing.minmax_scale(datos, feature_range=(-1, 1), copy=False)

    datos = datos.reshape(V, H, B)
    datos = FloatTensor(datos)
    return (datos, H, V, B)


def read_seg(fichero):
    (H, V) = np.fromfile(fichero, count=2, dtype=np.uint32)
    datos = np.fromfile(fichero, count=H * V, offset=2 * 4, dtype=np.uint32)
    print("* Read segmentation:", fichero)
    print("  H:", H, "V:", V)
    print("  Read:", len(datos))
    return (datos, H, V)


def read_seg_centers(fichero):
    (H, V, nseg) = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=H * V, offset=3 * 4, dtype=np.uint32)
    print("* Read centers:", fichero)
    print("  H:", H, "V:", V, "nseg:", nseg)
    print("  Read:", len(datos))
    return (datos, H, V, nseg)


def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print("No puedo abrir ", filename)
        exit(0)
    else:
        f.write(struct.pack("i", B))
        f.write(struct.pack("i", H))
        f.write(struct.pack("i", V))
        output = output.reshape(H * V * B)
        for i in range(H * V * B):
            f.write(struct.pack("i", np.int(output[i])))
        f.close()
        print("* Saved file:", filename)


def read_pgm(fichero):
    try:
        pgmf = open(fichero, "rb")
    except IOError:
        print("No puedo abrir ", fichero)
    else:
        assert pgmf.readline().decode() == "P5\n"
        line = pgmf.readline().decode()
        while line[0] == "#":
            line = pgmf.readline().decode()
        (H, V) = line.split()
        H = int(H)
        V = int(V)
        depth = int(pgmf.readline().decode())
        assert depth <= 255
        raster = []
        for i in range(H * V):
            raster.append(ord(pgmf.read(1)))
        print("* Read GT:", fichero)
        print("  H:", H, "V:", V, "depth:", depth)
        print("  Read:", len(raster))
        return (raster, H, V)


def save_pgm(output, H, V, nclases, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print("No puedo abrir ", filename)
        exit(0)
    else:
        # f.write(b'P5\n')
        cadena = "P5\n" + str(H) + " " + str(V) + "\n" + str(nclases) + "\n"
        f.write(bytes(cadena, "utf-8"))
        f.write(output)
        f.close()
        print("* Saved file:", filename)


def select_training_samples_seg(truth, center, H, V, sizex, sizey, train_size, val_size, seed=None):
    print("* Select training samples")
    # nclases = max(truth)
    nclases = 10
    lista = [[] for _ in range(nclases)]
    xmin = int(sizex / 2)
    xmax = H - int(math.ceil(sizex / 2))
    ymin = int(sizey / 2)
    ymax = V - int(math.ceil(sizey / 2))
    for ind in center:
        i = ind // H
        j = ind % H
        if i < ymin or i > ymax or j < xmin or j > xmax:
            continue
        if truth[ind] > 0:
            lista[truth[ind] - 1].append(ind)

    seed = 0 if seed is None else seed
    for i in range(nclases):
        random.Random(seed + i).shuffle(lista[i])

    label_map = dict()
    k = 1
    for i in range(nclases):
        if len(lista[i]) > 0:
            label_map[i + 1] = k
            k += 1

    # seleccionamos muestras para train, validation, test
    train = []
    validation = []
    test = []
    print("  Class    : seg.tot | train | validation")
    for i in range(nclases):
        if train_size < 1:  # Interpretamos que e unha porcentaxe
            train_samples = int(train_size * len(lista[i]))
        else:
            train_samples = train_size
        if train_samples < 1 and len(lista[i]) > 0:
            train_samples = 1

        if val_size < 1:
            val_samples = int(val_size * len(lista[i]))
        else:
            val_samples = val_size
        if val_samples < 1 and len(lista[i]) > 0:
            val_samples = 1

        if train_samples >= len(lista[i]):  # Deixamos polo menos unha mostra para validacion
            train_samples = len(lista[i]) // 2
            val_samples = len(lista[i]) - train_samples
        elif train_samples + val_samples > len(lista[i]):
            val_samples = len(lista[i]) - train_samples

        for j in range(len(lista[i])):
            test.append(lista[i][j])  # testeamos con todo
            if j < train_samples:
                train.append(lista[i][j])
            elif j < train_samples + val_samples:
                validation.append(lista[i][j])
        print("  Class", f"{i+1:2d}", ":", f"{len(lista[i]):7d}", "|", f"{train_samples:5d}", "|", f"{val_samples:10d}")
    return (train, validation, test, label_map)


def select_patch(datos, sizex, sizey, x, y):
    x1 = x - int(sizex / 2)
    x2 = x + int(math.ceil(sizex / 2))
    y1 = y - int(sizey / 2)
    y2 = y + int(math.ceil(sizey / 2))
    patch = datos[:, y1:y2, x1:x2]
    return patch


class DatasetManager:
    """
    The DatasetManager object is used to manage train, validation and test sets.
    """

    def __init__(self, folder, train=True, val_size=None, download=True, hyperdataset=False, batch_size=64):

        if train == False and val_size is not None:
            raise ValueError("Validation size is only available for train set.")

        if val_size is not None and val_size <= 0:
            raise ValueError("Validation size must be a positive integer.")

        if hyperdataset:
            self.__init_hyper__(folder, batch_size)
        else:
            self.__init_mnist__(folder, train, download, val_size)

        print("* Dataset inicializado.")

    def __init_mnist__(self, folder, train, download, val_size):
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

    def __init_hyper__(self, folder, batch_size):
        hyperdataset_index = self.getHyperdatasetIndex(folder)
        if isinstance(hyperdataset_index, str):
            raise ValueError(hyperdataset_index)
        # Load data
        (datos, H, V, B) = read_raw(f"{folder}/{DATASETS[hyperdataset_index]}")
        (truth, H1, V1) = read_pgm(f"{folder}/{GTS[hyperdataset_index]}")
        (seg, H2, V2) = read_seg(f"{folder}/{SEGS[hyperdataset_index]}")
        # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
        sizex = 32
        sizey = 32
        # necesitamos los datos en band-vector para hacer convoluciones
        datos = np.transpose(datos, (2, 0, 1))

        train_size = TRAIN_SIZE
        val_size = VAL_SIZE

        # 3. Selection training, testing sets
        # (center,nseg)=seg_center(seg,H,V)
        (center, H3, V3, nseg) = read_seg_centers(f"{folder}/{CENTERS[hyperdataset_index]}")

        # We will need them for testing at pixel level
        self.truth = truth
        self.center = center
        self.seg = seg
        self.H = H
        self.V = V

        (train_samples, val_samples, test_samples, label_map) = select_training_samples_seg(
            truth, center, H, V, sizex, sizey, train_size, val_size, seed=1
        )
        fill = True if TRAIN_SIZE < 1 else False
        data_train = HyperDataset(datos, truth, train_samples, H, V, sizex, sizey, label_map, fill=False)
        print("  - train dataset:", len(data_train))
        data_val = HyperDataset(
            datos, truth, val_samples, H, V, sizex, sizey, label_map, fill=fill, batch_size=batch_size
        )
        print("  - val dataset:", len(data_val))
        data_test = HyperDataset(
            datos, truth, test_samples, H, V, sizex, sizey, label_map, fill=fill, batch_size=batch_size
        )
        print("  - test dataset:", len(data_test))

        self.data_train = data_train
        self.data_val = data_val if val_size is not None else None
        self.data_test = data_test

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

    def getHyperdatasetIndex(self, folder):
        name = folder.split("/")[-1]

        name_to_index = {
            "OITAVEN": 0,
            "XESTA": 1,
            "EIRAS": 2,
            "ERMIDAS": 3,
            "FERREIRAS": 4,
            "MESTAS": 5,
            "MERA": 6,
            "ULLA": 7,
            "PAVIA": 8,
        }
        return name_to_index.get(name, "Invalid hyperdataset name.")

    def getImageInfo(self):
        return (self.truth, self.center, self.seg, self.H, self.V)


class HyperDataset(data.Dataset):
    def __init__(self, data, truth, samples, H, V, sizex, sizey, label_map, fill=False, batch_size=64):
        self.data = data
        self.truth = truth
        self.samples = samples
        self.H = H
        self.V = V
        self.sizex = sizex
        self.sizey = sizey
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])

        self.label_map = label_map
        self.label_dim = len(set(truth)) - 1

        self.fill_size = batch_size - (len(self.samples) % batch_size)
        self.length = len(self.samples) + self.fill_size if fill else len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            idx = idx % len(self.samples)  # needed to fill the batch size
        datos = self.data
        truth = self.truth
        H = self.H
        V = self.V
        sizex = self.sizex
        sizey = self.sizey
        x = self.samples[idx] % H
        y = int(self.samples[idx] / H)
        patch = select_patch(datos, sizex, sizey, x, y)

        if AUM == 1:
            patch = self.transform(patch)

        label_index = self.label_map[truth[self.samples[idx]]] - 1  # to classify from 0 to n-1
        label = F.one_hot(tensor(label_index), num_classes=self.label_dim).float().cuda()

        return patch, label


def cycle(iterable):
    """
    Transform an iterable into a generator.
    """
    while True:
        for i in iterable:
            yield i


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
