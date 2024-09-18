<!--
SPDX-FileCopyrightText: 2024, Antón Gómez López

SPDX-License-Identifier: GPL-3.0-or-later
-->

# StyleGAN2 Conditioned in Pytorch for Multispectral Image Classification

This repository is a conditioned version of [lucidrains' repository](https://github.com/lucidrains/stylegan2-pytorch). The paper presenting the StyleGAN2 architecture can be found [here](https://arxiv.org/abs/1912.04958).

## Project Description :ledger:

This network was used to classify multispectral images of Galician rivers obtained through remote sensing. The images were taken with the aim of automatically monitoring the river basins of Galicia. Specifically, the goal is to determine the state of the vegetation, identify areas occupied by invasive species, and detect artificial structures that occupy the river basin using multispectral images.

In particular, these are 8 high-resolution multispectral images that were used in [this paper](https://www.mdpi.com/2072-4292/13/14/2687) to evaluate various segmentation and classification algorithms. The images consist of five bands corresponding to the following wavelengths: 475 nm (blue), 560 nm (green), 668 nm (red), 717 nm (red-edge), and 840 nm (near-infrared).

These datasets present two main challenges:

- Large sample imbalance between classes.
- Scarcity of samples for some classes.

Thus, this GAN was used to try to address these issues by augmenting the dataset with the generator and using the discriminator as a classifier. The results obtained were worse than expected and similar to those achieved using a standard convolutional network.

### Experiment Replication :bar_chart:

The network's ability to classify multispectral images was evaluated in the following bachelor's thesis: [Effective adaptation of generative adversarial networks for processing multidimensional remote sensing images](https://nubeusc-my.sharepoint.com/:b:/g/personal/anton_gomez_lopez_rai_usc_es/EbY99we4GYRIsw4A0Zq3nhEBSVcDZ19kQSEA426UbsMTBg?e=Rk6hR8). In this work, a series of experiments were conducted, in which various hyperparameters of the network were tested, and an adaptive learning rate was introduced.

If you have access to the multispectral datasets corresponding to the 8 Galician rivers used in the experiments, they can be reproduced by running the bash `scripts` located in the scripts directory:

- Experiment 1: 1_mnist.sh.
- Experiment 2-1: 2_1_learning_rate.sh.
- Experiment 2-2: 2_2_network_capacity.sh.
- Experiment 2-3: 2_3_ada_learning_rate.sh.
- Experiment 3: 3_all_datasets.sh.

### Future Directions :telescope:

- [ ] Conduct a deeper study on the influence of dataset imbalance on the performance of the conditioned StyleGAN2 in classification and image generation problems conditioned on a class label.
- [ ] Explore different modifications to the conditioned StyleGAN2 architecture to improve its performance in classification problems, such as those presented by ResBaGAN.

## Requirements :page_with_curl:

An `environment.yml` file is provided to create a Conda environment with the necessary dependencies. To create the environment, run:

```bash
conda env create -f environment.yml
```

## Datasets :file_folder:

The multispectral datasets used are not yet publicly available. However, the Computational Intelligence Group at the University of the Basque Country provides a series of multispectral datasets for download, which can be accessed through this [OneDrive folder](https://nubeusc-my.sharepoint.com/personal/anton_gomez_lopez_rai_usc_es/_layouts/15/onedrive.aspx?sw=bypass&bypassReason=abandoned&id=%2Fpersonal%2Fanton%5Fgomez%5Flopez%5Frai%5Fusc%5Fes%2FDocuments%2FTFG%2FStyleGAN2%2Dcondicionada%2Dclasificacion%2Fdata%2FPAVIA&ga=1) or from the [group's page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

In particular, the network is prepared to process one of these multispectral images: _Pavia University_. To use another dataset, it is necessary to modify the `ctyleGAN2/dataset.py` file to be able to read the image, the segments, the centers, and the class map (ground truth).

The Pavia University image files must be placed in the data/PAVIA directory and named as follows:

- Image: `pavia_university.raw`.
- Ground Truth: `pavia_university_gt.pgm`.
- Segment Map: `pavia_university_seg.raw`.
- Segment Centers: `pavia_university_seg_centers.raw`.

Additionally, the network also works with datasets that contain RGB or black-and-white images. The `datasets` directory includes the MNIST dataset.

## Usage :wrench:

### MNIST

Within the scripts directory, there are several bash scripts. The `1_mnist.sh` script allows training the model with the MNIST dataset:

```bash
bash scripts/1_mnist.sh
```

This script allows modifying the number of train steps, as well as the learning rate of the discriminator and the generator. The save every and evaluate every parameters allow setting how many batches are processed before saving the model weights and generating a series of images per class, respectively.

### PAVIA

Within the scripts directory, a script is also provided to perform training with the Pavia University multispectral image (`scripts/pavia_example.sh`). It is necessary to download the dataset files from [here](https://nubeusc-my.sharepoint.com/personal/anton_gomez_lopez_rai_usc_es/_layouts/15/onedrive.aspx?view=0&id=%2Fpersonal%2Fanton%5Fgomez%5Flopez%5Frai%5Fusc%5Fes%2FDocuments%2FTFG%2FStyleGAN2%2Dcondicionada%2Dclasificacion%2Fdata%2FPAVIA).

To evaluate pixel-level accuracy, the `cstylegan2/test_D.py` Python script can be used.

## License :memo:

This repository is under the GNU General Public License v3.0. See the `LICENSE` file for more details.
