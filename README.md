# StyGAN2 condicionada en Pytorch para clasificación de imaxes multiespectrais

Este repositorio é unha versión do repositorio de [CIA-Oceanix](https://github.com/CIA-Oceanix/Conditional_StyleGAN2_pytorch), que así mesmo, é unha versión modificada do repositorio de [lucidrains](https://github.com/lucidrains/stylegan2-pytorch)

O artigo no que se presenta a arquitectura da [StyleGAN2](https://github.com/NVlabs/stylegan2) pódese atopar [aquí](https://arxiv.org/abs/1912.04958).

# Requisitos

Proporciónase un arquivo `environment.yml` para crear un entorno conda coas dependencias necesarias. Para crear o entorno, executar:

```bash
conda env create -f environment.yml
```

# Conxuntos de datos

Os conxuntos de datos multiespectrais empregados non son de dominio público. Porén, o Grupo de Intelixencial Computacional da Universidade do País Vasco pona disposición do usuario unha serie de conxuntos de datos multiespectrais para a súa descarga que se poden descargar dende [aquí](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

En particular, a rede está preparada para procesar unha destas imaxes multiespectrais: _Pavia University_. Para empregar outro conxunto de datos, é necesario modificar o arquivo `ctyleGAN2/dataset.py` para poder ler a imaxe, os segmentos, os centros e o mapa de clases (_ground truth_).

Os arquivos da imaxe _Pavia University_ deben ser colocados dentro do directorio `data/PAVIA` e deben ser nomeados do seguinte xeito:

- Imaxe: `pavia_university.raw`.
- Ground Truth: `pavia_university_gt.pgm`.
- Mapa de segemntos: `pavia_university_seg.raw`.
- Centros dos segementos: `pavia_university_seg_centers.raw`.

Ademais, a rede tamén funciona con conxuntos de datos con imaxes RGB ou en branco e negro. No directorio `datasets` inclúese o conxutno de datos MNIST.

# Funcionamento

## MNIST

Dentro do directorio `scripts` hai unha serie de scripts de bash. O script `1_mnist` permite adestrar o modelo co conxunto de datos MNIST:

```bash
bash scripts/1_mnist.sh
```

Este script permite modificar o número de _train steps_, así como o _learning rate_ do discriminador e do xerador. Os parámetros _save every_ e _evaluate every_ permiten establecer cada cantos _batches_ se almacenan os peso do modelo e se xeran unha serie de imaxes por clase co modelo, respectivamente.

## PAVIA

Dentro do directorio `scripts` tamén se proporciona un script para realizar un adestramento coa imaxe multiespectral _Pavia University_. Para iso, é necesario descargala dende [aquí](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) e nomear os arquivos como se indicou.

Para avaliar a precisión a nivel de píxel pode empregarse o script de Python `cstylegan2/test_D.py`.

# Réplica dos experimentos

No caso de dispoñer dos conxuntos de datos multiespectrais correspondentes aos 8 ríos galegos cos que se desenvolveron os experimentos, estes pódense reporducir executando os scripts de bash dentro do directorio `scripts`:

- Experimento 1: `1_mnist.sh`.
- Experimento 2-1: `2_1_learning_rate.sh`.
- Experimento 2-2: `2_2_network_capacity.sh`.
- Experimento 2-3: `2_3_ada_learning_rate.sh`.
- Experimento 3: `3_all_datasets.sh`.

# Licenza

Este repositorio está baixo a licenza Nvidia Source Code License-NC. Ver o arquivo `LICENSE` para máis información.
