# StyGAN2 condicionada en Pytorch para clasificación de imaxes multiespectrais

Este repositorio é unha versión do repositorio de [CIA-Oceanix](https://github.com/CIA-Oceanix/Conditional_StyleGAN2_pytorch), que así mesmo, é unha versión modificada do repositorio de [lucidrains](https://github.com/lucidrains/stylegan2-pytorch). O artigo no que se presenta a arquitectura da [StyleGAN2](https://github.com/NVlabs/stylegan2) pódese atopar [aquí](https://arxiv.org/abs/1912.04958).

# Descripción do uso da rede :ledger:

Utilizouse esta rede para clasificar imaxes multiespectrais de ríos galegos obtidas mediante teledetección. As imaxes foron tomadas co obxectivo de monitorizar automaticamente as cuncas dos ríos de Galicia. Especificamente,
preténdese determinar o estado da vexetación, identificar áreas ocupadas por especies invasoras e detectar estruturas artificiais que ocupan a cunca do río empregando imaxes multiespectrais.

En particular, trátase de 8 imaxes de alta resolución multiespectral que foron utilizadas [neste artigo](https://www.mdpi.com/2072-4292/13/14/2687) para avaliar diversos algoritmos de segmentación e de clasificación. As imaxes están formadas por cinco bandas correspondentes ás lonxitudes de onda de 475 nm (azul), 560 nm (verde), 668 nm (vermello), 717 nm (red-edge) e 840 nm (infravermello próximo).

Estes conxuntos de datos presentan dous principais problemas:

 - Grande desbalanceo de mostras entre clases.
 - Escaseza de mostras dalgunhas clases.

Deste xeito, utilizouse esta GAN co obxectivo de intentar facer fronte a estes problemas aumentando o conxunto de datos mediante o xerador e utilizando o discrimiandor como clasificador. Os resultados obtidos son peores do esperado e son similares aos obtidos mediante unha rede convolucional estándar.

## Liñas futuras :telescope:

 - [ ] Facer un estudo máis profundo sobre a influencia do desbalanceamento dos conxuntos de datos no rendemento da StyleGAN2 condicionada en problemas de clasificación e xeración de imaxes condicionadas a unha etiqueta de clase.
 - [ ] Explorar diferentes modificacións sobre a arquitectura da StyleGAN2 condicionada para mellorar o seu rendemento en problemas de clasificación, como as que presenta a [ResBaGAN](https://github.com/alvrogd/ResBaGAN).

# Requisitos :page_with_curl:

Proporciónase un arquivo `environment.yml` para crear un entorno conda coas dependencias necesarias. Para crear o entorno, executar:

```bash
conda env create -f environment.yml
```

# Conxuntos de datos :file_folder:

Os conxuntos de datos multiespectrais empregados non son de dominio público. Porén, o Grupo de Intelixencial Computacional da Universidade do País Vasco pona disposición do usuario unha serie de conxuntos de datos multiespectrais para a súa descarga que se poden descargar dende esta [carpeta de OneDrive](https://nubeusc-my.sharepoint.com/personal/anton_gomez_lopez_rai_usc_es/_layouts/15/onedrive.aspx?sw=bypass&bypassReason=abandoned&id=%2Fpersonal%2Fanton%5Fgomez%5Flopez%5Frai%5Fusc%5Fes%2FDocuments%2FTFG%2FStyleGAN2%2Dcondicionada%2Dclasificacion%2Fdata%2FPAVIA&ga=1) ou dende a [páxina do grupo](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

En particular, a rede está preparada para procesar unha destas imaxes multiespectrais: _Pavia University_. Para empregar outro conxunto de datos, é necesario modificar o arquivo `ctyleGAN2/dataset.py` para poder ler a imaxe, os segmentos, os centros e o mapa de clases (_ground truth_).

Os arquivos da imaxe _Pavia University_ deben ser colocados dentro do directorio `data/PAVIA` e deben ser nomeados do seguinte xeito:

- Imaxe: `pavia_university.raw`.
- Ground Truth: `pavia_university_gt.pgm`.
- Mapa de segemntos: `pavia_university_seg.raw`.
- Centros dos segementos: `pavia_university_seg_centers.raw`.

Ademais, a rede tamén funciona con conxuntos de datos con imaxes RGB ou en branco e negro. No directorio `datasets` inclúese o conxutno de datos MNIST.

# Uso :wrench:

## MNIST

Dentro do directorio `scripts` hai unha serie de scripts de bash. O script `1_mnist` permite adestrar o modelo co conxunto de datos MNIST:

```bash
bash scripts/1_mnist.sh
```

Este script permite modificar o número de _train steps_, así como o _learning rate_ do discriminador e do xerador. Os parámetros _save every_ e _evaluate every_ permiten establecer cada cantos _batches_ se almacenan os peso do modelo e se xeran unha serie de imaxes por clase co modelo, respectivamente.

## PAVIA

Dentro do directorio `scripts` tamén se proporciona un script para realizar un adestramento coa imaxe multiespectral _Pavia University_. Para iso, é necesario descargala dende [aquí](https://nubeusc-my.sharepoint.com/personal/anton_gomez_lopez_rai_usc_es/_layouts/15/onedrive.aspx?view=0&id=%2Fpersonal%2Fanton%5Fgomez%5Flopez%5Frai%5Fusc%5Fes%2FDocuments%2FTFG%2FStyleGAN2%2Dcondicionada%2Dclasificacion%2Fdata%2FPAVIA).

Para avaliar a precisión a nivel de píxel pode empregarse o script de Python `cstylegan2/test_D.py`.

# Réplica dos experimentos :bar_chart:

No caso de dispoñer dos conxuntos de datos multiespectrais correspondentes aos 8 ríos galegos cos que se desenvolveron os experimentos, estes pódense reporducir executando os scripts de bash dentro do directorio `scripts`:

- Experimento 1: `1_mnist.sh`.
- Experimento 2-1: `2_1_learning_rate.sh`.
- Experimento 2-2: `2_2_network_capacity.sh`.
- Experimento 2-3: `2_3_ada_learning_rate.sh`.
- Experimento 3: `3_all_datasets.sh`.

# Licenza :memo:

Este repositorio está baixo a licenza Nvidia Source Code License-NC. Ver o arquivo `LICENSE` para máis información.
