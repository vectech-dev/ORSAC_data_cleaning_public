# ransac-label-verification

Algorithm inspired by the ransac algorithm that iteratively cleans datasets for classification problems. 
This repository is created for the paper "Ransac for Deep Learning Classification" (placeholder)

## RUN

Uses [`pydantic`](https://docs.pydantic.dev/) to configure/orchestrate settings for training. Install requirements from [`requirements.txt`](./requirements.txt). And `pip` install package:


```sh
$ pip install -e .
```

## Usage
### Train

```sh
$ python -m ransac_label_verification --config configs/config.json
```

### Test
For multiple experiments, run: 
```sh
python -m ransac_label_verification --mode test --exp-dir experiments/example_exp
```

### Evaluation

```sh
python -m ransac_label_verification --mode eval --exp-dir experiments/example_exp

```

### Settings 
For this repository we used `cuda =11.7`. 
Create `.env` file in `ransac_label_verification`
```sh
IMAGE_BASE_DIRECTORY="/example/example"
```
Format the metadata for your dataset with the following columns:
```sh
Id:" the unique Id for each image in the dataset.  IMAGE_BASE_DIRECTORY/Id gives the full path for each image. "
y: "the true label of each image"
View: "For multiple views of the same object, specify the type of view here. If not applicable set View to None for all images."
| Id | y | View |
|----|---|------|
| Row 1, Column 1 | Row 1, Column 2 | Row 1, Column 3 |
| Row 2, Column 1 | Row 2, Column 2 | Row 2, Column 3 |

Column 1: This column represents ...
Column 2: This column represents ...
Column 3: This column represents ...

```
### Dataset
We used the John Hopkins University mosquito dataset.
We also used the CIFAR100 dataset 

### Script

```sh
usage: ransac_label_verification [-h] [--mode {eval,test,train}] [--config CONFIG] [--exp-dir EXP_DIR] [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --mode {test,train}   Overwrite mode from the configuration file
  --config CONFIG       The configuration file for training
  --exp-dir EXP_DIR     The experiment directory for tests
  --batch-size BATCH_SIZE
                        Overwrite the batch size from configuration
```
