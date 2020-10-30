# An Open-World Extension for Knowledge Graph Completion Models
This repository contains the official Pytorch code for [An Open-World Extension for Knowledge Graph Completion Models](https://aaai.org/ojs/index.php/AAAI/article/view/4162).

## Setup

Resolve dependencies by executing the following command:
```bash
pip install -e .
```

Then download the
[FB15k-237-OWE](https://github.com/haseebs/haseebs.github.io/raw/master/assets/FB15k-237-OWE.zip), 
[DBPedia50](https://github.com/haseebs/haseebs.github.io/raw/master/assets/dbpedia50.zip) and
[FB20k](https://github.com/haseebs/haseebs.github.io/raw/master/assets/FB20K_preprocessed.zip) datasets as
required. These contain the datasets in format required for both OpenKE
and our code.

## Usage
### Training the KGC model

Before using OWE you need to train a KGC model. You can use any knowledge graph embedding framework for training. 
We used the PyTorch version of [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch).  

To use a previously trained KGC model with OWE you need to export the entity and relation matrices as 
*numpy arrays* using pickle:
 - For TransE and DistMult the matrices are called `entities.p` and `relations.p`
 - For CompLEx `entities_r.p`, `entities_i.p`, `relations_r.p` and `relations_i.p`.  

Furthermore you need to provide two more files: `entity2id.txt` and `relation2id.txt`, which contain a mapping
of an entity/relation string to the corresponding index (id) in the embedding files.

The trained 300D ComplEx embeddings for FB15k-237-OWE can be obtained from [here](https://github.com/haseebs/haseebs.github.io/raw/master/assets/FB15k-237-OWE-closed-world-embeddings.zip).

For closed-world evaluation:
```bash
python owe/run_closed_world.py -e -c <PATH_TO_DATASET> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```
where `KGC_MODEL_NAME` can be `complex`, `distmult` or `transe`.

#### Instructions for OpenKE

Follow the instructions to install OpenKE. Under [openke_scripts](openke_scripts/) you can find exemplary training scripts (make sure you update the paths to the dataset in OpenKE format). As the API of OpenKE frequently changes, scripts work with the 
[following version of OpenKE](https://github.com/thunlp/OpenKE/tree/0a55399b3e800bc779582c4784cac96f00230fd8).

### Training the Open-World Extension

Word embeddings are required for training the extension. You may obtain the wikipedia2vec word embeddings from [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/). 
An example config file to reproduce the results is provided as config.ini in the root folder.

For training:
```bash
python owe/run_open_world.py -t -c <PATH_TO_DATASET> -d  <PATH_TO_CONFIG_AND_OUTPUT_DIR> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```

For evaluation:
```bash
python owe/run_open_world.py -e -lb -c <PATH_TO_DATASET> -d  <PATH_TO_CONFIG_AND_OUTPUT_DIR> --<KGC_MODEL_NAME> <PATH_TO_KGC_EMBEDDINGS>
```

## Citation
If you found our work helpful in your research, consider citing the following:
```
@inproceedings{shah2019open,
  title={An Open-World Extension to Knowledge Graph Completion Models},
  author={Shah, Haseeb and Villmow, Johannes and Ulges, Adrian and Schwanecke, Ulrich and Shafait, Faisal},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3044--3051},
  year={2019}
}
```
