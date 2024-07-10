# Hybrid X-Linker: Automated Data Generation and Extreme Multi-label Ranking for Biomedical Entity Linking


This repository includes the necessary code to reproduce all the experiments described [here](https://arxiv.org/abs/2407.06292).

Citation:

```
@misc{ruas2024,
      title={Hybrid X-Linker: Automated Data Generation and Extreme Multi-label Ranking for Biomedical Entity Linking}, 
      author={Pedro Ruas and Fernando Gallego and Francisco J. Veredas and Francisco M. Couto},
      year={2024},
      eprint={2407.06292},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.06292}, 
}
```



## Environment setup
The experiments were done in an environment based on Python 3.9.1.

To install the requirements:

```
./setup.sh
```

Run in the root directory of the project:

```
export PYTHONPATH="${PYTHONPATH}:"
```

## Data preparation

Data is stored in https://zenodo.org/records/12704543.

Get knowledge base files, datasets/corpora (and PubTator annotations if building training data from scratch):

```
./src/bash/get_data.sh
```

If you want to generate all the training data from scratch and train PECOS-EL models follow the instructions in the file 'build_train.md'. To just perform evaluation or use X-Linker for inference follow the next steps in this file.


## Evaluation
 
### X-Linker

To evaluate X-Linker in datasets:

```
python src/python/xlinker/evaluate.py -dataset <> -ent_type <> -kb <> -model_dir <> -top_k <> -clustering <> --abbrv --pipeline --ppr --fuzzy_top_k
```

Arguments:
* -dataset: evaluation dataset, 'bc5cdr', 'biored', 'nlm_chem' or 'ncbi:disease'; type=str, required=True
* -ent_type: entity type, 'Disease' or 'Chemical' type=str, required=True
* -kb: target knowledge organization system, 'medic' or 'ctd_chemicals; type=str, required=True
* -model_dir: type=str, required=True
* -top_k: top-k candidates to output; type=int, default=5
* -clustering: type=str, default="pifa"
* --abbrv: use abbreviation detector or not; default=False, action=BooleanOptionalAction
* --pipeline: use the X-Linker pipeline or not default=False, action=BooleanOptionalAction
* --threshold: threshold to filter candidates; type=float, default=0.1
* --ppr: use the Personalized PageRank or not default=False, action=BooleanOptionalAction
* --fuzzy_top_k: number of candidates to be retrieved by the string matcher; type=int, default=1


Example:

```
python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir data/models/trained/disease_200_1ep -top_2 --abbrv --pipeline --threshold 0.1 --ppr
```

### SapBERT:

```
python src/python/sapbert.py -ent_type <ent_type> -dataset <disease> -kb <kb> -top_k <top_k> --abbrv
```

Arguments:
- ent_type: entity type, 'Disease' or 'Chemical' type=str, required=True
- dataset: evaluation dataset, 'bc5cdr', 'biored', 'nlm_chem' or 'ncbi:disease'; type=str, required=True
- kb: target knowledge organization system, 'medic' or 'ctd_chemicals; type=str, required=True
- top_k: top-k candidates to output; type=int, default=5
-- abbrv: use abbreviation detector or not; default=False, action=BooleanOptionalAction


Example:

```
python3.9 src/python/sapbert.py -dataset bc5cdr -ent_type Disease -kb medic -top_k 1
```
