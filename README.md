# GATAS
Implementation of *Graph Representation Learning Network via Adaptive Sampling*: [http://arxiv.org/abs/2006.04637](http://arxiv.org/abs/2006.04637)

The algorithm represents nodes by reducing their neighbour representations with attention. Multi-step neighbour representations incorporate different path properties. Neighbours are sampled using learnable depth coefficients.


## Overview
This repository is organized as follows:
- `data/` contains the necessary files for the Cora, Pubmed, Citeseer, PPI, Twitter and YouTube datasets.
- `framework/` contains helper libraries for model development, training and evaluation.
- `gatas/` contains the implementation of GATAS.
- `node_classification/` contains a node label classifier using the model.
- `link_prediction/` contains a link predictor using the model.


## Instructions
First we must create a CSR binary representation of the graph where the values are the edge type indices. For the Cora, Citeseer and PubMed datasets, we have precomputed and placed them in `data/`. For PPI, it can be computed with:

```bash
python3 -m node_classification.datasets.ppi --path data/ppi
```

For the Twitter and YouTube datasets, it can be computed with:
```bash
python3 -m link_prediction.datasets.gatne --path {path to dataset} --num-edge-types {number of edge types}
```

These scripts will also collect and preprocess the node features when available, and create dataset splits with inputs and targets for the tasks. Once we have a CSR graph representation, we can compute the transition probabilities by running:
```bash
python3 -m gatas.transitions --path {path to dataset} --num_steps {number of steps}
```

GATAS has two components: `NeighbourSampler` and `NeighbourAggregator`. `NeighbourSampler` can be initialized with a path so the precomputed transition data can be used:

```python
from gatas.sampler import NeighbourSampler

neighbour_sampler = NeighbourSampler.from_path(num_steps=3, path='data/ppi')
```

`NeighbourAggregator` can receive a matrix of node features and can be initialized as follows:
```python
import numpy as np
from gatas.aggregator import NeighbourAggregator

node_features = np.load('data/ppi/node_embeddings.npy')

neighbour_aggregator = NeighbourAggregator(
    input_noise_rate=0.,
    dropout_rate=0.,
    num_nodes=node_features.shape[0],
    num_edge_types=neighbour_sampler.num_edge_types,
    num_steps=3,
    edge_type_embedding_size=5,
    node_embedding_size=None,
    layer_size=256,
    num_attention_heads=10,
    node_features=node_features,
)
```

We can call `neighbour_aggregator` with the output of `neighbour_sampler`. This pattern is used in the node classification and link prediction tasks. You can train those models with:

```bash
python3 -m node_classification.train --data-path {path to dataset}
```

or:

```bash
python3 -m link_prediction.train --data-path {path to dataset}
```

where additional parameters can be passed through the command line. Run with `--help` for a list of them:

```bash
python3 -m node_classification.train --help
```


## Reference
```tex
@misc{andrade2020graph,
    title={Graph Representation Learning Network via Adaptive Sampling},
    author={Anderson de Andrade and Chen Liu},
    year={2020},
    eprint={2006.04637},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## License
MIT
