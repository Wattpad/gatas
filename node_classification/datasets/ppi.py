import json
import os
from typing import Iterable, Tuple, List, Dict

import defopt
import numpy as np
import scipy.sparse as sp


def get_triplets(links: List[Dict[str, int]]) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    heads, tails, edge_types = zip(*(
        (head, tail, edge_type)
        for link in links
        for head, tail, edge_type in ((link['source'], link['target'], 0), (link['target'], link['source'], 0))
    ))

    return heads, tails, edge_types


def main(*, path: str):
    graph = json.load(open(os.path.join(path, 'ppi-G.json')))

    num_nodes = len(graph['nodes'])

    train_indices, validation_indices, test_indices = [], [], []

    for node in graph['nodes']:
        index = node['id']

        if node['val']:
            validation_indices.append(index)

        elif node['test']:
            test_indices.append(index)

        else:
            train_indices.append(index)

    np.save(os.path.join(path, 'train_indices.npy'), np.array(train_indices, dtype=np.int32))
    np.save(os.path.join(path, 'validation_indices.npy'), np.array(validation_indices, dtype=np.int32))
    np.save(os.path.join(path, 'test_indices.npy'), np.array(test_indices, dtype=np.int32))

    heads, tails, edge_types = get_triplets(graph['links'])

    adjacency_matrix = sp.coo_matrix((edge_types, (heads, tails)), dtype=np.int32, shape=(num_nodes, num_nodes)).tocsr()

    adjacency_matrix.indptr.tofile(os.path.join(path, 'accumulated_num_edges.bin'))
    adjacency_matrix.indices.tofile(os.path.join(path, 'adjacencies.bin'))
    adjacency_matrix.data.tofile(os.path.join(path, 'edge_types.bin'))

    features = np.load(os.path.join(path, 'ppi-feats.npy')).astype(np.float32)

    np.save(os.path.join(path, 'node_embeddings.npy'), features)

    class_map = json.load(open(os.path.join(path, 'ppi-class_map.json')))
    num_labels = len(next(iter(class_map.values())))

    targets = np.zeros((num_nodes, num_labels), dtype=np.int32)

    for key, labels in sorted([(int(key), value) for key, value in class_map.items()], key=lambda values: values[0]):
        targets[key, :] = labels

    np.save(os.path.join(path, 'class_ids.npy'), targets)


if __name__ == '__main__':
    defopt.run(main)
