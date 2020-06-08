import os
import sys
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def create_triplets(heads: np.ndarray, tails: np.ndarray) -> Generator[Tuple[int, int, int], None, None]:
    for head, tail in zip(heads, tails):
        if head != tail:
            yield head, tail, 0
            yield tail, head, 1

        else:
            yield tail, head, 2


def normalize_features(features: sp.spmatrix) -> sp.spmatrix:
    row_sum = np.array(features.sum(1))

    row_inverse = np.power(row_sum, -1).flatten()
    row_inverse[np.isinf(row_inverse)] = 0

    normalized_features = sp.diags(row_inverse).dot(features)

    return normalized_features


def main(path: str):
    cites = pd.read_csv(os.path.join(path, 'cites.csv'), header=None, names=('cited', 'citing'))
    content = pd.read_csv(os.path.join(path, 'content.csv'), header=None, names=('id', 'word'))
    paper = pd.read_csv(os.path.join(path, 'paper.csv'), header=None, names=('id', 'label'))

    paper_ids = set(cites['cited'].values) | set(cites['citing'].values) | set(content['id'].values) | set(paper['id'].values)
    paper_id_to_index = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    label_to_index = {value: index for index, value in enumerate(set(paper['label']))}

    cites['cited'] = cites['cited'].map(lambda paper_id: paper_id_to_index[paper_id])
    cites['citing'] = cites['citing'].map(lambda paper_id: paper_id_to_index[paper_id])
    content['id'] = content['id'].map(lambda paper_id: paper_id_to_index[paper_id])
    paper['id'] = paper['id'].map(lambda paper_id: paper_id_to_index[paper_id])

    heads, tails, edge_types = zip(*create_triplets(cites['cited'].values, cites['citing'].values))

    adjacency_matrix = sp.coo_matrix((edge_types, (heads, tails)), dtype=np.int32).tocsr()
    adjacency_matrix.indptr.tofile(os.path.join(path, 'accumulated_num_edges.bin'))
    adjacency_matrix.indices.tofile(os.path.join(path, 'adjacencies.bin'))
    adjacency_matrix.data.tofile(os.path.join(path, 'edge_types.bin'))

    content['word'] = content['word'].map(lambda string: int(string.replace('word', '')) - 1)
    features = sp.coo_matrix((np.ones((len(content),), np.float32), (content['id'], content['word']))).todense()
    np.save(os.path.join(path, 'node_embeddings.npy'), features)

    paper['label'] = paper['label'].map(lambda string: label_to_index[string])
    targets = paper.sort_values('id')['label'].values.astype(np.int32)
    np.save(os.path.join(path, 'class_ids.npy'), targets)


if __name__ == '__main__':
    main(*sys.argv[:1])
