import json
import multiprocessing as mp
import re
from collections import defaultdict
from typing import List, Optional, Set

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH, minhash
from dpu_utils.utils.iterators import ThreadedIterator
from tqdm import tqdm


class DuplicationIndex:
    def __init__(
        self,
        *,
        duplication_jaccard_threshold: float = 0.85,
        num_perm: int = 256,
        min_num_tokens: int = 10,
    ):
        self.__duplication_jaccard_threshold = duplication_jaccard_threshold
        self.__num_perm = num_perm
        self.__min_num_tokens = min_num_tokens
        self.__index = MinHashLSH(
            threshold=self.__duplication_jaccard_threshold, num_perm=self.__num_perm
        )

        self.__duplicate_clusters = defaultdict(set)

    def get_min_hash(self, tokens: List[str]) -> Optional[MinHash]:
        if len(tokens) < self.__min_num_tokens:
            return None
        min_hash = MinHash(num_perm=self.__num_perm)
        for token in set(tokens):
            min_hash.update(token.encode())
        return min_hash

    def add(self, filename: str, min_hash: MinHash) -> None:
        close_duplicates = self.__index.query(min_hash)
        if filename in self.__index.keys:
            print("Duplicate key %s" % filename)
            return

        self.__index.insert(filename, min_hash)
        if len(close_duplicates) > 0:
            # print("`%s` duplicate of: %s" % (filename, close_duplicates))

            for base_duplicate in close_duplicates:
                if base_duplicate in self.__duplicate_clusters:
                    self.__duplicate_clusters[base_duplicate].add(filename)
                    break
            else:
                self.__duplicate_clusters[close_duplicates[0]].add(filename)

    def save(self, filepath) -> None:
        duplicate_clusters = []
        for base, duplicates in self.__duplicate_clusters.items():
            duplicate_clusters.append(list(duplicates) + [base])

        with open(filepath, "w") as f:
            json.dump(duplicate_clusters, f)
