# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import struct
import numpy as np
import datasets
from transformers import AutoTokenizer
import multiprocessing as mp
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description="Load a dataset.")
parser.add_argument("--save_dir", type=str, default="output")
parser.add_argument("--name", type=str, default="pg19")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--stream", action="store_true")
parser.add_argument("--pre_sep", type=bytes, default=b"\xff\xff")
parser.add_argument("--post_sep", type=bytes, default=b"")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



split = args.split
save_dir = args.save_dir
dataset_name = args.name


if dataset_name == "c4":
    ds = datasets.load_dataset(
        dataset_name, "en",
        split=split,
        trust_remote_code=True,
        streaming=True,
    )
else:
    ds = datasets.load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True,
        streaming=args.stream,
    )
# assert isinstance(ds, tf.data.Dataset)
print(ds)
def tokenization(example):
    output = tokenizer(example["text"])
    out = [
        np.array(x, dtype=np.uint16).view(np.uint8).tobytes()
        for x in output["input_ids"]
    ]
    return {"idbytes": out}

dataset = ds.map(tokenization, batched=True)
if dataset_name == "pg19":
    dataset = dataset.remove_columns([
        "short_book_title",
        "publication_date",
        "url",
        "text",
    ])

pre_sep = args.pre_sep
post_sep = args.post_sep

UID = 0


def sep():
    global UID
    UID += 1
    return pre_sep + struct.pack("<I", UID) + post_sep


Path(save_dir).mkdir(parents=True, exist_ok=True)

fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")

with mp.get_context("fork").Pool(mp.cpu_count()) as p:
    i = 0
    sizes = [0]
    for example in dataset:
        print(i)
        x = example["idbytes"]
        next_line = sep() + x
        fout.write(next_line)
        sizes.append(sizes[-1] + len(next_line))
        i += 1

open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
    np.array(sizes, dtype=np.uint64).tobytes()
)
