import datasets
from transformers import AutoTokenizer
import streaming
from multiprocessing import Pool

from datadup.data import StreamingTextDataset

tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

pg19 = datasets.load_dataset("pg19")
c4 = streaming.text.c4.StreamingC4(
    tokenizer_name=tokenizer_name,
    remote="/share/rush/c4",
    local="/scratch/jtc257/c4",
    split="train",
    max_seq_len=-1,
    group_method="truncate",
)
# get the text without tokenizing it
c4.super_get_item = super(type(c4), c4).get_item


test = pg19["test"]
train = c4


# check if train contains anything in test

with Pool(64) as p:
    p.map()
import pdb

pdb.set_trace()
