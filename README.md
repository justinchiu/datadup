# datadup :skull_and_crossbones:

Check if the test set of your data is duplicated in your pretraining data :skull_and_crossbones:
This should be a bit easier than checking if your training dataset has duplications if your
test set is small -- you only have to index the test set.

# Recommendation: Just use infinigram API.
See `scripts/query_infinigram.py`.

[Suffix array implementation by N Carlini](https://github.com/google-research/deduplicate-text-datasets/tree/master).
