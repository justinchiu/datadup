
echo "# Load pg19 test"
pdm run python scripts/load_dataset.py --name pg19 --split test
echo "# Load c4 train"
pdm run python scripts/load_dataset.py --name c4 --split train


RUST_BACKTRACE=full
echo "# Make pg19 suffix array"
pdm run python scripts/make_suffix_array.py output/pg19.test
echo "# Make c4 suffix array"
pdm run python scripts/make_suffix_array.py output/c4.train
