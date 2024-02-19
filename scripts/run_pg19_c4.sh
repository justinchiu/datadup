
pdm run python scripts/load_dataset.py --name pg19 --split test
pdm run python scripts/load_dataset.py --name c4 --split train --stream
RUST_BACKTRACE=full
pdm run python scripts/make_suffix_array.py output/pg19.test
pdm run python scripts/make_suffix_array.py output/c4.train
