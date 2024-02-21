import datasets
import requests
import json
from pathlib import Path
import tqdm


URL = "https://api.infini-gram.io/"


def check_overlap(name):
    data = datasets.load_dataset(
        name,
        split="test",
        trust_remote_code=True,
    )

    counts = []
    bad_idxs = []
    bad_docs = []
    repeat_queries = []
    repeat_counts = []
    for i, ex in enumerate(tqdm.tqdm(data)):
        text = ex["text"]
        lines = text.replace("\n\n", "\n").replace("\n\n", "\n").split("\n")
        # find first line with > 10 words
        for line in lines:
            if len(line.split()) > 10:
                query = " ".join(line.split()[1:-1])
                break

        payload = {
            #"corpus": "v4_piletrain_llama",
            "corpus": "v4_c4train_llama",
            "query_type": "count",
            "query": query,
        }

        # Headers to specify that we are sending JSON data
        headers = {"Content-Type": "application/json"}

        # Sending the POST request
        response = requests.post(URL, json=payload, headers=headers)

        #print(response, response.json())

        count = response.json()["count"]
        counts.append(count)
        if count > 0:
            bad_idxs.append(i)
            bad_docs.append(ex)
            repeat_queries.append(query)
            repeat_counts.append(count)

    output = {
        "bad_idxs": bad_idxs,
        #"bad_docs": bad_docs,
        "repeat_queries": repeat_queries,
        "repeat_counts": repeat_counts,
    }

    path = Path(f"{name}-duplicates.json")
    with path.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    check_overlap("pg19")
    check_overlap("lm1b")
