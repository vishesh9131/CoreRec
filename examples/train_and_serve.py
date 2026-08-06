"""Train a model and serve it over HTTP — the whole path, end to end.

Run it:

    python examples/train_and_serve.py          # train, self-check, exit
    python examples/train_and_serve.py --serve  # train, then serve on :8000

Then:

    curl -X POST localhost:8000/recommend \
         -H 'Content-Type: application/json' \
         -d '{"user_id": 1, "top_k": 5}'

This uses generated interactions so it runs anywhere with no download. To use
a real dataset instead, replace build_interactions() with your own loader --
anything that returns three parallel lists. For MovieLens 1M:

    from cr_learn import ml_1m                      # pip install corerec[datasets]
    r = ml_1m.load()["ratings"]
    users, items, ratings = r["user_id"], r["movie_id"], r["rating"]

tests/test_train_and_serve.py runs this file, so the example cannot rot.
"""

import argparse

import numpy as np

from corerec.engines.two_tower import TwoTower
from corerec.serving import ModelServer

N_USERS, N_ITEMS = 60, 120


def build_interactions(seed: int = 0):
    """Generate (users, items, ratings) with latent taste, so recall is learnable.

    Each user and item gets a hidden group; matching groups rate highly. A model
    that learns nothing scores near random, which is what makes the assertion in
    main() meaningful rather than decorative.
    """
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, 4, N_USERS)
    item_group = rng.integers(0, 4, N_ITEMS)

    users, items, ratings = [], [], []
    for u in range(N_USERS):
        # Mostly in-group items, plus a few outside so the data isn't separable.
        in_group = np.flatnonzero(item_group == user_group[u])
        picks = rng.choice(in_group, size=min(12, len(in_group)), replace=False)
        for i in picks:
            users.append(u)
            items.append(int(i))
            ratings.append(float(rng.integers(4, 6)))
        for i in rng.choice(N_ITEMS, size=3, replace=False):
            users.append(u)
            items.append(int(i))
            ratings.append(float(rng.integers(1, 3)))
    return users, items, ratings, user_group, item_group


def main(serve: bool = False):
    users, items, ratings, user_group, item_group = build_interactions()

    model = TwoTower(embedding_dim=32, num_epochs=15, verbose=False)
    model.fit(users, items, ratings)

    server = ModelServer(model)

    # Query the real HTTP app rather than calling the model directly: this is
    # the path a user's traffic takes, so it is the path worth checking.
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    assert client.get("/health").status_code == 200

    resp = client.post("/recommend", json={"user_id": 0, "top_k": 5})
    resp.raise_for_status()
    recs = resp.json()["recommendations"]
    assert len(recs) == 5, f"expected 5 recommendations, got {len(recs)}"

    # Did it learn the group structure? Compare against picking at random.
    hits = sum(item_group[r] == user_group[0] for r in recs)
    baseline = float((item_group == user_group[0]).mean())
    print(f"user 0 -> {recs}")
    print(f"  in-group hits: {hits}/5   random baseline: {baseline * 5:.1f}/5")
    assert hits >= 2, f"model learned nothing: {hits}/5 in-group (baseline {baseline * 5:.1f})"

    if serve:
        print("serving on http://0.0.0.0:8000  (Ctrl-C to stop)")
        server.start()
    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="keep serving after training")
    main(serve=parser.parse_args().serve)
