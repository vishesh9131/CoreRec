# The Production Loop

A served model goes stale. CoreRec closes the loop from serving back to
training: log what users saw and clicked, read online metrics, A/B test a new
model, retrain on fresh data and swap the model without a restart, and get
alerted when traffic drifts.

```text
corerec serve --feedback-log  ->  POST /feedback  ->  GET /metrics
        ^                                                 |
        |   POST /reload  <-  corerec retrain  <----------+
```

## 1. Log impressions and feedback

```bash
corerec train events.csv -o artifacts/m
corerec serve artifacts/m --feedback-log feedback.jsonl
```

Every `/recommend` response now carries a `request_id`, and the list shown is
logged as an impression. When the user acts on an item, send it back with that
id so the click is credited to the list (and A/B variant) that showed it:

```bash
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
     -d '{"user_id": "u0042", "top_k": 5}'
# {"recommendations": ["i444", "i097", ...], "variant": "default", "request_id": "77e6...", ...}

curl -X POST localhost:8000/feedback -H 'Content-Type: application/json' \
     -d '{"user_id": "u0042", "item_id": "i097", "request_id": "77e6...", "event": "click"}'
```

`event` is free text; `click` and `purchase` are used for retraining. The log
is one JSON object per line, easy to ship to a warehouse.

## 2. Online metrics

```bash
curl localhost:8000/metrics
```

Per variant and overall:

| Field | Meaning |
|---|---|
| `ctr` | clicks on shown items / items shown |
| `requests_with_click` | share of lists that got at least one click |
| `mrr` | mean of 1 / rank of the first clicked item per list (0 if none) |
| `distinct_items_shown` | catalogue coverage |
| `fallback_share` | share of requests answered with popular items (unknown users) |

These are what the offline NDCG from `corerec train` is a proxy for. The same
numbers are available offline with `FeedbackLog("feedback.jsonl").metrics()`.

## 3. A/B test a new model

```bash
corerec serve artifacts/m --challenger artifacts/m2 --challenger-share 0.1 \
              --feedback-log feedback.jsonl
```

Users are assigned by a stable hash of their id, so each one keeps seeing the
same variant. `/metrics` then includes `ab_test`: both CTRs, the lift and a
two-proportion z-test p-value. In Python:
`ModelServer({"control": m1, "treatment": m2}, traffic={"control": 0.9, "treatment": 0.1}, feedback_log="feedback.jsonl")`.

## 4. Retrain on a schedule

```bash
corerec retrain artifacts/m --feedback feedback.jsonl
```

Retrain re-reads the training file (append new events to it, or pass
`--data newer.csv`) and adds clicks and purchases from the feedback log. Rows
newer than the model's last training are new data, split in time:

- the candidate trains on everything old plus the earlier half of the new rows
- both the candidate and the current model are scored on the later half,
  which neither has seen, with the popular-item fallback the server uses

The candidate replaces the model only if its NDCG@10 is at least the current
one's (`--tolerance` relaxes that). It is then retrained on all rows; the old
artifact stays in `artifacts/m/previous/` for rollback. `--dry-run` compares
without changing anything. A random holdout can't do this job: the deployed
model was trained on those rows, and its exclude-seen filter removes exactly
the items being tested.

Pick up the new model without a restart:

```bash
curl -X POST localhost:8000/reload
```

Nightly with cron:

```text
0 3 * * *  corerec retrain /srv/artifacts/m --feedback /srv/feedback.jsonl && curl -s -X POST localhost:8000/reload
```

Retrain needs a timestamp column in the data, to tell new rows from old.

## 5. Drift alerts

`/metrics` also returns `drift`, comparing the last 1000 requests
(`?recent=N`) with everything before them. It alerts, and logs a warning,
when:

- CTR fell by more than 30%
- the fallback (unknown-user) share rose by more than 10 points
- the distribution of clicked items moved (total variation distance > 0.2)

Thresholds are arguments of `FeedbackLog.drift()`. A rising fallback share
usually means many new users the model hasn't seen: retrain.
