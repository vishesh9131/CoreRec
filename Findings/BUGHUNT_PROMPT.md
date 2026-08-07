# Bug-hunt loop prompt

Paste the block below as the `/loop` prompt in Claude Code (Termux or anywhere).

    /loop 45m <paste the prompt below>

Or run it without an interval and let it self-pace:

    /loop <paste the prompt below>

---

## The prompt

```
You are hunting integration bugs in CoreRec by USING it, not by reading it.

Every bug found in this repo so far came from running an end-to-end path, never
from code review. All three had the same shape: two layers written separately
that had never been executed together.

  - ModelServer's /recommend passed exclude_items to models that did not accept
    it, so every HTTP request against a two-tower model returned 500.
  - PointwiseRanker called model.predict(X), the sklearn convention, while every
    CoreRec model takes predict(user_id, item_id), so the pipeline could not
    consume the library's own models.
  - test_docs.py compiled documented code without importing it, so 74 of 117
    documented import paths were broken and no test noticed.

Your job is to find more of these.

## Each iteration

1. Read Findings/bug.md if it exists. Note every combination already tried, in
   the "Combinations exercised" table. Do not repeat one.

2. Pick ONE untried combination that crosses layers. The layers are:

     models     corerec.engines / .collaborative / .content_based /
                .matrix_factorization  (34 importable models)
     data       corerec.data  (11 dataset classes)
     pipeline   corerec.retrieval -> corerec.ranking -> corerec.reranking
     serving    corerec.serving  (ModelServer, OnlineRecommender,
                BatchInferenceEngine, ModelLoader)
     persist    model.save() / .load(), then use the reloaded object
     eval       corerec.evaluation, corerec.metrics

   Prefer pairs that look like nobody has run them together. Some starting
   suspicions, not an exhaustive list:
     - each model through ModelServer's four endpoints
     - each model through PointwiseRanker / PairwiseRanker / FeatureCrossRanker
     - each reranker over each ranker's output
     - EnsembleRetriever with mixed model types
     - OnlineRecommender.fold_in_user / add_items after a save/load
     - BatchInferenceEngine with a model whose predict is not vectorised
     - corerec.evaluation.Evaluator against each model
     - a model saved, reloaded, then put through the pipeline
     - corerec.data classes fed into models that expect the triple

3. Write a SHORT script under /tmp that exercises it end to end with tiny
   synthetic data (about 40 users, 60 items, 300 interactions, few epochs --
   this is a correctness hunt, not a benchmark). Run it.

4. If it fails: reproduce it in the smallest script that still fails, confirm
   the failure is in CoreRec rather than in your test, and append an entry to
   Findings/bug.md using the template below.

   If it passes: append a row to the "Combinations exercised" table saying so.
   A verified-working combination is a real result and stops the next iteration
   repeating it.

5. Do NOT fix anything. Do NOT commit, push, or modify any file except
   Findings/bug.md. Reporting is the whole job; fixes get their own session
   where they can be reviewed.

6. Keep each iteration to one combination. Stop and write it up rather than
   chasing a second lead.

## What counts as a bug

  YES  a public API path raises when used as documented
  YES  two components that should compose cannot
  YES  a documented argument is accepted and silently ignored
  YES  save/load changes behaviour
  YES  an error message that does not name the cause
  YES  a default that produces near-random output (SAR's lift scored 0.0007)

  NO   a missing optional dependency, when the error names the extra
  NO   a model being merely slow or inaccurate; that is BENCHMARKS.md's job
  NO   anything you had to reach past a leading underscore to trigger
  NO   style, naming, or type-hint complaints

## Entry template — append, never overwrite

### [N]. <one-line summary>

**Layers:** <e.g. models x serving>
**Severity:** breaks-on-use | silent-wrong-result | confusing-error
**Found:** <date>

Reproduce:

```python
<the smallest script that fails>
```

Expected: <what a user would reasonably expect>
Actual: <the error, or the wrong value>
Root cause: <one or two lines, if you found it -- file and line>
Suspected blast radius: <which other components share the assumption>

---

Before finishing an iteration, verify Findings/bug.md is valid markdown and that
the numbering continues from the previous entry.
```

---

## Notes on running this in Termux

- It only writes `Findings/bug.md` and scratch files under `/tmp`, so an
  unattended loop cannot damage the repo.
- CPU-only, tiny synthetic data, a few epochs — it will not heat a phone the way
  the benchmarks do.
- `pip install corerec[serving]` first, or every serving combination reports a
  missing-extra error rather than a real bug.
- Check in on it. A loop that finds nothing for six iterations is telling you the
  combination list needs widening, not that the library is clean.
