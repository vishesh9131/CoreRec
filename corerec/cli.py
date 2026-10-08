#!/usr/bin/env python
"""
CoreRec Command Line Interface
"""
import argparse
import warnings
import os
import sys

# Quiet the MKL threading notices and library warnings. stderr itself stays
# open: `corerec train` and `corerec serve` must be able to report errors.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
warnings.filterwarnings("ignore")


# Try importing argcomplete for tab completion
try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


def get_version():
    """Get CoreRec version."""
    try:
        from corerec import __version__

        return __version__
    except ImportError:
        return "Unknown"


def show_version(args):
    """Display CoreRec version information."""
    version = get_version()
    print(
        f"""
╔══════════════════════════════════════════════╗
║           CoreRec Version Info               ║
╚══════════════════════════════════════════════╝

Version: {version}
Python Package: corerec
Description: Advanced Recommendation Systems Library

Repository: https://github.com/vishesh9131/CoreRec
Mail : sciencely98@gmail.com
    """
    )


def list_engines(args):
    """List every shipped model, grouped by family."""
    from corerec.engines import get_engine_info

    print("\nCoreRec models (corerec.engines):")
    for family, models in get_engine_info().items():
        print(f"\n{family.upper()}:")
        for name, summary in models.items():
            print(f"  - {name:<18} {summary}")
    print("\nUsage:\n  from corerec.engines import ALS\n  model = ALS(factors=64)")


def list_models(args):
    """List available models, optionally for one family."""
    category = getattr(args, "category", "all")
    from corerec.engines import list_models as names

    if category == "all":
        list_engines(args)
        return
    print(f"\n{category.upper()} models:")
    for name in names(category):
        print(f"  - {name}")


def show_info(args):
    """Show information about CoreRec installation."""
    import sys

    version = get_version()

    print(
        f"""
╔══════════════════════════════════════════════╗
║        CoreRec Installation Info             ║
╚══════════════════════════════════════════════╝

CoreRec Version: {version}
Python Version: {sys.version.split()[0]}
Python Path: {sys.executable}

Installation Status:
"""
    )

    # Check for key dependencies
    dependencies = ["torch", "pandas", "numpy", "sklearn", "scipy"]

    for dep in dependencies:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {dep:<15} {ver}")
        except ImportError:
            print(f"  ✗ {dep:<15} NOT INSTALLED")

    print("\nFor more info: corerec info --verbose")


def show_examples(args):
    """Show example usage code."""
    print(
        """
╔══════════════════════════════════════════════╗
║          CoreRec Quick Examples              ║
╚══════════════════════════════════════════════╝

1. Matrix factorization (ALS):
from corerec.engines import ALS

model = ALS(factors=64, iterations=15)
model.fit(user_ids, item_ids, ratings)
recommendations = model.recommend(user_id=123, top_k=10)


2. Ranking (DCN):
from corerec.engines import DCN

model = DCN(embedding_dim=64, epochs=10)
model.fit(user_ids, item_ids, ratings)
recommendations = model.recommend(user_id=123, top_k=10)


3. Sequential (SASRec):
from corerec.engines import SASRec

model = SASRec(hidden_units=64, epochs=10)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
next_items = model.recommend(user_id=123, top_k=10)


4. Content-based (TF-IDF):
from corerec.engines import TFIDFRecommender

model = TFIDFRecommender()
model.fit(items=[101, 102], docs={101: "action film", 102: "romantic comedy"})
similar_items = model.recommend_by_text(query_text="action", top_n=5)


For more examples, visit:
https://github.com/vishesh9131/CoreRec/tree/main/examples
    """
    )


def show_help(args):
    """Display detailed help information."""
    print(
        """
╔══════════════════════════════════════════════╗
║           CoreRec CLI Help                   ║
╚══════════════════════════════════════════════╝

Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  train         Train on an interactions file and save an artifact
  serve         Serve recommendations over HTTP (from a file or artifact)
  version       Show CoreRec version information
  engines       List all available recommendation engines
  models        List available models (optionally by family)  
  info          Show installation and dependency info
  examples      Show quick usage examples
  help          Show this help message

Usage Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  corerec serve events.csv     # Train on a CSV and serve it on :8000
  corerec train events.csv -o artifacts/m
  corerec version              # Show version info
  corerec engines              # List all engines
  corerec models               # List all models
  corerec models ranking       # List one model family    
  corerec info                 # Show installation info
  corerec examples             # Show code examples

For more information, visit:
https://github.com/vishesh9131/CoreRec
    """
    )


def _add_training_args(p):
    p.add_argument("--model", default="ALS",
                   help="model to train (default: ALS; `corerec models` lists all)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="model constructor argument, repeatable (e.g. --param factors=128)")
    p.add_argument("--user-col", help="user column (default: detected)")
    p.add_argument("--item-col", help="item column (default: detected)")
    p.add_argument("--rating-col", help="rating/weight column (default: detected, else 1 per row)")
    p.add_argument("--timestamp-col", help="timestamp column (default: detected)")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="share of each user's interactions held out for evaluation (default 0.2)")
    p.add_argument("--k", type=int, default=10, help="cutoff for NDCG@k / Recall@k (default 10)")
    p.add_argument("--no-eval", action="store_true", help="skip the holdout evaluation")
    p.add_argument("--seed", type=int, default=42)


def _train(args):
    from corerec.serving.from_csv import parse_params, train_from_csv

    print(f"Reading {args.data}")
    result = train_from_csv(
        args.data, model=args.model, params=parse_params(args.param),
        evaluate=not args.no_eval, test_fraction=args.test_fraction, k=args.k, seed=args.seed,
        user=args.user_col, item=args.item_col, rating=args.rating_col,
        timestamp=args.timestamp_col,
    )
    print(result.report())
    return result


def train_command(args):
    """corerec train FILE -o DIR"""
    from corerec.serving.from_csv import save_artifact

    result = _train(args)
    out = save_artifact(result, args.output)
    print(f"Saved     {out}/  (serve it with: corerec serve {out})")


def serve_command(args):
    """corerec serve FILE|ARTIFACT"""
    from corerec.serving.from_csv import build_server, is_artifact, load_artifact, save_artifact

    artifact = None
    if is_artifact(args.data):
        model, manifest = load_artifact(args.data)
        artifact = args.data
        print(f"Loaded    {manifest['model']} from {args.data}")
    else:
        result = _train(args)
        manifest = result.manifest()
        model = result.model
        if args.save:
            print(f"Saved     {save_artifact(result, args.save)}/")
    challenger = None
    if args.challenger:
        challenger = load_artifact(args.challenger)[0]
        print(f"A/B       challenger {args.challenger} gets {args.challenger_share:.0%} of users")
    server = build_server(model, manifest, host=args.host, port=args.port,
                          feedback_log=args.feedback_log, challenger=challenger,
                          challenger_share=args.challenger_share, artifact=artifact)
    if args.feedback_log:
        print(f"Feedback  logging to {args.feedback_log}  (POST /feedback, GET /metrics)")
    print(f"Serving   http://{args.host}:{args.port}  (docs at /docs, Ctrl-C to stop)")
    print(f"""Try       curl -X POST localhost:{args.port}/recommend -H 'Content-Type: application/json' -d '{{"user_id": "<a user>", "top_k": 5}}'""")
    server.start()


def retrain_command(args):
    """corerec retrain ARTIFACT [--data FILE] [--feedback LOG]"""
    from corerec.serving.from_csv import retrain_artifact

    d = retrain_artifact(args.artifact, data=args.data, feedback=args.feedback,
                         tolerance=args.tolerance, k=args.k, dry_run=args.dry_run)
    print(f"Data      {d['rows']:,} interactions, {d['new_rows']:,} new since the last training "
          f"({d['feedback_rows']:,} from feedback)")
    if d["candidate"] is None:
        print(f"Kept      the current model: {d['reason']}")
        return d
    print(f"Test      the newest {d['test_rows']:,} interactions, unseen by both models")
    print(f"          {d['metric']:>10}")
    print(f"current   {d['current']:>10.4f}")
    print(f"candidate {d['candidate']:>10.4f}")
    if d["promoted"]:
        print(f"Promoted  the candidate; the old model is in {args.artifact}/previous "
              f"(POST /reload to serve it without a restart)")
    elif d["would_promote"]:
        print("Dry run   the candidate would be promoted")
    else:
        print("Kept      the current model; the candidate scored lower")
    return d


def main():
    """Main entry point for CoreRec CLI."""
    parser = argparse.ArgumentParser(
        description="CoreRec Command Line Interface",
        prog="corerec")

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute")

    # Version command
    version_parser = subparsers.add_parser(
        "version", help="Show CoreRec version")
    version_parser.set_defaults(func=show_version)

    # Engines command
    engines_parser = subparsers.add_parser(
        "engines", help="List all available engines")
    engines_parser.set_defaults(func=list_engines)

    # Models command
    models_parser = subparsers.add_parser(
        "models", help="List available models")
    models_parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=["all", "classic", "retrieval", "graph", "ranking", "sequential",
                 "autoencoder", "content"],
        help="Model family to list",
    )
    models_parser.set_defaults(func=list_models)

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train a model on an interactions file and save it",
        description="Train on a CSV/TSV/Parquet of interactions, report holdout "
                    "NDCG/Recall against a most-popular baseline, and save an artifact.")
    train_parser.add_argument("data", help="interactions file (one row per user-item event)")
    train_parser.add_argument("-o", "--output", required=True, help="artifact directory to write")
    _add_training_args(train_parser)
    train_parser.set_defaults(func=train_command)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Serve recommendations over HTTP from a file or a saved artifact",
        description="Given an interactions file: train, report, then serve. Given a "
                    "directory written by `corerec train`: load and serve it.")
    serve_parser.add_argument("data", help="interactions file, or artifact directory")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--save", metavar="DIR", help="also save the trained artifact here")
    serve_parser.add_argument("--feedback-log", metavar="FILE",
                              help="log impressions and feedback here; enables /feedback and /metrics")
    serve_parser.add_argument("--challenger", metavar="ARTIFACT",
                              help="A/B test: serve this artifact to a share of users")
    serve_parser.add_argument("--challenger-share", type=float, default=0.1,
                              help="share of users the challenger gets (default 0.1)")
    _add_training_args(serve_parser)
    serve_parser.set_defaults(func=serve_command)

    retrain_parser = subparsers.add_parser(
        "retrain", help="Retrain an artifact on fresh data and feedback; replace it only if it isn't worse",
        description="Run from cron to keep a served model fresh: retrains with the artifact's own "
                    "model and settings, compares against the current model on a recent holdout, "
                    "and swaps it in only when the candidate scores at least as well.")
    retrain_parser.add_argument("artifact", help="artifact directory written by corerec train")
    retrain_parser.add_argument("--data", help="interactions file (default: the artifact's training file)")
    retrain_parser.add_argument("--feedback", metavar="LOG", help="feedback log from corerec serve --feedback-log")
    retrain_parser.add_argument("--tolerance", type=float, default=0.0,
                                help="promote if candidate NDCG >= current - tolerance (default 0)")
    retrain_parser.add_argument("--k", type=int, default=10)
    retrain_parser.add_argument("--dry-run", action="store_true", help="compare only; change nothing")
    retrain_parser.set_defaults(func=retrain_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show installation info")
    info_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information")
    info_parser.set_defaults(func=show_info)

    # Examples command
    examples_parser = subparsers.add_parser(
        "examples", help="Show usage examples")
    examples_parser.set_defaults(func=show_examples)

    # Help command
    help_parser = subparsers.add_parser("help", help="Show detailed help")
    help_parser.set_defaults(func=show_help)

    # Enable tab completion if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 1

    # Execute the command
    if hasattr(args, "func"):
        args.func(args)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
