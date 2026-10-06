#!/usr/bin/env python
"""
CoreRec Command Line Interface
"""
from typing import List, Dict
import argparse
import warnings
import os
import sys

# Suppress all warnings at environment level
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# Redirect stderr to suppress MKL and other system warnings

warnings.filterwarnings("ignore")

# Save original stderr and redirect to /dev/null
_original_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
except BaseException:
    pass


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

  version       Show CoreRec version information
  engines       List all available recommendation engines
  models        List available models (optionally by family)  
  info          Show installation and dependency info
  examples      Show quick usage examples
  help          Show this help message

Usage Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
