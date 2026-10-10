"""Moved to ``corerec.experimental.integrations`` (#79): untested and unused by CoreRec."""

import warnings

warnings.warn("corerec.integrations moved to corerec.experimental.integrations, which is untested; "
              "import it from there", DeprecationWarning, stacklevel=2)

from corerec.experimental.integrations import *  # noqa: E402,F401,F403
from corerec.experimental.integrations import __all__  # noqa: E402,F401
