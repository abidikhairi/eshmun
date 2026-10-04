"""Aborting with a readable message instead of a traceback.

Every script here reports a fatal, expected problem -- a missing file, a
recipe typo, a tokenizer that cannot render the prompt -- the same way: the
message through the logging setup, then `SystemExit(1)`. The helper lives here
so the scripts share one definition; they import it as `fail`, or as
`fail as _fail` to keep their call sites unchanged.
"""

from __future__ import annotations

import logging
from typing import NoReturn


def fail(message: str) -> NoReturn:
    """Abort with a readable message instead of a traceback."""
    logging.getLogger("eshmun").error(message)
    raise SystemExit(1)
