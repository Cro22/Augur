"""Make the sidecar package importable when running pytest from sidecar/.

Keeps the suite runnable with a bare ``pytest`` and no editable install, so the
checkpoint is one command on a clean checkout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
