"""augur_predict — the optional Python sidecar for Augur.

Augur's v1 is pure Go: it measures an agent's real token usage by running it
through a recording proxy, then projects the bill at production scale. This
sidecar is the one analytical piece the SPEC deliberately defers to a second
language (Hito 5): a *predictive output-length model*.

The premise is PreflightLLMCost's: completion length — not prompt length — is
what you cannot guess for an agent, and it is the dominant cost driver in the
tail. So we learn it from a trace Augur already recorded, then use that model to
estimate cost for inputs we have NOT run, without spending tokens.

Coupling to the Go core is deliberately loose: this package reads the same
JSONL cost-trace the proxy writes and (for ``emit-trace``) writes one back. No
RPC, no shared library, no import in either direction — just the trace file as
the contract. You can delete this directory and Augur's v1 is untouched.
"""

__version__ = "0.1.0"

from .model import Model, ModelFit, fit
from .trace import Record, load_trace

__all__ = ["Model", "ModelFit", "fit", "Record", "load_trace", "__version__"]
