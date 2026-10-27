"""K=1 structured single-adapter baseline.

Stage B is intentionally *not* a warm start for Stage C.  Both start from the
same Base checkpoint and initialise their adapter independently.  This module
only documents and builds the K=1 variant; the shared factory lives in
``sbona.baselines``.
"""

from __future__ import annotations

from sbona.models.adapter import SBoN

SingleAdapter = SBoN

__all__ = ["SingleAdapter"]
