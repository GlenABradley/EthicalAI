from __future__ import annotations

from fastapi import APIRouter
from coherence.api.models import WhatIfRequest, WhatIfResponse

router = APIRouter()


@router.post("", response_model=WhatIfResponse)
def what_if(req: WhatIfRequest) -> WhatIfResponse:
    """Counterfactual what-if analysis (stub).

    TODO(@builder): Implement span-level recompute after edits. For now, return empty deltas.
    """
    return WhatIfResponse(deltas=[])
