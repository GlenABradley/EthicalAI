from __future__ import annotations

"""Three-valued logic evaluation (Milestone 5).

Implements Kleene's K3 with values: TRUE, FALSE, UNKNOWN.
"""

from enum import Enum


class Truth(Enum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


def lnot(a: Truth) -> Truth:
    if a == Truth.TRUE:
        return Truth.FALSE
    if a == Truth.FALSE:
        return Truth.TRUE
    return Truth.UNKNOWN


def land(a: Truth, b: Truth) -> Truth:
    # Kleene AND truth table
    if a == Truth.FALSE or b == Truth.FALSE:
        return Truth.FALSE
    if a == Truth.TRUE and b == Truth.TRUE:
        return Truth.TRUE
    return Truth.UNKNOWN


def lor(a: Truth, b: Truth) -> Truth:
    # Kleene OR truth table
    if a == Truth.TRUE or b == Truth.TRUE:
        return Truth.TRUE
    if a == Truth.FALSE and b == Truth.FALSE:
        return Truth.FALSE
    return Truth.UNKNOWN
