"""Numeric conversions at the boundary.

Pandas hands back numpy scalars, and ``json.dumps`` cannot serialise them: a report that is not
JSON-serialisable is a report nobody downstream can consume. Every number that leaves this
package as a plain Python scalar goes through here, so the policy lives in one place instead of
at thirty call sites.

The guards are not ceremony. ``float("")``, ``float(None)`` and ``int("n/a")`` all raise, and a
column that arrives unexpectedly textual would otherwise take down a report halfway through
being written — which is the worst possible moment.
"""

from __future__ import annotations


def as_int(value: object, *, context: str) -> int:
    """A plain Python ``int``, with the offending context in the error."""
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected a count for {context}, got {value!r}") from exc


def as_float(value: object, *, context: str) -> float:
    """A plain Python ``float``, for the same reason as :func:`as_int`."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected a number for {context}, got {value!r}") from exc
