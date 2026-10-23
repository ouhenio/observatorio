"""Deterministic quality gate for generated cluster names and descriptions.

Separate from the privacy audit (which judges identifiability only): catches
missing or placeholder names, unfinished generations, markup residue and the
model commenting on its own instructions. Used to retry in label/hierarchy and
as a publish-time gate (review 2026-09-29, "Output controls").
"""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(
    r"^(?:(?:unnamed|unlabeled|unlabelled)(?: cluster)?(?:\s+L\d+-\d+)?|"
    r"name withheld|cluster(?:\s+(?:L\d+-\d+|\d+))?)$",
    re.I,
)
COMMENTARY = re.compile(
    r"(?:^|[.!?]\s+)(?:I will|I'll|I have|I've|I am|I'm|as an AI)\b|"
    r"\b(?:insert (?:your|the) (?:name|description|summary))\b|\[(?:name|description)\]",
    re.I,
)
# Our own prompt/answer tags; other markup (e.g. users asking about <br/>) is content.
OUR_TAGS = re.compile(
    r"</?(summary|name|answer|cluster|coherent|scratchpad|justification|rating|children|child)\b", re.I
)
TERMINAL = re.compile(r"[.!?][\"'”’)\]]?$")


def check_name(name: str | None) -> list[str]:
    """Hard failures only: topic words alone are not instruction leakage."""
    issues = []
    name = (name or "").strip()
    if not name or PLACEHOLDER.match(name):
        issues.append("missing name")
    else:
        words = len(name.split())
        if words < 2 or len(name) < 8:
            issues.append("name too short")
        if words > 10:
            issues.append("name too long")
        if re.search(r"[<>\[\]{}]", name):
            issues.append("markup in name")
        if COMMENTARY.search(name):
            issues.append("commentary in name")
    return issues


def check(name: str | None, description: str | None) -> list[str]:
    """Issues with a (name, description) pair; empty list = passes."""
    issues = check_name(name)
    description = (description or "").strip()
    if len(description) < 20:
        issues.append("missing description")
    else:
        if not TERMINAL.search(description):
            issues.append("unfinished description")
        if OUR_TAGS.search(description):
            issues.append("markup in description")
        if COMMENTARY.search(description):
            issues.append("commentary in description")
    return issues
