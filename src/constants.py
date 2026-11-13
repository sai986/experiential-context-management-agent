"""
Central location for all tag names, operation names, and model IDs.

Importing from here (instead of scattering string literals) means a typo
is caught at import time and a rename only needs to happen in one place.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Observation tags
# ---------------------------------------------------------------------------

# Applied to the most important constraint in a task — must never be lost.
TAG_CRITICAL = "critical"

# A binding rule that cannot be relaxed (e.g. "must not use cloud").
TAG_HARD_CONSTRAINT = "hard_constraint"

# Content contains a negation (e.g. "do not", "must not").
TAG_NEGATION = "negation"

# A constraint stated early that only becomes relevant several steps later.
TAG_DELAYED_DEPENDENCY = "delayed_dependency"

# Content that looks relevant but is not needed to answer the question.
TAG_DISTRACTOR = "distractor"

# The question item itself — managers must never compress or archive this.
TAG_TRIGGER = "trigger"

# A user preference that informs tone but is not a hard requirement.
TAG_PREFERENCE = "preference"

# Tags that make a decision suspicious for counterfactual replay.
SUSPECT_TAGS: frozenset[str] = frozenset({
    TAG_CRITICAL,
    TAG_HARD_CONSTRAINT,
    TAG_NEGATION,
    TAG_DELAYED_DEPENDENCY,
})

# ---------------------------------------------------------------------------
# Context operations
# ---------------------------------------------------------------------------

OP_KEEP       = "KEEP"
OP_COMPRESS   = "COMPRESS"
OP_ARCHIVE    = "ARCHIVE"
OP_UNARCHIVE  = "UNARCHIVE"
OP_RESTORE    = "RESTORE"

# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

FAMILY_HARD_CONSTRAINT    = "hard_constraint"
FAMILY_DELAYED_DEPENDENCY = "delayed_dependency"
FAMILY_DEPENDENCY_CHAIN   = "dependency_chain"
FAMILY_FALSE_DISTRACTOR   = "false_distractor"
FAMILY_NEGATION_CRITICAL  = "negation_critical"
FAMILY_SYNONYM_MATCH      = "synonym_match"

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------

# Default model for all LLM calls (compression, investigation, task agent).
MODEL_DEFAULT = "claude-sonnet-4-6"

# Lighter model for compression — cheaper, still accurate for one-sentence summaries.
MODEL_COMPRESSION = "claude-haiku-5-5"
