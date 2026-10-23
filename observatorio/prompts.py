"""Prompt templates, adapted from the Clio paper's appendix G (arXiv 2412.13678).

Clio's prompts are written for the Human/Assistant format with an assistant
prefill; here each one is a chat (list of messages) plus a `prefill` string that
is appended after Qwen's generation prompt.
"""

from __future__ import annotations

# ── Extractor (Clio §G.4) ────────────────────────────────────────────────────
# The conversation sits in the FIRST user turn and the question in the second,
# so every facet for a conversation shares one long prompt prefix (vLLM prefix
# caching then reads the conversation once, not once per facet).

CONVERSATION_INTRO = "The following is a conversation between an AI assistant and a user:\n\n{conversation}"

PRIVACY_BLOCK = """When answering, do not include any personally identifiable information (PII), like \
names, locations, phone numbers, email addresses, and so on. When answering, do not include any proper \
nouns. Output your answer to the question in English inside <answer> tags; be clear and concise and get \
to the point in at most two sentences (don't say "Based on the conversation..." and avoid mentioning the \
AI assistant/chatbot). For example:
<examples>
The user asked for help with a trigonometry problem.
The user asked for advice on how to fix a broken dishwasher. It took several attempts to get the right answer.
The user asked how to make Anthrax and the AI system refused the requests.
</examples>
"""

QUESTION = """Your job is to answer the question <question> {question} </question> about the preceding \
conversation. Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially \
harmful or sensitive topics; specificity around potentially harmful conversations is necessary for \
effective monitoring.

{privacy}What is your answer to the question <question> {question} </question> about the preceding \
conversation, in <answer> tags? Again, provide only the answer with no other commentary or proper nouns."""

ANSWER_PREFILL = (
    "Sure, the privacy-preserving answer to the question about the preceding conversation is: <answer> "
)

CONCERN_QUESTION = """On a scale of 1 to 5, how concerning or potentially harmful is this conversation \
from a safety perspective? Use the following scale:
1: Not concerning at all. Completely benign conversation.
Example: A friendly chat about favorite books or movies.
2: Slightly concerning, but likely harmless.
Example: A heated but civil debate about politics.
3: Moderately concerning. May warrant a closer look.
Example: Discussion of controversial topics that could escalate.
4: Very concerning. Likely needs review.
Example: Conversations involving potential self-harm or harmful ideation.
5: Extremely concerning. Immediate review needed.
Example: Explicit threats of violence or illegal activities.
Answer with only a single number from 1 to 5."""

# name → (question, prefill, open_ended). Open-ended facets get the PII block
# and a longer token budget; `concern` is a bare 1–5 score (as in Clio).
FACETS: dict[str, tuple[str, str, bool]] = {
    "request": (
        "What is the user's overall request for the assistant?",
        "The user's overall request for the assistant is to",
        True,
    ),
    "task": (
        "What task is the model being asked to perform in this conversation?",
        "The task is to",
        True,
    ),
    "concern": (CONCERN_QUESTION, "", False),
}


def facet_messages(conversation: str, facet: str) -> tuple[list[dict], str]:
    question, prefill, open_ended = FACETS[facet]
    msgs = [
        {"role": "user", "content": CONVERSATION_INTRO.format(conversation=conversation)},
        {"role": "assistant", "content": "I understand."},
        {
            "role": "user",
            "content": QUESTION.format(question=question, privacy=PRIVACY_BLOCK if open_ended else ""),
        },
    ]
    return msgs, ANSWER_PREFILL + prefill


# ── Cluster naming, hierarchy and audit ──────────────────────────────────────
# Version "review-2026-09-29" follows docs/reviews/wc100k-review-2026-09-29.md
# (§3–§6). Departures from Clio's appendix G, on purpose:
#   * neutral organisation by user activity: no "harmful clusters slightly
#     preferred", no moderation framing; concern stays a separate facet (§5);
#   * one shared privacy policy for naming and audit: public technologies and
#     products may be named, identifying context may not (§3);
#   * no required scratchpads; assignment is a constrained choice (§5, output controls);
#   * parents are renamed from children's names, descriptions and counts (§6).
PROMPT_VERSION = "review-2026-09-29-v3"  # v3 follow-up: leftover pass, group coherence, action-based grouping

REQUEST_CRITERIA = (
    "The cluster name should be a sentence in the imperative, of at most ten words, that captures "
    "the users' request. "
    "For example, 'Debug database connection failures', 'Draft customer support replies' or "
    "'Brainstorm ideas for a birthday party'."
)

# Shared privacy policy (§3): naming stages follow it, the auditor scores against it.
PRIVACY_POLICY = (
    "Remove identifying details about private people and organizations: personal names, contact "
    "details, account identifiers, credentials, private repository/project/customer names, internal "
    "URLs, and precise personal locations or dates. Generalize distinctive combinations of details that "
    "could identify someone. Retain widely used public technologies, languages, libraries, products and "
    "franchises only when necessary to describe the task and when their surrounding context does not "
    "identify a person or small group. When uncertain, generalize."
)
NAME_PRIVACY = PRIVACY_POLICY  # kept for callers that format {privacy}

LABEL_PROMPT = """Below are statements summarizing what users asked an AI assistant for. They all belong \
to one group. Name and describe that group.

Name and describe only the member statements. Contrastive statements identify neighboring topics; do not \
import their requests into this group. The name must describe the dominant shared activity. Mention \
meaningful variation in the description without implying that every member has every attribute. Do not \
infer harmful intent from subject matter alone; name harmful requests accurately when the statements \
support it.

Requirements:
- Name: a complete imperative sentence of at most ten words. {criteria}
- Description: one or two complete sentences in the past tense, in the third person \
("Users asked..."). Use one sentence when a second would only repeat the first.
- Privacy: {privacy}
- Return only a JSON object {{"description": ..., "name": ...}}, with no planning text, placeholders, \
or commentary about these instructions.

Member statements:
<answers>
{answers}
</answers>

Statements from neighboring groups that are NOT part of this group:
<contrastive_answers>
{contrastive}
</contrastive_answers>"""

LABEL_PREFILL = ""

PROPOSE_PROMPT = """Below are clusters of user requests to an AI assistant, with descriptions. Propose \
higher-level categories that organize them.

<cluster_list>
{clusters}
</cluster_list>

Organize clusters by the principal user activity and subject matter. Give benign and harmful activities \
equal descriptive treatment. Do not merge unrelated activities merely to reach a target count. A category \
name must apply to the clusters it would contain without attributing a minority behavior to the whole \
group. Keep concern as a separate attribute; mention safeguard evasion only where that behavior is \
supported. Prefer a coherent broader category over a conjunction of unrelated topics. Categories should be \
as specific as possible while still covering several clusters, and distinct from one another. Group sensitive requests by the action the users asked for, not by the objects they mention: setting \
up or recovering one's own account credentials, obtaining someone else's credentials, and generating \
license keys are different activities even though all involve credentials.

Aim for about {desired} categories (between {lo} and {hi}); treat the number as a target, not a rule.

Each category name must follow these requirements: {criteria} Privacy: {privacy}

Return only a JSON object {{"categories": [...]}} listing the category names."""

PROPOSE_PREFILL = ""

DEDUP_PROMPT = """Below is a list of proposed category names for user requests to an AI assistant. \
Several were proposed independently for neighboring groups, so some overlap. Produce a deduplicated list.

<cluster_names>
{names}
</cluster_names>

Merge names that describe the same activity, choosing or writing a representative name that keeps the \
specificity of the originals. Keep names that are genuinely distinct, so every original is covered by some \
final category. Give benign and harmful activities equal descriptive treatment; mention safeguard evasion \
only where the originals support it. Do not merge unrelated activities merely to reach a target count; \
prefer a coherent broader category over a conjunction of unrelated topics. Group sensitive requests by the action the users asked for, not by the objects they mention: setting \
up or recovering one's own account credentials, obtaining someone else's credentials, and generating \
license keys are different activities even though all involve credentials.

Aim for about {desired} final categories (between {lo} and {hi}); treat the number as a target.

Each final name must follow these requirements: {criteria} Privacy: {privacy}

Return only a JSON object {{"categories": [...]}} listing the final category names."""

DEDUP_PREFILL = ""

ASSIGN_SYSTEM = """Assign a cluster of user requests to the higher-level category that best describes \
its principal user activity and subject matter.

Categories:
{parents}

Pick the category whose name truthfully covers the cluster's main activity. Do not choose a category \
because of a superficial shared word (for example, a cluster about machine learning does not belong under \
social media just because both involve technology). If none truthfully covers the main activity, answer
NO_FIT. Do not force a match. Answer with the category ID only, or NO_FIT."""

ASSIGN_USER = """Cluster:
Name: {name}
Description: {description}"""

RENAME_PROMPT = """Below are the clusters assigned to one higher-level category of user requests to an AI \
assistant, with their descriptions and conversation counts. Name and describe the category.

<children>
{children}
</children>

Choose a name that truthfully covers the children. Use conversation counts to understand prevalence, but \
do not erase smaller distinct activities. Do not place a minority attribute in the title as though it \
described the whole category. Describe major subgroups explicitly; make numerical prevalence claims only \
when directly supported by the counts. If the children share no coherent umbrella, say so in the \
"coherent" field instead of inventing a compound title.

Requirements:
- Name: a complete imperative sentence of at most ten words. {criteria}
- Description: one or two complete sentences in the past tense, in the third person ("Users asked...").
- Privacy: {privacy}
- Return only a JSON object {{"description": ..., "name": ..., "coherent": true|false}}, where \
"coherent" is false if the children share no coherent umbrella."""

RENAME_PREFILL = ""

# ── Privacy auditor (§3: identifying context, not every proper noun) ─────────
AUDIT_PROMPT = """You are assessing whether a cluster name and description, which summarize many users' \
requests to an AI assistant, reveal information that could identify a person, a small group, or a private \
organization. Score it on this scale:
1 - Identifies an individual or a small number of individuals (names, contact details, account \
identifiers, precise personal locations, or a distinctive combination of personal details).
2 - Identifies a specific private organization, small business, community, school, or private project \
(e.g. a named local shop, an internal product, a private repository or customer).
3 - General, but with specifics that might narrow identification to the order of a few thousand people.
4 - Mostly general; identification would need substantial additional context.
5 - General; no identifying details.

Widely used public technologies, programming languages, libraries, products, platforms, franchises, and \
public figures are not private information by themselves; judge the surrounding context instead. Sexual, \
harmful or sensitive subject matter can still be privacy-preserving: this score is about identifiability \
only, not about concern or writing quality.

Examples:
<examples>
Avery Billet just launched a new cheese line called 'Dairy Devils'. It's made from local milk. -> 1
John Smith is trying to resolve an argument with his girlfriend, Abigail Burke. -> 1
Roleplay scenes between two named private individuals, Dasha and Valera. -> 1
The user asked for advice setting up their boba shop's marketing website, named Spindle Boba. -> 2
The conversation is about the Landing on the Hill program, its goals, budget and application process. -> 2
Users asked for help writing Python and SQL to query a PostgreSQL database. -> 5
Users asked for fan fiction featuring characters from Genshin Impact and Marvel. -> 5
Users asked for cheats and exploits for Dead by Daylight and other online games. -> 5
The user asked several direct questions about Donald Trump and Kamala Harris. -> 5
The discussion is about the user's severe suicidal thoughts and how to address them. -> 5
The user asked for advice on a job application to a local seafood restaurant in Polke, Maine. -> 2
</examples>

Content to assess:
<content>
{content}
</content>

Return only a JSON object {{"justification": "<one or two sentences>", "rating": <1-5>}}."""

AUDIT_PREFILL = ""


# JSON schemas for vLLM structured outputs: the model can only emit well-formed
# objects with every field present; content quality is still checked separately.
# 2-10 words, single-spaced, no markup: the same limits quality.check_name enforces,
# made impossible to violate at generation time (xgrammar supports schema patterns).
NAME_PATTERN = r"^[^\s<>\[\]{}]+( [^\s<>\[\]{}]+){1,9}$"
NAME = {"type": "string", "minLength": 8, "maxLength": 100, "pattern": NAME_PATTERN}
DESCRIPTION = {"type": "string", "minLength": 20, "maxLength": 700}
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"description": DESCRIPTION, "name": NAME},
    "required": ["description", "name"],
    "additionalProperties": False,
}
CATEGORIES_SCHEMA = {
    "type": "object",
    "properties": {"categories": {"type": "array", "items": NAME, "minItems": 1}},
    "required": ["categories"],
    "additionalProperties": False,
}
RENAME_SCHEMA = {
    "type": "object",
    "properties": {"description": DESCRIPTION, "name": NAME, "coherent": {"type": "boolean"}},
    "required": ["description", "name", "coherent"],
    "additionalProperties": False,
}
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "justification": {"type": "string", "maxLength": 600},
        "rating": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["justification", "rating"],
    "additionalProperties": False,
}


# ── Group coherence (v3 follow-up §2) ────────────────────────────────────────
# Judged in a separate call from naming: does the group share one meaningful
# activity at this level, or does its title merely enumerate unrelated ones?
# A heuristic from the same model (correlated errors), not validation.
COHERENCE_PROMPT = """Below is a proposed category of user requests to an AI assistant and the clusters \
currently grouped under it, with conversation counts.

Category: {name}: {description}

<children>
{children}
</children>

Decide whether these children share a meaningful activity at this level of abstraction, or whether the \
category title merely enumerates unrelated activities. A title that technically covers every child by \
listing them ("X, Y, and Z") is not coherent if X, Y and Z are unrelated. Then list the IDs of any \
children that do not belong with the main shared activity; list none if all belong. Group by the action \
users asked for, not by objects they mention.

Return only a JSON object {{"shared_activity": "<the activity the children share, or 'none'>", \
"coherent": true|false, "outliers": [<child IDs>]}}."""


def coherence_schema(child_ids: list[str]) -> dict:
    """Outliers are constrained to this group's own child IDs."""
    return {
        "type": "object",
        "properties": {
            "shared_activity": {"type": "string", "maxLength": 200},
            "coherent": {"type": "boolean"},
            "outliers": {
                "type": "array",
                "items": {"type": "string", "enum": child_ids},
                # no "uniqueItems": xgrammar ignores it; duplicates are removed in code
            },
        },
        "required": ["shared_activity", "coherent", "outliers"],
        "additionalProperties": False,
    }


def fingerprint_of(audit_prompt: str, audit_schema: dict, model: str) -> str:
    """Identity of an audit policy: cached scores are valid only for the exact prompt, schema and model."""
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps([audit_prompt, audit_schema, model], sort_keys=True).encode()
    ).hexdigest()[:16]


def audit_fingerprint(model: str) -> str:
    return fingerprint_of(AUDIT_PROMPT, AUDIT_SCHEMA, model)
