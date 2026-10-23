# Coherence annotation rubric v2

Purpose: evaluate whether a hierarchy is useful to browse and whether its labels accurately describe
its members. These are separate judgments. This is a diagnostic set selected around known failures,
not a representative sample for estimating whole-run accuracy.

## Review procedure

Read the parent name/description, level, and every child's name **and description** before inspecting
draft labels or model verdicts. Use `coherence-set-v1.md` as the blind worksheet; the JSONL retains
provenance and drafts for later adjudication. Reviewers who have already seen the verdicts should
record that exposure; hiding them now does not undo it. Do not confirm an example from its name alone.

Record both judgments, a short rationale, reviewer identity, rubric version `v2`, and any outlier child
IDs in the JSONL `labels` object. Leave uncertain labels unconfirmed and explain uncertainty in a
review note. A model may propose labels but must not certify its own drafts as human ground truth.
Ideally, two people independently review borderline examples and adjudicate disagreements.

## Coherence: yes / partial / no

At level 1, prefer a specific task family or domain with closely related user needs. At level 2, broader
workflow and domain umbrellas are acceptable. Breadth or a large conversation share alone is not a
defect. Both operations (debugging across technologies) and subjects (travel logistics across transit,
lodging, and attractions) can support a useful category. State the grouping principle in the rationale
and apply it consistently; do not accept a shared subject in one example while rejecting another solely
because its operations differ. Generic properties such as “uses AI,” “asks questions,” or “generates
content” are not enough on their own.

- **yes:** all children fit a useful common activity, workflow, or sufficiently specific domain at this
  level. Different operations within that umbrella do not automatically create outliers.
- **partial:** there is one defensible core of at least two children that can be retained, with specific
  outliers removed and the group renamed if necessary. List those outliers and describe the retained
  umbrella. The core must be substantively useful, not merely any pair chosen from a miscellaneous list.
  Counts inform how much is retained but do not determine correctness or excuse small misfits.
- **no:** no single defensible core supports trimming as the repair; the group needs partitioning into
  multiple peer groups or individual children. Explain why the apparent commonality is too generic or
  why choosing one core would be arbitrary. A compound title alone is not grounds for `no`.

Example applications: software development mixed with business planning and curricula has a plausible
software core (`partial`), while fact verification across arbitrary subjects plus sports-rule explanations
does not establish a specific common task/domain (`no`). Recipes, nutrition, and food recommendations
can share a food-related domain; adult-fiction generation and adult-media recommendations must likewise
be assessed for the usefulness of their domain umbrella, not rejected solely for different verbs.
Borderline domain breadth should be adjudicated, not hidden by a numeric majority rule.

## Title faithfulness: yes / no

Assess the name **and description as written**, against all child evidence, independently of coherence.

- **yes:** wording accurately covers or explicitly enumerates distinct subgroups without unsupported
  claims. “Develop software and plan businesses” can faithfully enumerate a poor grouping. A broad or
  awkward title may be faithful while still needing editorial improvement.
- **no:** wording asserts a shared behavior, intent, or attribute that the child evidence does not
  support, invents specifics, or materially misdescribes members. “Bypass safeguards to obtain content”
  attributes evasion to the covered requests; explicit fiction or legitimate account setup does not
  establish that action. “Some requests attempted bypass” has a different scope from “Users attempted
  bypass” applied to the entire category.

The word “and” does not automatically distribute every listed activity to every child. Read the complete
name and description: “creative fiction and bypass requests,” with separate subgroups in the description,
can be faithful even though the grouping needs repair. If attribution remains ambiguous, record the
ambiguity for adjudication rather than silently choosing a harsher interpretation for sensitive topics.
Record child IDs contradicting an unfaithful claim in the rationale; these need not be the same as the
coherence outliers. A title error can require renaming rather than removing those children.

## Split and evaluation policy

Related main/leftover groups, overlapping child sets, and known cross-version variants belong to the
same family and split. Child IDs can change between versions, so compare names/descriptions and
provenance, not IDs alone. The grouped split is fixed independently of draft label values. Family
grouping reduces leakage but does not prove independence between all examples.

These examples have already been inspected during development. The held-out portion is a **reserved
regression set**, not a previously unseen test set. Do not tune on its diagnostics. For a claim about
generalization, obtain a fresh separately labeled set after freezing the prompt. Freeze confirmed labels,
family assignments, rubric, and split before a comparison run; version subsequent changes.

Report confirmed-label counts and excluded/unconfirmed counts. Score coherence and title faithfulness
separately. For `partial` groups, evaluate outlier selection as well as keep/dissolve behavior: retaining
the whole group is not a fully correct repair. Report failure-selected and control examples separately
before making prevalence claims. Draft-only diagnostics are not accuracy measurements.
