"""
Pure logic for turning KG triples into a symbolic reasoning trace and a
constructed response, plus the shuffling policy.

No network calls, no file I/O -- everything here is a pure function so it can
be unit-tested directly.

Adapted from the InstructProtein-pilot version of this module (external repo,
scripts/python/thinking/reasoning.py): the triple/response-template/shuffling
logic is unchanged (already validated there -- the thinking-tuned pilot
checkpoint beat its zero-shot baseline on generation-direction extraction
rate and pLDDT). `encode_sequence` matches the pilot's own `Ƥ`-per-residue
join exactly (see its docstring), since the dataset currently trains that
same InstructProtein-based pilot, whose tokenizer has no protein-specific
vocabulary.
"""

import random
from dataclasses import dataclass

SUBJECT_PLACEHOLDER = "protein"

# Generation-direction reasoning cites CATEGORY-level aggregate stats (see
# build_generation_kg.py), not one instance's real facts -- so the triple
# subject should read as the category being described, not "protein" (which
# implies a claim about the one sequence being generated). One or two words,
# reusing GO's own namespace/term names where the relation is GO-backed.
GENERATION_CONTEXT_SUBJECTS = {
    "member_of": "family",
    "has_function": "molecular_function",
    "involved_in": "biological_process",
    "located_in": "cellular_component",
    "catalyzes": "catalytic_activity",
    "scop_fold": "fold",
    "scop_superfamily": "superfamily",
    "scop_family": "scop_family",
}

# Non-target relations are capped per-relation when building the reasoning
# context, so one chatty relation (e.g. involved_in with 15 GO terms) doesn't
# drown out the others.
MAX_CONTEXT_VALUES_PER_RELATION = 3


@dataclass(frozen=True)
class Triple:
    relation: str
    value: str
    subject: str = SUBJECT_PLACEHOLDER


def format_triple(triple: Triple) -> str:
    value = triple.value
    if triple.relation == "has_length":
        value = f"{value} amino acids"
    return f"({triple.subject}, {triple.relation}, {value})"


def with_subject(triples: list[Triple], subject: str) -> list[Triple]:
    """Re-labels a list of triples with a different subject, e.g. turning
    select_context_triples' default-"protein" output into "protein_a"/
    "protein_b" triples for two-subject reasoning traces (PPI annotation:
    both proteins are given inputs, so citing real facts about either one is
    legitimate context, unlike citing facts about a not-yet-generated
    sequence)."""
    return [Triple(relation=t.relation, value=t.value, subject=subject) for t in triples]


def select_context_triples(
    triples_by_relation: dict[str, list[str]],
    target_relation: str,
    rng: random.Random,
    max_per_relation: int = MAX_CONTEXT_VALUES_PER_RELATION,
) -> list[Triple]:
    """All triples except the target relation, capped per relation (sampled, not truncated)."""
    context: list[Triple] = []
    for relation, values in triples_by_relation.items():
        if relation == target_relation:
            continue
        chosen = values if len(values) <= max_per_relation else rng.sample(values, max_per_relation)
        context.extend(Triple(relation=relation, value=v) for v in chosen)
    return context


def shuffle_triples(triples: list[Triple], rng: random.Random) -> list[Triple]:
    """Returns a new, randomly-ordered list. Does not mutate the input.

    Order carries no meaning here: these are independent observations about
    the protein, not steps in a deduction, so the model must not be able to
    learn a fixed positional pattern.
    """
    shuffled = list(triples)
    rng.shuffle(shuffled)
    return shuffled


def build_reasoning_block(triples: list[Triple], rng: random.Random) -> str:
    ordered = shuffle_triples(triples, rng)
    lines = [format_triple(t) for t in ordered]
    return "<think>\n" + "\n".join(lines) + "\n</think>"


def join_values(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


RESPONSE_TEMPLATES = {
    # Values already contain "family"/"subfamily"/"superfamily" (e.g. "Cytochrome
    # P450 family"), so the template must not append it again.
    "member_of": {
        "single": "The protein belongs to the {joined}.",
        "multi": "The protein belongs to the {joined}.",
    },
    "has_function": {
        "single": "The protein has the molecular function of {joined}.",
        "multi": "The protein has the following molecular functions: {joined}.",
    },
    "involved_in": {
        "single": "The protein is involved in {joined}.",
        "multi": "The protein is involved in the following biological processes: {joined}.",
    },
    "located_in": {
        "single": "The protein is located in the {joined}.",
        "multi": "The protein is located in the following cellular components: {joined}.",
    },
    "catalyzes": {
        "single": "The protein catalyzes reactions classified under EC {joined}.",
        "multi": "The protein catalyzes reactions classified under the following EC numbers: {joined}.",
    },
    # "(SCOP classification)" suffix is deliberate: task family B's
    # scop_family is a structural-classification concept from a different
    # data source than member_of's SwissProt family string -- same English
    # word, different meaning, so the answer text disambiguates them too,
    # not just the internal relation name.
    "scop_fold": {
        "single": "The protein adopts the {joined} fold.",
        "multi": "The protein adopts the following folds: {joined}.",
    },
    "scop_superfamily": {
        "single": "The protein belongs to the {joined} superfamily.",
        "multi": "The protein belongs to the following superfamilies: {joined}.",
    },
    "scop_family": {
        "single": "The protein belongs to the {joined} family (SCOP classification).",
        "multi": "The protein belongs to the following families (SCOP classification): {joined}.",
    },
}


def build_response_text(target_relation: str, values: list[str]) -> str:
    if target_relation not in RESPONSE_TEMPLATES:
        raise ValueError(f"unsupported target relation: {target_relation}")
    if not values:
        raise ValueError("values must be non-empty")

    template_key = "single" if len(values) == 1 else "multi"
    template = RESPONSE_TEMPLATES[target_relation][template_key]
    return template.format(joined=join_values(values))


PROTEIN_PREFIX_TOKEN = "Ƥ"


def encode_sequence(sequence: str) -> str:
    """The dataset currently trains the InstructProtein-based pilot
    (khairi/Eshmun-Thinking-Pilot), whose tokenizer has no protein-specific
    vocabulary and otherwise BPE-merges multiple residues into one token --
    a `Ƥ`-per-residue join disambiguates amino-acid letters from English text,
    matching the pilot's own build_pairs.py / build_generation_pairs.py
    convention (`Ƥ.join(sequence)`, verified against
    src/eshmun/trainer/grpo/reward_functions/protein.py's PROTEIN_PREFIX_TOKEN).
    Revisit once training moves to Eshmun's own tokenizer, which has dedicated
    protein BPE tokens and needs no such marker."""
    return f"<protein>{PROTEIN_PREFIX_TOKEN.join(sequence)}</protein>"


# Generation direction (property -> sequence): the conditioning property is
# the *input*, not the answer, so unlike RESPONSE_TEMPLATES there is no
# leakage risk in stating it directly -- no DeepSeek humanization pass needed,
# a handful of hand-written variants is enough for phrasing diversity.
GENERATION_INSTRUCTION_TEMPLATES = {
    "member_of": [
        "Design a protein sequence that belongs to the {joined}.",
        "Generate a protein sequence classified in the {joined}.",
        "Give me a protein sequence belonging to the {joined}.",
    ],
    "has_function": [
        "Design a protein sequence with the molecular function of {joined}.",
        "Generate a protein sequence that performs {joined}.",
        "Give me a protein sequence whose molecular function is {joined}.",
    ],
    "involved_in": [
        "Design a protein sequence involved in {joined}.",
        "Generate a protein sequence that participates in {joined}.",
        "Give me a protein sequence involved in the biological process of {joined}.",
    ],
    "located_in": [
        "Design a protein sequence located in the {joined}.",
        "Generate a protein sequence found in the {joined}.",
        "Give me a protein sequence localized to the {joined}.",
    ],
    "catalyzes": [
        "Design a protein sequence that catalyzes reactions classified under EC {joined}.",
        "Generate a protein sequence with catalytic activity EC {joined}.",
        "Give me a protein sequence classified under EC {joined}.",
    ],
    "scop_fold": [
        "Design a protein sequence that adopts the {joined} fold.",
        "Generate a protein sequence with the {joined} fold.",
        "Give me a protein sequence that folds into the {joined} structure.",
    ],
    "scop_superfamily": [
        "Design a protein sequence belonging to the {joined} superfamily.",
        "Generate a protein sequence classified in the {joined} superfamily.",
        "Give me a protein sequence from the {joined} superfamily.",
    ],
    "scop_family": [
        "Design a protein sequence belonging to the {joined} family, per SCOP classification.",
        "Generate a protein sequence classified in the {joined} family (SCOP).",
        "Give me a protein sequence from the {joined} family, based on structural classification.",
    ],
}


def build_generation_instruction(
    target_relation: str, values: list[str], rng: random.Random
) -> str:
    if target_relation not in GENERATION_INSTRUCTION_TEMPLATES:
        raise ValueError(f"unsupported target relation: {target_relation}")
    if not values:
        raise ValueError("values must be non-empty")

    template = rng.choice(GENERATION_INSTRUCTION_TEMPLATES[target_relation])
    return template.format(joined=join_values(values))


def build_generation_instruction_from_pool(
    values: list[str], pool: list[str], rng: random.Random
) -> str:
    """Same idea as build_generation_instruction, but samples from an
    arbitrary externally-supplied pool (e.g. DeepSeek-generated, for more
    phrasing diversity than the ~3 hand-written GENERATION_INSTRUCTION_TEMPLATES
    variants) instead of the fixed per-relation template dict. No leakage
    risk either way -- the property is the given input, not a hidden answer
    -- so target_relation isn't needed here, just the pool to sample from."""
    if not values:
        raise ValueError("values must be non-empty")
    if not pool:
        raise ValueError("pool must be non-empty")

    template = rng.choice(pool)
    return template.format(joined=join_values(values))
