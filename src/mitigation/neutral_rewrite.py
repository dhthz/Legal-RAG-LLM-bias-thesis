from src.mitigation.base import MitigationArm
from toolkit.fixes import neutral_rewrite as _rewrite
from toolkit.fixes.neutral_rewrite import GENDERED

# The rewrite functions live in the toolkit (toolkit/fixes/neutral_rewrite.py), so the thesis and the plugin run the
# same validated code (rewrite fidelity 581/581, logs/mitigation/rewrite_fidelity.json). The thesis runs them without
# the toolkit's gendered person nouns (nouns=False), as validated.

__all__ = ["GENDERED", "BlindQuery", "neutralize", "neutralize_many", "swap_gender", "swap_gender_many"]


def neutralize(text):
    return _rewrite.neutralize(text, nouns=False)


def swap_gender(text):
    return _rewrite.swap_gender(text, nouns=False)


def neutralize_many(texts, batch_size=64, n_process=1):
    return _rewrite.neutralize_many(texts, batch_size, n_process, nouns=False)


def swap_gender_many(texts, batch_size=64, n_process=1):
    return _rewrite.swap_gender_many(texts, batch_size, n_process, nouns=False)


# Arm 1: blindness on the query (Garg et al. 2019), realised as the gender-neutral rewrite
class BlindQuery(MitigationArm):
    name = "blind_query"

    def rewrite_query(self, query: str) -> str:
        return neutralize(query)
