import re
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional

import spacy
import torch
from spacy.tokens import Doc, Token
from spacy.util import compile_infix_regex
from transformers import AutoModelForCausalLM, AutoTokenizer

# Gender rewriting shared by the mitigation arms:
#   neutralize(): gender-neutral rewrite with singular "they", the rule-based algorithm of Sun et al. 2021,
#                 "They, Them, Theirs: Rewriting with Gender-Neutral English" (arXiv:2102.06788): pronouns are mapped
#                 from POS/morphology/dependency information and verbs re-agreed with "they". Their seq2seq model
#                 (trained on the rule output) is not used.
#   swap_gender(): two-sided counterfactual swap (CDA; Zhao et al. 2018, Garg et al. 2019) for the PCF and fine-tune arms
#                  and for building he/she twin questions.
# "her" is the one form the rules cannot settle (their/them, his/him): as in Sun et al., both candidate sentences are
# scored with a language model (GPT-2) and the more likely one is kept.
# Gendered nouns follow the hand-written audit variants: husband/wife <-> spouse, titles Mr/Mrs/Ms.
# With nouns=True (the toolkit default) gendered person nouns are rewritten too (nephew <-> niece, or "relative" in the
# neutral rewrite). The thesis ran with nouns=False: its legal questions name people only by pronoun, spouse and title.

SPACY_MODEL = "en_core_web_sm"
LM_NAME = "gpt2"
LM_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pinned so the her-resolution is reproducible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENDERED = re.compile(r"\b(he|she|him|his|her|hers|himself|herself|husbands?|wi(fe|ves)|mrs?|ms)\b\.?", re.IGNORECASE)

TITLES = {"mr", "mrs", "ms", "mr.", "mrs.", "ms."}
NEUTRAL_WORDS = {"he": "they", "she": "they", "him": "them", "hers": "theirs",
                 "himself": "themselves", "herself": "themselves",
                 "husband": "spouse", "wife": "spouse", "husbands": "spouses", "wives": "spouses"}
SWAP_WORDS = {"he": "she", "she": "he", "him": "her", "hers": "his", "himself": "herself", "herself": "himself",
              "husband": "wife", "wife": "husband", "husbands": "wives", "wives": "husbands",
              "mr": "mrs", "mrs": "mr", "ms": "mr", "mr.": "mrs.", "mrs.": "mr.", "ms.": "mr."}
# (he form, she form, neutral form) and their plurals
PERSON_NOUNS = [("father", "mother", "parent"), ("dad", "mom", "parent"), ("son", "daughter", "child"),
                ("brother", "sister", "sibling"), ("uncle", "aunt", "relative"), ("nephew", "niece", "relative"),
                ("boy", "girl", "child"), ("man", "woman", "person"), ("boyfriend", "girlfriend", "partner"),
                ("grandfather", "grandmother", "grandparent"), ("grandpa", "grandma", "grandparent"),
                ("grandson", "granddaughter", "grandchild"), ("stepfather", "stepmother", "step-parent"),
                ("stepson", "stepdaughter", "stepchild"), ("groom", "bride", "newlywed"), ("widower", "widow", "widowed spouse"),
                ("fiance", "fiancee", "partner"), ("fiancé", "fiancée", "partner"), ("gentleman", "lady", "person")]
IRREGULAR_PLURALS = {"man": "men", "woman": "women", "gentleman": "gentlemen", "lady": "ladies", "child": "children",
                     "person": "people", "widowed spouse": "widowed spouses"}
PERSON_SWAP, PERSON_NEUTRAL = {"mum": "dad", "mums": "dads"}, {"mum": "parent", "mums": "parents"}
for he_form, she_form, neutral_form in PERSON_NOUNS:
    for a, b in ((he_form, she_form), (she_form, he_form)):
        PERSON_SWAP[a] = b
        PERSON_SWAP[IRREGULAR_PLURALS.get(a, a + "s")] = IRREGULAR_PLURALS.get(b, b + "s")
        PERSON_NEUTRAL[a] = neutral_form
        PERSON_NEUTRAL[IRREGULAR_PLURALS.get(a, a + "s")] = IRREGULAR_PLURALS.get(neutral_form, neutral_form + "s")
PERSON = re.compile(r"\b(" + "|".join(sorted(PERSON_SWAP, key=len, reverse=True)) + r")\b", re.IGNORECASE)
PLURAL_VERBS = {"is": "are", "was": "were", "has": "have", "does": "do"}
SUBJECT_DEPS = {"nsubj", "nsubjpass"}


@lru_cache(maxsize=1)
def _nlp():
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    # The legal texts sometimes drop the space around brackets ("proceedings)she", "her[1]"); split there so the pronoun is a token
    infixes = list(nlp.Defaults.infixes) + [r"(?<=[)\]])(?=[A-Za-z])", r"(?<=[A-Za-z])(?=[(\[])"]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    return nlp


@lru_cache(maxsize=1)
def _lm():
    tokenizer = AutoTokenizer.from_pretrained(LM_NAME, revision=LM_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, AutoModelForCausalLM.from_pretrained(LM_NAME, revision=LM_REVISION).to(DEVICE).eval()


@torch.no_grad()
def _log_likelihood(texts: List[str], batch_size: int = 32) -> List[float]:
    tokenizer, model = _lm()
    scores = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(DEVICE)
        log_probs = torch.log_softmax(model(**enc).logits[:, :-1].float(), dim=-1)
        token_log_probs = log_probs.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores.extend((token_log_probs * enc.attention_mask[:, 1:]).sum(dim=1).tolist())
    return scores


def _match_case(word: str, like: str) -> str:
    if like.isupper() and len(like) > 1:
        return word.upper()
    if like[0].isupper():
        return word[0].upper() + word[1:]
    return word


def _is_possessive_determiner(token: Token) -> bool:
    return token.tag_ == "PRP$" and token.dep_ != "attr"


# "his" as a determiner ("his house") vs a pronoun ("the house is his"); "her" is resolved by _resolve_her
def _neutral_pronoun(token: Token) -> Optional[str]:
    if token.lower_ == "his":
        return "their" if _is_possessive_determiner(token) else "theirs"
    return NEUTRAL_WORDS.get(token.lower_)


def _swapped_pronoun(token: Token) -> Optional[str]:
    if token.lower_ == "his":
        return "her" if _is_possessive_determiner(token) else "hers"
    return SWAP_WORDS.get(token.lower_)


# A person noun used as a noun ("my nephew"), not as a modifier ("sister company") or a verb ("man the desk")
def _person_noun(token: Token, words: Dict[str, str]) -> Optional[str]:
    if token.pos_ in ("NOUN", "PROPN") and token.dep_ != "compound":
        return words.get(token.lower_)
    return None


# "an uncle" -> "a relative": the article before a replaced noun follows the new word
def _replace_noun(doc: Doc, replacements: Dict[int, str], token: Token, word: str) -> None:
    replacements[token.i] = word
    if token.i and doc[token.i - 1].lower_ in ("a", "an"):
        article = "an" if word[0].lower() in "aeiou" else "a"
        if article != doc[token.i - 1].lower_:
            replacements[token.i - 1] = article


def _gendered(text: str, nouns: bool) -> bool:
    return bool(GENDERED.search(text) or (nouns and PERSON.search(text)))


def _sentence_with(token: Token, word: str) -> str:
    return "".join(_match_case(word, t.text) + t.whitespace_ if t.i == token.i else t.text_with_ws for t in token.sent)


# options = (possessive form, object form), e.g. ("their", "them"); the likelier sentence under the LM wins
def _resolve_her(doc: Doc, replacements: Dict[int, str], options) -> None:
    hers = [t for t in doc if t.lower_ == "her"]
    if not hers:
        return
    scores = _log_likelihood([_sentence_with(t, word) for t in hers for word in options])
    for k, token in enumerate(hers):
        replacements[token.i] = options[0] if scores[2 * k] >= scores[2 * k + 1] else options[1]


# Finite verbs that agree with a subject: the head verb or its auxiliaries, plus verbs conjoined to it
# that have no subject of their own ("he lives in Sofia and works as ...")
def _agreeing_verbs(subject: Token) -> Iterator[Token]:
    stack = [subject.head]
    while stack:
        verb = stack.pop()
        yield verb
        for child in verb.children:
            if child.dep_ in ("aux", "auxpass"):
                yield child
            elif child.dep_ == "conj" and child.pos_ in ("VERB", "AUX") \
                    and not any(c.dep_ in SUBJECT_DEPS for c in child.children):
                stack.append(child)


def _plural_verb(verb: Token) -> Optional[str]:
    lower = verb.lower_
    if lower in PLURAL_VERBS and verb.tag_ in ("VBZ", "VBD"):
        return PLURAL_VERBS[lower]
    if verb.tag_ == "VBZ":
        return verb.lemma_
    return None


def _render(doc: Doc, replacements: Dict[int, str]) -> str:
    out = []
    for token in doc:
        if token.i not in replacements:
            out.append(token.text_with_ws)
        elif replacements[token.i]:
            out.append(_match_case(replacements[token.i], token.text) + token.whitespace_)
    return "".join(out)


def _neutralize_doc(doc: Doc, nouns: bool = True) -> str:
    replacements = {}
    for token in doc:
        if nouns and (word := _person_noun(token, PERSON_NEUTRAL)):
            _replace_noun(doc, replacements, token, word)
            continue
        if token.lower_ in TITLES:
            replacements[token.i] = ""
            continue
        neutral = _neutral_pronoun(token)
        if neutral is None:
            continue
        replacements[token.i] = neutral
        if token.lower_ in ("he", "she") and token.dep_ in SUBJECT_DEPS:
            for verb in _agreeing_verbs(token):
                plural = _plural_verb(verb)
                if plural and verb.i not in replacements:
                    replacements[verb.i] = plural
    _resolve_her(doc, replacements, ("their", "them"))
    return _render(doc, replacements)


def _swap_doc(doc: Doc, nouns: bool = True) -> str:
    replacements = {t.i: s for t in doc if (s := _swapped_pronoun(t)) is not None}
    for token in doc if nouns else ():
        if word := _person_noun(token, PERSON_SWAP):
            _replace_noun(doc, replacements, token, word)
    _resolve_her(doc, replacements, ("his", "him"))
    return _render(doc, replacements)


def _rewrite_many(texts: Iterable[str], rewrite_doc, batch_size: int, n_process: int, nouns: bool) -> List[str]:
    texts = list(texts)
    out = list(texts)
    todo = [i for i, t in enumerate(texts) if _gendered(t, nouns)]
    docs = _nlp().pipe((texts[i] for i in todo), batch_size=batch_size, n_process=n_process)
    for i, doc in zip(todo, docs):
        out[i] = rewrite_doc(doc, nouns)
    return out


def neutralize(text: str, nouns: bool = True) -> str:
    return _neutralize_doc(_nlp()(text), nouns) if _gendered(text, nouns) else text


def swap_gender(text: str, nouns: bool = True) -> str:
    return _swap_doc(_nlp()(text), nouns) if _gendered(text, nouns) else text


def neutralize_many(texts: Iterable[str], batch_size: int = 64, n_process: int = 1, nouns: bool = True) -> List[str]:
    return _rewrite_many(texts, _neutralize_doc, batch_size, n_process, nouns)


def swap_gender_many(texts: Iterable[str], batch_size: int = 64, n_process: int = 1, nouns: bool = True) -> List[str]:
    return _rewrite_many(texts, _swap_doc, batch_size, n_process, nouns)

