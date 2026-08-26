"""Preprocessing is checked against the real guides, not toy strings, so these
keep working as guides are added."""

import re

import pytest

from rag.config import settings
from rag.emoji_map import EMOJI_MAP
from rag.preprocess import expand_emoji, preprocess
from rag.sources import guide_paths

GUIDES = guide_paths(settings.guide_dir)
CLEANED = {p.name: (p.read_text(encoding="utf-8"), preprocess(p.read_text(encoding="utf-8")))
           for p in GUIDES}

ALL_GUIDES = pytest.mark.parametrize("name", sorted(CLEANED))


def test_guides_exist():
    assert len(GUIDES) >= 3


@ALL_GUIDES
def test_no_emoji_markup_survives(name):
    _, clean = CLEANED[name]
    assert re.search(r"<:[a-zA-Z0-9_]+:\d+>", clean) is None


@ALL_GUIDES
def test_every_emoji_has_a_name(name):
    """Fails loudly when a new guide introduces an unmapped shortcode."""
    raw, _ = CLEANED[name]
    found = set(re.findall(r"<:([a-zA-Z0-9_]+):\d+>", raw))
    assert found - set(EMOJI_MAP) == set()


@ALL_GUIDES
def test_no_pvme_directives_survive(name):
    _, clean = CLEANED[name]
    for directive in [".tag:", ".embed:json", ".img:", ".pin:"]:
        assert directive not in clean


@ALL_GUIDES
def test_no_raw_json_survives(name):
    _, clean = CLEANED[name]
    assert '"embed"' not in clean
    assert '"inline"' not in clean


@ALL_GUIDES
def test_meaningful_shrinkage(name):
    """Roughly 30-40% of each guide is markup. Guard against doing nothing."""
    raw, clean = CLEANED[name]
    assert len(clean) < 0.80 * len(raw)


@ALL_GUIDES
def test_headers_preserved_for_the_splitter(name):
    _, clean = CLEANED[name]
    assert clean.startswith("# Arch-Glacor")


@ALL_GUIDES
def test_video_urls_intact(name):
    """Guides use both youtu.be/ and youtube.com/watch?v= forms."""
    _, clean = CLEANED[name]
    assert re.search(r"https?://(?:www\.)?(?:youtu\.be|youtube\.com)/\S+", clean)
    assert "Example kill video" in clean


@ALL_GUIDES
def test_no_doubled_ability_names(name):
    """The guides write a name in prose AND add its emoji, which doubles the
    phrase on expansion. Regression guard for collapse_expansion_repeats."""
    _, clean = CLEANED[name]
    for phrase in ["Powder of Penance Powder of Penance",
                   "Master Magic Cape Master Magic Cape",
                   "Necromancy Necromancy"]:
        assert phrase not in clean


def test_rotation_reads_as_ability_names():
    _, clean = CLEANED["arch_glacor_high_enrage_necromancy.txt"]
    for ability in ["Invoke Death", "Split Soul", "Death Skulls"]:
        assert ability in clean


def test_hybrid_abilities_expand():
    _, clean = CLEANED["arch_glacor_high_enrage_melee_ranged.txt"]
    for ability in ["Greater Death's Swiftness", "Bow of the Last Guardian"]:
        assert ability in clean


def test_buffs_checklist_survived_the_embed():
    _, clean = CLEANED["arch_glacor_high_enrage_necromancy.txt"]
    assert "Kwuarm incense at 4 stacks" in clean
    assert "Limitless Sigil" in clean


def test_expand_emoji_falls_back_to_shortcode():
    assert expand_emoji("<:notarealthing:123>") == "notarealthing"


def test_emoji_mode_both_keeps_both_forms():
    """Players ask 'when do I use fsoa?', so the abbreviation must survive."""
    out = expand_emoji("<:fsoa:1>", mode="both")
    assert "Fractured Staff of Armadyl" in out
    assert "fsoa" in out


def test_emoji_mode_full():
    assert expand_emoji("<:fsoa:1>", mode="full") == "Fractured Staff of Armadyl"


def test_emoji_mode_alias():
    assert expand_emoji("<:fsoa:1>", mode="alias") == "fsoa"


def test_non_alias_codes_are_never_annotated():
    """Only genuine jargon is affected; the shortcode already IS the name."""
    for mode in ("full", "alias", "both"):
        assert expand_emoji("<:bloat:1>", mode=mode) == "Bloat"


def test_unknown_emoji_mode_raises():
    with pytest.raises(ValueError):
        expand_emoji("<:fsoa:1>", mode="nonsense")


def test_alias_codes_all_have_mappings():
    from rag.emoji_map import ALIAS_CODES
    assert ALIAS_CODES - set(EMOJI_MAP) == set()


def test_preprocess_is_idempotent():
    raw = "# Boss <:bloat:1>\n.tag:intro\n⬥ Use <:soulsap:2>\n"
    once = preprocess(raw)
    assert preprocess(once) == once
