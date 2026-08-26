"""Chunking and metadata checks. These run the real splitter over the real
guides, so they catch corpus problems as new guides are added."""

import pytest

from rag.ingest import load_documents
from rag.sources import boss_of, style_of

CHUNKS = load_documents()


def test_chunks_from_every_guide():
    styles = {c.metadata["style"] for c in CHUNKS}
    assert styles == {"necromancy", "melee_magic", "melee_ranged"}


def test_every_chunk_is_attributable():
    """Without source metadata, relevance labels cannot tell the guides apart -
    both hybrids have a section called 'Melee Phase'."""
    for chunk in CHUNKS:
        assert chunk.metadata.get("source")
        assert chunk.metadata.get("style")
        assert chunk.metadata.get("boss") == "arch_glacor"


def test_chunks_carry_header_metadata():
    assert any(c.metadata.get("Header 3") for c in CHUNKS)


def test_no_chunk_is_mostly_empty():
    assert all(len(c.page_content.strip()) > 20 for c in CHUNKS)


def test_no_chunk_exceeds_configured_size_by_much():
    from rag.config import settings
    assert all(len(c.page_content) <= settings.chunk_size * 1.5 for c in CHUNKS)


@pytest.mark.parametrize(
    "video_id",
    ["B-5v0Qb4XUE", "ETA6vtiuOvA"],
)
def test_video_url_intact_within_a_single_chunk(video_id):
    """A URL split across two chunks yields a broken link, which is worse than
    no link at all."""
    assert any(video_id in c.page_content for c in CHUNKS)


def test_style_and_boss_parsing():
    assert style_of("data/arch_glacor_high_enrage_melee_magic.txt") == "melee_magic"
    assert boss_of("data/arch_glacor_high_enrage_melee_magic.txt") == "arch_glacor"
    assert style_of("something_else.txt") == "something_else"
