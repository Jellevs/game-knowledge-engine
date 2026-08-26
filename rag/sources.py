"""Working out which guide a chunk came from.

Filenames follow the pattern <boss>_high_enrage_<style>.txt, e.g.
    arch_glacor_high_enrage_melee_magic.txt -> boss=arch_glacor, style=melee_magic

Style matters because the three Arch-Glacor guides answer the same questions
differently. 'What is the arms rotation?' has three correct answers, and both
hybrid guides even have a section literally called 'Melee Phase'. Without this
metadata, retrieval cannot be scored and the model cannot be filtered.
"""

from pathlib import Path

SPLIT_ON = "_high_enrage_"


def style_of(path: Path) -> str:
    """melee_magic, melee_ranged, necromancy, ... Falls back to the whole stem."""
    stem = Path(path).stem
    return stem.split(SPLIT_ON, 1)[1] if SPLIT_ON in stem else stem


def boss_of(path: Path) -> str:
    """arch_glacor, ... Falls back to the whole stem."""
    stem = Path(path).stem
    return stem.split(SPLIT_ON, 1)[0] if SPLIT_ON in stem else stem


def guide_paths(guide_dir: Path) -> list[Path]:
    """Every guide file, in a stable order."""
    return sorted(Path(guide_dir).glob("*.txt"))
