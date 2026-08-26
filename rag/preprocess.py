"""Clean the PvME guide markup so the text is readable to the embedding model."""

import json
import re

from rag import enrage
from rag.emoji_map import ALIAS_CODES, EMOJI_MAP

# Embeds that are navigation rather than content
SKIP_EMBED_TITLES = ["table of contents"]


def expand_emoji(text, mode="both"):
    """Turn <:bloat:1159433682403201044> into Bloat.

    mode controls how ALIAS_CODES entries render:
      full  -> "Fractured Staff of Armadyl"
      alias -> "fsoa"
      both  -> "Fractured Staff of Armadyl (fsoa)"

    Codes outside ALIAS_CODES always use their name, since the shortcode
    already is the name (bloat -> Bloat).
    """
    if mode not in ("full", "alias", "both"):
        raise ValueError(f"unknown emoji mode: {mode!r}")

    found = set(re.findall(r"<:([a-zA-Z0-9_]+):\d+>", text))
    missing = found - set(EMOJI_MAP)
    if missing:
        print(f"   !! No name for these emoji: {sorted(missing)}")

    for code, name in EMOJI_MAP.items():
        if code in ALIAS_CODES and mode == "alias":
            replacement = code
        elif code in ALIAS_CODES and mode == "both":
            replacement = f"{name} ({code})"
        else:
            replacement = name
        text = re.sub(rf"<:{code}:\d+>", replacement, text)

    # Anything left over: at least drop the useless numeric ID
    text = re.sub(r"<:([a-zA-Z0-9_]+):\d+>", r"\1", text)
    return collapse_expansion_repeats(text)


def collapse_expansion_repeats(text):
    """Fix 'Powder of Penance Powder of Penance'.

    The guides often write an item's name in prose AND put its emoji beside it,
    so expansion doubles the phrase. Only known mapped names are collapsed -
    a blanket repeated-phrase rule would corrupt rotations that legitimately
    cast the same ability twice.
    """
    for name in set(EMOJI_MAP.values()):
        doubled = f"{name} {name}"
        while doubled in text:
            text = text.replace(doubled, name)
    return text


def _tidy_link(value):
    """Turn [Link](https://youtu.be/abc) into a bare, quotable URL."""
    return re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", r"\2", value)


def _embed_to_text(data):
    """Turn one parsed embed into plain markdown lines."""
    embed = data.get("embed", data)
    title = (embed.get("title") or "").strip("_ ")

    if title.lower() in SKIP_EMBED_TITLES:
        return ""

    parts = []
    if title:
        parts.append(f"### {title}")

    description = (embed.get("description") or "").strip()
    if description:
        parts.append(description)

    for field in embed.get("fields", []):
        name = (field.get("name") or "").strip("_ :;")
        value = _tidy_link((field.get("value") or "").strip())

        # Label videos explicitly - the raw guide never says "video" anywhere,
        # so without this nobody can retrieve them by asking for one.
        urls = re.findall(r"https?://(?:www\.)?(?:youtu\.be|youtube\.com)\S+", value)
        if urls:
            label = name if name.lower() not in ("example kills", "") else "Arch-Glacor"
            for url in urls:
                # Guides wrap some links as <https://...>, leaving a trailing >
                parts.append(f"- Example kill video, {label}: {url.rstrip('>).,')}")
        elif name:
            parts.append(f"**{name}**\n{value}")
        else:
            parts.append(value)

    return "\n".join(parts)


def _parse_embed_above(lines, marker_index):
    """Find the JSON block that ends just above `.embed:json`.

    Walks upwards adding one line at a time until the text parses as JSON.
    Returns (parsed_data, start_line) or (None, None).
    """
    for start in range(marker_index - 1, -1, -1):
        block = "\n".join(lines[start:marker_index])
        try:
            return json.loads(block), start
        except json.JSONDecodeError:
            continue
    return None, None


def flatten_embeds(text):
    """Replace every JSON embed block with readable markdown."""
    lines = text.split("\n")

    while True:
        # Re-scan each pass: replacing lines changes the numbering
        marker = None
        for i, line in enumerate(lines):
            if line.strip() == ".embed:json":
                marker = i
                break

        if marker is None:
            break

        data, start = _parse_embed_above(lines, marker)
        if data is None:
            print(f"   !! Could not parse the embed ending on line {marker + 1}")
            del lines[marker]
            continue

        lines[start : marker + 1] = _embed_to_text(data).split("\n")

    return "\n".join(lines)


def strip_markup(text):
    """Remove PvME control lines and Discord references."""
    lines = []
    for line in text.split("\n"):
        if line.startswith("."):
            continue
        line = re.sub(r"<[#@!]+\d+>", "", line)
        lines.append(line)
    return "\n".join(lines)


def normalize_bullets(text):
    for bullet in ["⬥", "⬩", "•"]:
        text = text.replace(bullet, "-")
    return text


def tidy_blank_lines(text):
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def preprocess(text, mode="both", enrage_bands=False):
    """Run the cleaning steps. Order matters - see the notes below.

    Kept free of rag.config on purpose: pure functions with the options passed
    in are far easier to test than ones that reach for global settings.
    """
    text = expand_emoji(text, mode)  # before embeds: emoji live in the JSON
    text = flatten_embeds(text)    # before strip_markup: it deletes .embed:json
    text = strip_markup(text)
    text = normalize_bullets(text)
    text = tidy_blank_lines(text)
    if enrage_bands:
        # Last: needs final line structure, and appends rather than substitutes
        text = enrage.annotate(text)
    return text
