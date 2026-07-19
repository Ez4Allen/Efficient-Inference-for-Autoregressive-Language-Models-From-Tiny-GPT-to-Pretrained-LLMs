
from __future__ import annotations

import unicodedata


def normalize_entity_name(value: str) -> str:
    """
    Convert an entity name into a normalized lookup key.

    Examples:
        "Night's Edge"  -> "nightsedge"
        "NIGHT’S EDGE"  -> "nightsedge"
        "Night Edge"    -> "nightedge"
        " Mechanical Eye " -> "mechanicaleye"
        "机械魔眼"          -> "机械魔眼"

    This function only normalizes formatting.
    It does not perform fuzzy matching.
    """

    if not isinstance(value, str):
        raise TypeError(
            "Entity name must be a string, "
            f"but received {type(value).__name__}."
        )

    # Normalize Unicode variants.
    # For example, full-width English letters and punctuation
    # are converted into their standard forms where possible.
    value = unicodedata.normalize("NFKC", value)

    # Case-insensitive normalization.
    value = value.casefold()

    # Normalize common apostrophe variants.
    value = value.replace("’", "'")
    value = value.replace("‘", "'")
    value = value.replace("`", "'")

    # Remove apostrophes.
    # This makes "Night's Edge" and "Nights Edge" equivalent.
    value = value.replace("'", "")

    # Keep only letters and numbers.
    # Spaces, hyphens, punctuation, and underscores are removed.
    normalized_characters = []

    for character in value:
        if character.isalnum():
            normalized_characters.append(character)

    return "".join(normalized_characters)
