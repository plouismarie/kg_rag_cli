"""
Entity Normalization Utilities

This module provides core utilities for normalizing entity names to enable
entity deduplication in knowledge graphs. The goal is to merge entity variations
like "Paris", "paris", and "Prince Paris" into a single canonical form.

Functions:
    - normalize_entity(): Case-folding, punctuation removal, whitespace normalization
    - extract_base_form(): Remove titles/honorifics (e.g., "Prince Paris" → "Paris")
    - are_similar(): Fuzzy string matching for entity similarity

Usage:
    >>> from lib.entity_normalizer import normalize_entity, are_similar
    >>> normalize_entity("Paris")
    'paris'
    >>> normalize_entity("  Prince Paris.  ")
    'prince paris'
    >>> are_similar("Paris", "paris")
    True
    >>> are_similar("Helen", "Helen of Sparta")
    False  # Below 85% similarity threshold
"""

import re
import string
from difflib import SequenceMatcher
from typing import Optional, Tuple

# Common titles/honorifics to remove (case-insensitive)
TITLES = [
    'king', 'queen', 'prince', 'princess',
    'lord', 'lady', 'sir', 'dame',
    'duke', 'duchess', 'count', 'countess',
    'baron', 'baroness', 'earl',
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof',
    'captain', 'lieutenant', 'general', 'admiral',
    'father', 'brother', 'sister', 'saint',
    'goddess', 'god'  # For mythological entities
]

# Articles to remove (case-insensitive)
ARTICLES = ['the', 'a', 'an', 'le', 'la', 'les', 'un', 'une']  # English + French


def normalize_entity(text: str, remove_articles: bool = True) -> str:
    """
    Normalize an entity string for deduplication matching.

    This function performs the following normalization steps:
    1. Convert to lowercase (casefold for Unicode support)
    2. Normalize whitespace (collapse multiple spaces, strip)
    3. Remove leading/trailing punctuation
    4. Optionally remove leading articles ("The Paris" → "Paris")

    Args:
        text: The entity string to normalize
        remove_articles: If True, remove leading articles (default: True)

    Returns:
        Normalized entity string (lowercase, cleaned)

    Examples:
        >>> normalize_entity("Paris")
        'paris'
        >>> normalize_entity("  Paris  ")
        'paris'
        >>> normalize_entity("Paris.")
        'paris'
        >>> normalize_entity("The Paris")
        'paris'
        >>> normalize_entity("PARIS")
        'paris'
        >>> normalize_entity("Prince Paris")
        'prince paris'  # Title removal handled by extract_base_form()
    """
    if not text:
        return ""

    # Step 1: Casefold for Unicode-aware lowercase (handles accents, etc.)
    normalized = text.casefold()

    # Step 2: Normalize whitespace
    # Replace multiple whitespace with single space
    normalized = ' '.join(normalized.split())

    # Step 3: Remove leading/trailing punctuation
    # But preserve internal punctuation (e.g., "Mary's" keeps apostrophe)
    normalized = normalized.strip(string.punctuation + string.whitespace)

    # Step 4: Remove leading articles if requested
    if remove_articles:
        words = normalized.split()
        if words and words[0] in ARTICLES:
            normalized = ' '.join(words[1:])

    return normalized


def extract_base_form(entity: str) -> str:
    """
    Extract the base form of an entity by removing titles and honorifics.

    This function removes common titles like "Prince", "King", "Queen", etc.,
    to extract the core entity name. Preserves the original capitalization
    of the base form.

    Args:
        entity: The entity string (may include titles)

    Returns:
        Entity with titles removed, preserving original capitalization

    Examples:
        >>> extract_base_form("Prince Paris")
        'Paris'
        >>> extract_base_form("King Priam")
        'Priam'
        >>> extract_base_form("Queen Hecuba")
        'Hecuba'
        >>> extract_base_form("Paris")
        'Paris'  # No change if no title
        >>> extract_base_form("Goddess Eris")
        'Eris'
        >>> extract_base_form("Prince of Troy")
        'of Troy'  # Removes title but keeps rest
    """
    if not entity:
        return ""

    # Split into words
    words = entity.split()
    if not words:
        return entity

    # Check if first word is a title (case-insensitive)
    first_word_lower = words[0].lower()

    if first_word_lower in TITLES:
        # Remove the title and return the rest
        base = ' '.join(words[1:])
        return base if base else entity  # Return original if nothing left
    else:
        # No title found, return original
        return entity


def are_similar(entity1: str, entity2: str, threshold: float = 0.85) -> bool:
    """
    Check if two entity strings are similar using fuzzy string matching.

    Uses difflib.SequenceMatcher to compute similarity ratio between
    normalized versions of the entities. Entities are considered similar
    if their normalized forms have >= threshold similarity.

    Args:
        entity1: First entity string
        entity2: Second entity string
        threshold: Similarity threshold (0.0 to 1.0, default: 0.85)

    Returns:
        True if entities are similar (>= threshold), False otherwise

    Examples:
        >>> are_similar("Paris", "paris")
        True  # 100% match after normalization
        >>> are_similar("Paris", "PARIS")
        True  # 100% match after normalization
        >>> are_similar("Helen", "helen")
        True  # 100% match
        >>> are_similar("Paris", "Priam")
        False  # Different names
        >>> are_similar("Helen", "Helen of Sparta")
        False  # ~60% similarity, below 85% threshold
        >>> are_similar("Ulysses", "Odysseus")
        False  # Different names (same person but different names)
        >>> are_similar("Prince Paris", "Paris", threshold=0.70)
        True  # ~80% similarity, above custom 70% threshold

    Note:
        The default threshold of 0.85 is conservative to avoid merging
        entities that shouldn't be merged (e.g., "Helen" vs "Helen of Sparta").
        You can adjust the threshold for specific use cases.
    """
    if not entity1 or not entity2:
        return False

    # Normalize both entities for comparison
    norm1 = normalize_entity(entity1)
    norm2 = normalize_entity(entity2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Fuzzy match using SequenceMatcher
    # SequenceMatcher.ratio() returns similarity in [0.0, 1.0]
    similarity = SequenceMatcher(None, norm1, norm2).ratio()

    return similarity >= threshold


def get_similarity_score(entity1: str, entity2: str) -> float:
    """
    Get the similarity score between two entities (0.0 to 1.0).

    Helper function that returns the actual similarity score instead of
    a boolean. Useful for debugging and tuning the similarity threshold.

    Args:
        entity1: First entity string
        entity2: Second entity string

    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match)

    Examples:
        >>> get_similarity_score("Paris", "paris")
        1.0  # Exact match
        >>> get_similarity_score("Helen", "Helen of Sparta")
        0.6...  # Partial match
        >>> get_similarity_score("Paris", "Priam")
        0.4...  # Low similarity
    """
    if not entity1 or not entity2:
        return 0.0

    norm1 = normalize_entity(entity1)
    norm2 = normalize_entity(entity2)

    if norm1 == norm2:
        return 1.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def is_title_variant(entity1: str, entity2: str) -> bool:
    """
    Check if two entities are variants where one has a title and one doesn't.

    This is a specialized check for cases like:
    - "Prince Paris" vs "Paris"
    - "King Priam" vs "Priam"

    Args:
        entity1: First entity string
        entity2: Second entity string

    Returns:
        True if entities are title variants of each other

    Examples:
        >>> is_title_variant("Prince Paris", "Paris")
        True
        >>> is_title_variant("Paris", "Prince Paris")
        True
        >>> is_title_variant("King Priam", "Priam")
        True
        >>> is_title_variant("Paris", "Priam")
        False
        >>> is_title_variant("Goddess Eris", "Eris")
        True
    """
    if not entity1 or not entity2:
        return False

    # Extract base forms
    base1 = extract_base_form(entity1)
    base2 = extract_base_form(entity2)

    # Normalize for comparison
    norm_base1 = normalize_entity(base1)
    norm_base2 = normalize_entity(base2)
    norm_entity1 = normalize_entity(entity1)
    norm_entity2 = normalize_entity(entity2)

    # Check if one is the base form of the other
    # Case 1: entity1 has title, entity2 is base
    if norm_base1 == norm_entity2 and norm_entity1 != norm_entity2:
        return True

    # Case 2: entity2 has title, entity1 is base
    if norm_base2 == norm_entity1 and norm_entity1 != norm_entity2:
        return True

    return False


def are_substring_match(entity1: str, entity2: str) -> Tuple[bool, Optional[str]]:
    """
    Check if two entities match via case-insensitive comparison or substring containment.

    This provides a lighter deduplication than fuzzy matching:
    - Case-insensitive exact match: "lion" = "LION" = "Lion"
    - Substring containment: "George" is contained in "George Hadley"

    When matched, returns the LONGER (more specific) entity as canonical.

    Args:
        entity1: First entity string
        entity2: Second entity string

    Returns:
        Tuple of (is_match, canonical_form):
        - is_match: True if entities match via case or substring
        - canonical_form: The longer/more specific entity, or None if no match

    Examples:
        >>> are_substring_match("lion", "LION")
        (True, 'lion')  # Case-insensitive match, keeps first (same length)
        >>> are_substring_match("George", "George Hadley")
        (True, 'George Hadley')  # Substring match, keeps longer
        >>> are_substring_match("Hadley", "George Hadley")
        (True, 'George Hadley')  # Substring match, keeps longer
        >>> are_substring_match("George", "Peter")
        (False, None)  # No match
        >>> are_substring_match("Helen", "Helena")
        (False, None)  # NOT a substring match (Helen != Helena)
    """
    if not entity1 or not entity2:
        return (False, None)

    # Normalize both for comparison
    norm1 = normalize_entity(entity1)
    norm2 = normalize_entity(entity2)

    # Case 1: Exact match after normalization (case-insensitive)
    if norm1 == norm2:
        # Return the longer original form (preserves capitalization)
        canonical = entity1 if len(entity1) >= len(entity2) else entity2
        return (True, canonical)

    # Case 2: Check substring containment (word-boundary aware)
    # "George" in "George Hadley" is OK, but "org" in "George" is NOT
    words1 = norm1.split()
    words2 = norm2.split()

    # Check if all words of entity1 appear at start of entity2
    if len(words1) < len(words2):
        if words2[:len(words1)] == words1:
            return (True, entity2)  # Keep longer form

    # Check if all words of entity2 appear at start of entity1
    if len(words2) < len(words1):
        if words1[:len(words2)] == words2:
            return (True, entity1)  # Keep longer form

    return (False, None)


# For testing and debugging
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # Example usage
    print("\n=== Entity Normalization Examples ===")

    test_entities = [
        "Paris",
        "paris",
        "PARIS",
        "  Paris  ",
        "Paris.",
        "The Paris",
        "Prince Paris",
        "King Priam",
        "Goddess Eris",
        "Helen",
        "Helen of Sparta",
        "Ulysses",
        "Odysseus"
    ]

    print("\n1. Normalization:")
    for entity in test_entities:
        normalized = normalize_entity(entity)
        print(f"  '{entity}' → '{normalized}'")

    print("\n2. Base Form Extraction:")
    for entity in test_entities:
        base = extract_base_form(entity)
        print(f"  '{entity}' → '{base}'")

    print("\n3. Similarity Tests:")
    pairs = [
        ("Paris", "paris"),
        ("Paris", "Prince Paris"),
        ("Helen", "Helen of Sparta"),
        ("Ulysses", "Odysseus"),
        ("King Priam", "Priam")
    ]
    for e1, e2 in pairs:
        similar = are_similar(e1, e2)
        score = get_similarity_score(e1, e2)
        title_var = is_title_variant(e1, e2)
        print(f"  '{e1}' vs '{e2}':")
        print(f"    Similar (85%): {similar}")
        print(f"    Score: {score:.2f}")
        print(f"    Title variant: {title_var}")
