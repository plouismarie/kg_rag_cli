"""
Phase 2: Enhanced Entity Extraction Module

Provides LLM-based entity extraction from user queries with:
- Multiple extraction strategies (regex, NLP, LLM)
- Fuzzy matching against known entities
- Entity resolution and normalization
- Caching for repeated queries
"""
from typing import List, Set, Dict, Optional, Tuple, Any
import re
from functools import lru_cache


class EntityExtractor:
    """
    LLM-enhanced entity extraction for Knowledge Graph queries.

    Combines multiple strategies:
    1. Pattern-based extraction (proper nouns, capitalized words)
    2. Known entity matching (substring, fuzzy)
    3. LLM-based extraction (fallback for complex queries)

    Usage:
        extractor = EntityExtractor(llm, known_entities)
        entities = extractor.extract("What is the relationship between Paris and France?")
        # Returns: ['Paris', 'France']
    """

    def __init__(self, llm=None, known_entities: Optional[Set[str]] = None,
                 use_cache: bool = True):
        """
        Initialize entity extractor.

        Args:
            llm: LangChain LLM instance for advanced extraction
            known_entities: Set of known entity names from the graph
            use_cache: Whether to cache extraction results
        """
        self.llm = llm
        self.known_entities = known_entities or set()
        self.use_cache = use_cache
        self._cache: Dict[str, List[str]] = {}

        # Build lowercase index for case-insensitive matching
        self._entity_lower_map: Dict[str, str] = {
            e.lower(): e for e in self.known_entities
        }

    def update_known_entities(self, entities: Set[str]):
        """Update the set of known entities (e.g., after graph updates)."""
        self.known_entities = entities
        self._entity_lower_map = {e.lower(): e for e in entities}
        self._cache.clear()

    def extract(self, query: str, use_llm: bool = True,
                max_entities: int = 10) -> List[str]:
        """
        Extract entities from a user query.

        Args:
            query: User's question or query string
            use_llm: Whether to use LLM for extraction (slower but better)
            max_entities: Maximum number of entities to return

        Returns:
            List of matched entity names from the knowledge graph
        """
        # Check cache first
        cache_key = f"{query}:{use_llm}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        matched = []

        # Strategy 1: Direct known entity matching
        matched.extend(self._match_known_entities(query))

        # Strategy 2: Pattern-based extraction (capitalized words, proper nouns)
        if len(matched) < max_entities:
            pattern_entities = self._extract_by_pattern(query)
            matched.extend(self._resolve_entities(pattern_entities))

        # Strategy 3: LLM-based extraction (most powerful but slowest)
        if use_llm and self.llm and len(matched) < 2:
            llm_entities = self._extract_with_llm(query)
            matched.extend(self._resolve_entities(llm_entities))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in matched:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        result = unique[:max_entities]

        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result

        return result

    def _match_known_entities(self, query: str) -> List[str]:
        """
        Match query against known entities using substring matching.

        Tries:
        1. Exact substring match
        2. Word-based matching for multi-word entities
        """
        matched = []
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 2)

        for entity in self.known_entities:
            entity_lower = entity.lower()

            # Exact substring match
            if entity_lower in query_lower:
                matched.append(entity)
                continue

            # Check if any query word matches part of entity
            entity_words = set(entity_lower.split())
            if entity_words & query_words:
                # At least one word matches
                matched.append(entity)

        return matched

    def _extract_by_pattern(self, query: str) -> List[str]:
        """
        Extract potential entities using patterns.

        Patterns:
        - Capitalized words (proper nouns)
        - Quoted strings
        - Words after "about", "of", "between"
        """
        candidates = []

        # Pattern 1: Capitalized words (likely proper nouns)
        # Exclude first word of sentence and common words
        cap_pattern = r'(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        caps = re.findall(cap_pattern, query)
        candidates.extend(caps)

        # Pattern 2: Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        candidates.extend(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        candidates.extend(quoted)

        # Pattern 3: Entity-indicating phrases
        indicators = [
            r'about\s+([A-Za-z][A-Za-z\s]+?)(?:\s+and|\s+or|\?|$)',
            r'between\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)',
            r'relationship\s+(?:between\s+)?([A-Za-z]+)',
            r'(?:who|what)\s+is\s+([A-Za-z]+)',
        ]

        for pattern in indicators:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    candidates.extend(m)
                else:
                    candidates.append(m)

        return [c.strip() for c in candidates if c.strip()]

    def _extract_with_llm(self, query: str) -> List[str]:
        """
        Use LLM to extract entities from complex queries.

        This is the most powerful but slowest method.
        """
        if not self.llm:
            return []

        prompt = f"""Extract the key entities (people, places, concepts, organizations) from this query.
Return ONLY a comma-separated list of entity names, nothing else.
Do not include common words like "relationship", "connection", "about".

Query: {query}

Entities:"""

        try:
            response = self.llm.invoke(prompt)

            # Handle different response types
            if isinstance(response, dict):
                text = response.get('result', '') or response.get('text', '')
            else:
                text = str(response)

            # Parse comma-separated list
            entities = [e.strip() for e in text.strip().split(',')]
            entities = [e for e in entities if e and len(e) > 1]

            return entities

        except Exception as e:
            print(f"[EntityExtractor] LLM extraction failed: {e}")
            return []

    def _resolve_entities(self, candidates: List[str]) -> List[str]:
        """
        Resolve candidate strings to known entities using fuzzy matching.

        Matching strategies:
        1. Exact match (case-insensitive)
        2. Substring match
        3. Word overlap
        """
        resolved = []

        for candidate in candidates:
            cand_lower = candidate.lower()

            # Strategy 1: Exact match
            if cand_lower in self._entity_lower_map:
                resolved.append(self._entity_lower_map[cand_lower])
                continue

            # Strategy 2: Substring match (candidate in entity or vice versa)
            best_match = None
            best_score = 0

            for entity_lower, entity in self._entity_lower_map.items():
                score = 0

                if cand_lower in entity_lower:
                    # Candidate is substring of entity
                    score = len(cand_lower) / len(entity_lower)
                elif entity_lower in cand_lower:
                    # Entity is substring of candidate
                    score = len(entity_lower) / len(cand_lower)
                else:
                    # Word overlap
                    cand_words = set(cand_lower.split())
                    entity_words = set(entity_lower.split())
                    overlap = cand_words & entity_words
                    if overlap:
                        score = len(overlap) / max(len(cand_words), len(entity_words)) * 0.8

                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = entity

            if best_match:
                resolved.append(best_match)

        return resolved

    def explain_extraction(self, query: str) -> Dict[str, any]:
        """
        Debug method to show how entities were extracted.

        Returns detailed breakdown of extraction process.
        """
        result = {
            'query': query,
            'strategies': {},
            'final_entities': []
        }

        # Strategy 1
        known_matches = self._match_known_entities(query)
        result['strategies']['known_entity_matching'] = known_matches

        # Strategy 2
        pattern_candidates = self._extract_by_pattern(query)
        pattern_resolved = self._resolve_entities(pattern_candidates)
        result['strategies']['pattern_extraction'] = {
            'candidates': pattern_candidates,
            'resolved': pattern_resolved
        }

        # Strategy 3
        if self.llm:
            llm_candidates = self._extract_with_llm(query)
            llm_resolved = self._resolve_entities(llm_candidates)
            result['strategies']['llm_extraction'] = {
                'candidates': llm_candidates,
                'resolved': llm_resolved
            }

        # Final result
        result['final_entities'] = self.extract(query)

        return result


class EntityResolver:
    """
    Resolves entity mentions to canonical forms with intelligent deduplication.

    This enhanced resolver handles:
    - Case variations ("Paris", "paris", "PARIS" → "Paris")
    - Title variants ("Prince Paris", "Paris" → "Paris")
    - Typos and fuzzy matching (configurable similarity threshold)
    - Automatic entity registration
    - Usage frequency tracking

    The resolver maintains:
    - canonical_map: normalized → canonical form
    - aliases: explicit alias mappings
    - entity_frequency: usage counts for each canonical entity

    Example:
        >>> resolver = EntityResolver(known_entities={"Paris", "Troy"})
        >>> resolver.resolve("paris")
        'Paris'
        >>> resolver.resolve("Prince Paris")
        'Paris'
        >>> resolver.resolve("PARIS OF TROY")
        'Paris'  # Fuzzy match
    """

    def __init__(self, known_entities: Set[str] = None, fuzzy_threshold: float = 0.85,
                 dedup_mode: str = 'standard'):
        """
        Initialize the entity resolver.

        Args:
            known_entities: Set of known entity names (optional)
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0, default: 0.85)
            dedup_mode: Deduplication mode:
                - 'standard': Full dedup (case + title + fuzzy matching)
                - 'light': Light dedup (case-insensitive + substring only)
                - 'none': No dedup (pass-through)
        """
        self.known_entities = known_entities or set()
        self.aliases: Dict[str, str] = {}  # alias → canonical
        self.canonical_map: Dict[str, str] = {}  # normalized → canonical
        self.entity_frequency: Dict[str, int] = {}  # canonical → usage count
        self.fuzzy_threshold = fuzzy_threshold
        self.dedup_mode = dedup_mode

        # Build initial canonical registry from known entities
        if self.known_entities:
            self._build_canonical_registry()

    def _build_canonical_registry(self):
        """Build canonical registry from known entities."""
        from .entity_normalizer import normalize_entity

        for entity in self.known_entities:
            normalized = normalize_entity(entity)
            # Only add if not already present (first one wins)
            if normalized not in self.canonical_map:
                self.canonical_map[normalized] = entity
                self.entity_frequency[entity] = 0

    def _build_default_aliases(self):
        """
        Build common aliases from known entities.

        DEPRECATED: Replaced by _build_canonical_registry().
        Kept for backward compatibility.
        """
        for entity in self.known_entities:
            # Lowercase version
            self.aliases[entity.lower()] = entity

            # Handle "The X" -> "X"
            if entity.lower().startswith('the '):
                self.aliases[entity[4:].lower()] = entity

    def add_alias(self, alias: str, canonical: str):
        """
        Add a custom alias mapping.

        Args:
            alias: The alias form
            canonical: The canonical entity name
        """
        from .entity_normalizer import normalize_entity

        alias_normalized = normalize_entity(alias)
        self.aliases[alias_normalized] = canonical

        # Note: Do NOT add to canonical_map - that's only for canonical entities
        # Aliases are stored separately in self.aliases

    def register_entity(self, canonical: str, aliases: List[str] = None):
        """
        Register a new canonical entity with optional aliases.

        Args:
            canonical: The canonical entity name
            aliases: List of alias forms (optional)
        """
        from .entity_normalizer import normalize_entity

        normalized_canonical = normalize_entity(canonical)

        # Add to canonical map
        self.canonical_map[normalized_canonical] = canonical
        self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0)
        self.known_entities.add(canonical)

        # Add aliases if provided
        if aliases:
            for alias in aliases:
                self.add_alias(alias, canonical)

    def _select_canonical(self, candidates: List[str]) -> str:
        """
        Choose the canonical form from candidates.

        Selection strategy:
        1. Prefer capitalized forms ("Paris" > "paris")
        2. Prefer shorter forms without titles ("Paris" > "Prince Paris")
        3. Use frequency if ambiguous (most common form)

        Args:
            candidates: List of candidate entity mentions

        Returns:
            The selected canonical form
        """
        from .entity_normalizer import extract_base_form, normalize_entity

        if not candidates:
            return ""

        if len(candidates) == 1:
            return candidates[0]

        # Step 1: Extract base forms and remove titles
        base_forms = [(extract_base_form(c), c) for c in candidates]

        # Step 2: Prefer forms without titles (base == original)
        without_titles = [c for base, c in base_forms if normalize_entity(base) == normalize_entity(c)]
        if without_titles:
            candidates = without_titles

        # Step 3: Prefer capitalized forms
        capitalized = [c for c in candidates if c and c[0].isupper()]
        if capitalized:
            # Step 4: Prefer shorter forms among capitalized
            return min(capitalized, key=len)

        # Fallback: Use frequency or first candidate
        candidates_with_freq = [(c, self.entity_frequency.get(c, 0)) for c in candidates]
        return max(candidates_with_freq, key=lambda x: x[1])[0] if candidates_with_freq else candidates[0]

    def merge_entities(self, entity1: str, entity2: str, keep: str = None):
        """
        Merge two entities into a single canonical form.

        Args:
            entity1: First entity to merge
            entity2: Second entity to merge
            keep: Which entity to keep as canonical (default: auto-select)
        """
        from .entity_normalizer import normalize_entity

        # Determine which to keep
        if keep is None:
            keep = self._select_canonical([entity1, entity2])

        remove = entity1 if keep == entity2 else entity2

        norm_keep = normalize_entity(keep)
        norm_remove = normalize_entity(remove)

        # Update canonical_map
        self.canonical_map[norm_remove] = keep
        if norm_keep not in self.canonical_map:
            self.canonical_map[norm_keep] = keep

        # Add as alias
        self.aliases[norm_remove] = keep

        # Merge frequencies
        freq_keep = self.entity_frequency.get(keep, 0)
        freq_remove = self.entity_frequency.get(remove, 0)
        self.entity_frequency[keep] = freq_keep + freq_remove

        # Remove merged entity from known_entities
        if remove in self.known_entities:
            self.known_entities.discard(remove)
        if keep not in self.known_entities:
            self.known_entities.add(keep)

    def resolve(self, mention: str, auto_register: bool = True) -> str:
        """
        Resolve entity mention to canonical form.

        Resolution strategy depends on dedup_mode:
        - 'standard': Full dedup (case + title + fuzzy matching)
        - 'light': Light dedup (case-insensitive + substring only)
        - 'none': No dedup (pass-through, just register)

        Args:
            mention: The entity mention to resolve
            auto_register: If True, register new entities automatically

        Returns:
            Canonical entity name
        """
        if self.dedup_mode == 'none':
            # No resolution - just register and return
            if auto_register and mention:
                from .entity_normalizer import normalize_entity
                normalized = normalize_entity(mention)
                if normalized not in self.canonical_map:
                    self.register_entity(mention)
                self.entity_frequency[mention] = self.entity_frequency.get(mention, 0) + 1
            return mention
        elif self.dedup_mode == 'light':
            return self._resolve_light(mention, auto_register)
        else:
            return self._resolve_standard(mention, auto_register)

    def _resolve_light(self, mention: str, auto_register: bool = True) -> str:
        """
        Light deduplication: case-insensitive + substring matching only.

        Resolution strategy:
        1. Exact match in canonical_map (after normalization)
        2. Check explicit aliases
        3. Substring match (keep longer form)
        4. Auto-register as new entity (if enabled)

        NO fuzzy matching - strictly case + substring based.

        Args:
            mention: The entity mention to resolve
            auto_register: If True, register new entities automatically

        Returns:
            Canonical entity name

        Examples:
            >>> resolver._resolve_light("lion")
            'Lion'  # Case-insensitive match
            >>> resolver._resolve_light("George")
            'George Hadley'  # Substring of known entity (keeps longer form)
            >>> resolver._resolve_light("Helen")
            'Helen'  # Does NOT merge with "Helena" (no fuzzy matching)
        """
        from .entity_normalizer import normalize_entity, are_substring_match

        if not mention:
            return mention

        normalized = normalize_entity(mention)

        # Step 1: Check canonical_map (exact match after normalization)
        if normalized in self.canonical_map:
            canonical = self.canonical_map[normalized]
            self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
            return canonical

        # Step 2: Check explicit aliases
        if normalized in self.aliases:
            canonical = self.aliases[normalized]
            self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
            return canonical

        # Step 3: Substring matching (keep longer form)
        for known_normalized, canonical in self.canonical_map.items():
            is_match, canonical_form = are_substring_match(mention, canonical)
            if is_match and canonical_form:
                # Found substring match
                if canonical_form != canonical:
                    # The new mention is longer - update canonical
                    self._update_canonical(canonical, canonical_form)
                    canonical = canonical_form
                self.add_alias(mention, canonical)
                self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
                return canonical

        # Step 4: No match found
        if auto_register:
            canonical = self._select_canonical([mention])
            self.register_entity(canonical, [mention])
            self.entity_frequency[canonical] = 1
            return canonical
        else:
            return mention

    def _update_canonical(self, old_canonical: str, new_canonical: str):
        """
        Update the canonical form when a longer version is found.

        Example: "George" becomes "George Hadley" when we encounter the longer form.
        """
        from .entity_normalizer import normalize_entity

        old_normalized = normalize_entity(old_canonical)
        new_normalized = normalize_entity(new_canonical)

        # Update canonical_map entries that pointed to old canonical
        for norm, canon in list(self.canonical_map.items()):
            if canon == old_canonical:
                self.canonical_map[norm] = new_canonical

        # Add new canonical to map
        self.canonical_map[new_normalized] = new_canonical

        # Transfer frequency
        freq = self.entity_frequency.get(old_canonical, 0)
        self.entity_frequency[new_canonical] = self.entity_frequency.get(new_canonical, 0) + freq
        if old_canonical in self.entity_frequency:
            del self.entity_frequency[old_canonical]

        # Update known_entities
        self.known_entities.discard(old_canonical)
        self.known_entities.add(new_canonical)

        # Add old as alias
        self.aliases[old_normalized] = new_canonical

    def _resolve_standard(self, mention: str, auto_register: bool = True) -> str:
        """
        Standard (full) deduplication: case + title + fuzzy matching.

        Resolution strategy:
        1. Exact match in canonical_map (after normalization)
        2. Check explicit aliases
        3. Title variant matching
        4. Fuzzy match against known entities (>= threshold similarity)
        5. Auto-register as new entity (if enabled)
        6. Return original mention (if auto_register disabled)

        Args:
            mention: The entity mention to resolve
            auto_register: If True, register new entities automatically

        Returns:
            Canonical entity name
        """
        from .entity_normalizer import normalize_entity, are_similar, is_title_variant

        if not mention:
            return mention

        # Normalize the mention
        normalized = normalize_entity(mention)

        # Step 1: Check canonical_map (exact match after normalization)
        if normalized in self.canonical_map:
            canonical = self.canonical_map[normalized]
            self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
            return canonical

        # Step 2: Check explicit aliases
        if normalized in self.aliases:
            canonical = self.aliases[normalized]
            self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
            return canonical

        # Step 3: Check for title variants (fast path)
        for known_normalized, canonical in self.canonical_map.items():
            if is_title_variant(mention, canonical):
                # Found title variant - add as alias
                self.add_alias(mention, canonical)
                self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
                return canonical

        # Step 4: Fuzzy match against known entities
        for known_normalized, canonical in self.canonical_map.items():
            if are_similar(normalized, known_normalized, threshold=self.fuzzy_threshold):
                # Found fuzzy match - add as alias
                self.add_alias(mention, canonical)
                self.entity_frequency[canonical] = self.entity_frequency.get(canonical, 0) + 1
                return canonical

        # Step 5: No match found
        if auto_register:
            # Register as new entity
            canonical = self._select_canonical([mention])
            self.register_entity(canonical, [mention])
            self.entity_frequency[canonical] = 1
            return canonical
        else:
            # Return original mention unchanged
            return mention

    def get_canonical(self, mention: str) -> Optional[str]:
        """
        Get the canonical form without auto-registration.

        Args:
            mention: The entity mention

        Returns:
            Canonical form if found, None otherwise
        """
        result = self.resolve(mention, auto_register=False)
        return result if result != mention else None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get entity resolution statistics.

        Returns:
            Dict with:
            - total_entities: Number of canonical entities
            - total_aliases: Number of alias mappings
            - total_mentions: Total entity mentions resolved
            - top_entities: Most frequently mentioned entities
        """
        total_mentions = sum(self.entity_frequency.values())
        top_entities = sorted(
            self.entity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_entities': len(self.canonical_map),
            'total_aliases': len(self.aliases),
            'total_mentions': total_mentions,
            'top_entities': top_entities,
            'fuzzy_threshold': self.fuzzy_threshold
        }

