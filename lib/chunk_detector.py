"""
Chunk Type Detection for Hybrid Documents

Three approaches available:
1. ChunkTypeDetector: Pattern-based (fast, English-only)
2. MultilingualChunkDetector: Pattern-based with auto language detection (French, English)
3. LLMChunkClassifier: LLM-based (any language, most accurate)

This enables processing documents with mixed content types:
- A narrative (Odyssey) may have technical sections (sailing, weapons)
- A technical manual may have conversational examples (dialogues)
- A research paper has narrative intro, technical methods, conversational quotes
"""
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Tuple, Dict, List, Optional
from .doc_classifier import DocumentType


# =============================================================================
# LANGUAGE-SPECIFIC PATTERN SETS
# =============================================================================

# English patterns
ENGLISH_PATTERNS = {
    'technical': [
        r'\d+\.\d+',                           # Decimal numbers (3.14, v2.0)
        r'\d+%',                               # Percentages (50%)
        r'\b(is defined as|refers to|means|configure|endpoint)\b',
        r'^\s*[-•*]\s+',                       # Bullet points
        r'\b(step\s*\d|figure\s*\d|table\s*\d)\b',
        r'\b(must|shall|should|requires|use)\b',
        r'\b[A-Z]{2,}\b',                      # Acronyms (API, HTTP)
        r'\b(config|setup|install|parameter|function|method|command)\b',
    ],
    'conversational': [
        r'^[A-Z][a-z]+\s*:',                   # Speaker: text
        r'"[^"]{10,}"',                        # Quoted speech
        r'\b(said|asked|replied|answered|responded)\b',
        r'\b(I think|we should|let\'s|you\'re|I\'m)\b',
        r'\?\s*$',                             # Questions at line end
        r'\b(meeting|discussion|agenda|action item|follow.?up)\b',
    ],
    'scientific': [
        r'\(\d{4}\)',                          # Year citations (2023)
        r'\bet al\.',                          # Academic citation
        r'\bp\s*[<>=]\s*0?\.\d+',              # P-values (p < 0.05)
        r'\b(hypothesis|method|results|conclusion|abstract)\b',
        r'\b(study|research|experiment|data|sample|population)\b',
        r'\b(significant|correlation|analysis|statistical)\b',
        r'[A-Z][a-z]+\s+(&|and)\s+[A-Z][a-z]+',
    ],
    'narrative': [
        r'\b(he|she|they|it)\s+(was|were|had|went|saw|came|stood|returned)\b',
        r'\b(suddenly|meanwhile|later|finally|once|then|afterward)\b',
        r'\b(said|thought|felt|wondered|knew|believed|realized)\b',
        r'\b(the hero|the king|the queen|the goddess|the god|the warrior)\b',
        r'\b(journey|battle|quest|return|voyage|adventure|kingdom)\b',
        r'\b[A-Z][a-z]+\s+(returned|traveled|journeyed|fought|loved|died)\b',
        r'\b(cunning|brave|wise|beautiful|mighty|ancient)\b',
        r'\b(years|days|nights|moons|seasons)\s+(passed|went|ago)\b',
    ],
    # Common words for language detection
    'detection': [
        r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\bin\b',
        r'\bis\b', r'\bwas\b', r'\bfor\b', r'\bthat\b', r'\bwith\b',
    ],
}

# French patterns
FRENCH_PATTERNS = {
    'technical': [
        r'\d+\.\d+',                           # Decimal numbers (same)
        r'\d+%',                               # Percentages (same)
        r'\b(est défini comme|signifie|correspond à|configurer)\b',
        r'^\s*[-•*]\s+',                       # Bullet points (same)
        r'\b(étape\s*\d|figure\s*\d|tableau\s*\d)\b',
        r'\b(doit|doivent|faut|nécessite|utiliser)\b',
        r'\b[A-Z]{2,}\b',                      # Acronyms (same)
        r'\b(configuration|paramètre|fonction|méthode|commande|installer)\b',
    ],
    'conversational': [
        r'^[A-ZÀ-Ý][a-zà-ÿ]+\s*:',            # Speaker: text (with accents)
        r'«[^»]{10,}»',                        # French quotes « »
        r'"[^"]{10,}"',                        # Standard quotes
        r'\b(dit|demandé|répondu|déclaré|expliqué)\b',
        r'\b(je pense|nous devrions|on devrait|tu es|je suis)\b',
        r'\?\s*$',                             # Questions (same)
        r'\b(réunion|discussion|ordre du jour|point d\'action|suivi)\b',
    ],
    'scientific': [
        r'\(\d{4}\)',                          # Year citations (same)
        r'\bet al\.',                          # Academic citation (same)
        r'\bp\s*[<>=]\s*0?\.\d+',              # P-values (same)
        r'\b(hypothèse|méthode|résultats|conclusion|résumé)\b',
        r'\b(étude|recherche|expérience|données|échantillon|population)\b',
        r'\b(significatif|corrélation|analyse|statistique)\b',
        r'[A-Z][a-zà-ÿ]+\s+(&|et)\s+[A-Z][a-zà-ÿ]+',
    ],
    'narrative': [
        r'\b(il|elle|ils|elles)\s+(était|étaient|avait|alla|vit|vint)\b',
        r'\b(soudain|pendant ce temps|plus tard|enfin|alors|ensuite)\b',
        r'\b(dit|pensa|sentit|se demanda|savait|croyait|réalisa)\b',
        r'\b(le héros|le roi|la reine|la déesse|le dieu|le guerrier)\b',
        r'\b(voyage|bataille|quête|retour|aventure|royaume)\b',
        r'\b[A-ZÀ-Ý][a-zà-ÿ]+\s+(retourna|voyagea|combattit|aima|mourut)\b',
        r'\b(rusé|brave|sage|belle|puissant|ancien)\b',
        r'\b(années|jours|nuits|lunes|saisons)\s+(passèrent|s\'écoulèrent)\b',
    ],
    # Common words for language detection
    'detection': [
        r'\ble\b', r'\bla\b', r'\bles\b', r'\bde\b', r'\bdu\b',
        r'\bet\b', r'\best\b', r'\bpour\b', r'\bque\b', r'\bavec\b',
        r'\bun\b', r'\bune\b', r'\bdans\b', r'\bà\b', r'\bdes\b',
    ],
}


class ChunkTypeDetector:
    """
    Detect chunk content type using pattern matching.

    Fast heuristics without LLM calls:
    - Technical: numbers, formulas, definitions, bullet points
    - Conversational: speaker markers, quotes, dialogue
    - Scientific: citations, methods, statistics
    - Narrative: action verbs, character names, past tense

    Usage:
        detector = ChunkTypeDetector()
        chunk_type, confidence = detector.detect("Once upon a time...")
        print(chunk_type)  # DocumentType.NARRATIVE
    """

    # Technical patterns - manuals, specs, factual docs
    TECHNICAL_PATTERNS = [
        r'\d+\.\d+',                           # Decimal numbers (3.14, v2.0)
        r'\d+%',                               # Percentages (50%)
        r'\b(is defined as|refers to|means|configure|endpoint)\b',  # Definitions
        r'^\s*[-•*]\s+',                       # Bullet points
        r'\b(step\s*\d|figure\s*\d|table\s*\d)\b',  # References (Step 1, Figure 2)
        r'\b(must|shall|should|requires|use)\b',   # Requirements language
        r'\b[A-Z]{2,}\b',                      # Acronyms (API, HTTP, DNS)
        r'\b(config|setup|install|parameter|function|method|command)\b',  # Tech terms
    ]

    # Conversational patterns - transcripts, meetings, chats
    CONVERSATIONAL_PATTERNS = [
        r'^[A-Z][a-z]+\s*:',                   # Speaker: text
        r'"[^"]{10,}"',                        # Quoted speech (10+ chars)
        r'\b(said|asked|replied|answered|responded)\b',  # Dialogue verbs
        r'\b(I think|we should|let\'s|you\'re|I\'m)\b',  # First/second person
        r'\?\s*$',                             # Questions at line end
        r'\b(meeting|discussion|agenda|action item|follow.?up)\b',  # Meeting terms
    ]

    # Scientific patterns - papers, research, studies
    SCIENTIFIC_PATTERNS = [
        r'\(\d{4}\)',                          # Year citations (2023)
        r'\bet al\.',                          # Academic citation
        r'\bp\s*[<>=]\s*0?\.\d+',              # P-values (p < 0.05)
        r'\b(hypothesis|method|results|conclusion|abstract)\b',
        r'\b(study|research|experiment|data|sample|population)\b',
        r'\b(significant|correlation|analysis|statistical)\b',
        r'[A-Z][a-z]+\s+(&|and)\s+[A-Z][a-z]+',  # Author pairs (Smith & Jones)
    ]

    # Narrative patterns - stories, literature, myths
    NARRATIVE_PATTERNS = [
        r'\b(he|she|they|it)\s+(was|were|had|went|saw|came|stood|returned)\b',  # Third person past
        r'\b(suddenly|meanwhile|later|finally|once|then|afterward)\b',  # Temporal markers
        r'\b(said|thought|felt|wondered|knew|believed|realized)\b',  # Narrative verbs
        r'\b(the hero|the king|the queen|the goddess|the god|the warrior)\b',  # Character archetypes
        r'\b(journey|battle|quest|return|voyage|adventure|kingdom)\b',  # Story elements
        r'\b[A-Z][a-z]+\s+(returned|traveled|journeyed|fought|loved|died)\b',  # Character + past verb
        r'\b(cunning|brave|wise|beautiful|mighty|ancient)\b',  # Epic adjectives
        r'\b(years|days|nights|moons|seasons)\s+(passed|went|ago)\b',  # Time passing
    ]

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize detector.

        Args:
            confidence_threshold: Min score to assign a specific type (0.0-1.0).
                                  Below this threshold, returns UNKNOWN (hybrid).
        """
        self.threshold = confidence_threshold
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.patterns = {
            DocumentType.TECHNICAL: [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in self.TECHNICAL_PATTERNS
            ],
            DocumentType.CONVERSATIONAL: [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in self.CONVERSATIONAL_PATTERNS
            ],
            DocumentType.SCIENTIFIC: [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in self.SCIENTIFIC_PATTERNS
            ],
            DocumentType.NARRATIVE: [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in self.NARRATIVE_PATTERNS
            ],
        }

    def detect(self, chunk: str) -> Tuple[DocumentType, float]:
        """
        Detect chunk type using pattern matching.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Tuple of (DocumentType, confidence score 0.0-1.0)
            Returns DocumentType.UNKNOWN if confidence below threshold.
        """
        if not chunk or not chunk.strip():
            return DocumentType.UNKNOWN, 0.0

        scores = {}

        for doc_type, patterns in self.patterns.items():
            matches = sum(1 for p in patterns if p.search(chunk))
            scores[doc_type] = matches / len(patterns) if patterns else 0.0

        # Find highest score
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # If below threshold, return UNKNOWN (will use hybrid prompt)
        if best_score < self.threshold:
            return DocumentType.UNKNOWN, best_score

        return best_type, best_score

    def detect_with_details(self, chunk: str) -> Dict:
        """
        Detect with detailed breakdown of pattern matches.

        Useful for debugging and understanding why a type was chosen.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Dict with:
                - scores: Dict of type -> score
                - detected_type: DocumentType
                - confidence: float
                - matched_patterns: Dict of type -> list of matched pattern strings
        """
        result = {
            'scores': {},
            'detected_type': DocumentType.UNKNOWN,
            'confidence': 0.0,
            'matched_patterns': {}
        }

        if not chunk or not chunk.strip():
            return result

        for doc_type, patterns in self.patterns.items():
            matched = [p.pattern for p in patterns if p.search(chunk)]
            score = len(matched) / len(patterns) if patterns else 0.0
            result['scores'][doc_type.value] = score
            result['matched_patterns'][doc_type.value] = matched

        detected, confidence = self.detect(chunk)
        result['detected_type'] = detected
        result['confidence'] = confidence

        return result

    def get_all_scores(self, chunk: str) -> Dict[DocumentType, float]:
        """
        Get scores for all document types without applying threshold.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Dict mapping DocumentType to score (0.0-1.0)
        """
        scores = {}

        for doc_type, patterns in self.patterns.items():
            if patterns:
                matches = sum(1 for p in patterns if p.search(chunk))
                scores[doc_type] = matches / len(patterns)
            else:
                scores[doc_type] = 0.0

        return scores

    def add_pattern(self, doc_type: DocumentType, pattern: str):
        """
        Add a custom pattern for a document type.

        Args:
            doc_type: Document type to add pattern for
            pattern: Regex pattern string
        """
        if doc_type not in self.patterns:
            self.patterns[doc_type] = []

        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.patterns[doc_type].append(compiled)

    @property
    def threshold(self) -> float:
        """Get current confidence threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        """Set confidence threshold (clamped to 0.0-1.0)."""
        self._threshold = max(0.0, min(1.0, value))


def detect_chunk_type(chunk: str, threshold: float = 0.3) -> Tuple[DocumentType, float]:
    """
    Quick helper to detect chunk type using pattern matching.

    Args:
        chunk: Text chunk to analyze
        threshold: Confidence threshold

    Returns:
        Tuple of (DocumentType, confidence)
    """
    detector = ChunkTypeDetector(confidence_threshold=threshold)
    return detector.detect(chunk)


class LLMChunkClassifier:
    """
    Language-agnostic chunk classifier using lightweight LLM calls.

    Supports any language (French, English, etc.) by using the LLM's
    multilingual capabilities. Optimized with caching to minimize LLM calls.

    Usage:
        classifier = LLMChunkClassifier(llm)
        chunk_type, confidence = classifier.classify("Il était une fois...")
        print(chunk_type)  # DocumentType.NARRATIVE

        # Or with batch processing
        types = classifier.classify_batch([chunk1, chunk2, chunk3])
    """

    # Very short classification prompt (works in any language)
    CLASSIFICATION_PROMPT = '''Classify this text into ONE category:
- NARRATIVE (stories, characters, events, fiction)
- TECHNICAL (instructions, definitions, specifications, manuals)
- CONVERSATIONAL (dialogue, meetings, discussions, interviews)
- SCIENTIFIC (research, hypotheses, citations, experiments)
- UNKNOWN (unclear or mixed content)

Text: "{text}"

Category (respond with ONE word only):'''

    # Batch classification prompt
    BATCH_PROMPT = '''Classify each text into ONE category (NARRATIVE, TECHNICAL, CONVERSATIONAL, SCIENTIFIC, or UNKNOWN).

{chunks}

Respond with ONLY the numbers and categories, one per line:
1. CATEGORY
2. CATEGORY
...'''

    def __init__(self, llm, use_cache: bool = True, max_text_length: int = 500,
                 timeout: int = 30, debug: bool = False):
        """
        Initialize the LLM-based chunk classifier.

        Args:
            llm: LangChain LLM instance for classification
            use_cache: Whether to cache results (default True)
            max_text_length: Max characters to send to LLM (default 500)
            timeout: Timeout in seconds for LLM calls (default 30)
            debug: Whether to print debug timing info
        """
        self.llm = llm
        self.use_cache = use_cache
        self.max_text_length = max_text_length
        self.timeout = timeout
        self.debug = debug
        self._cache: Dict[str, DocumentType] = {}
        self._stats = {'cache_hits': 0, 'llm_calls': 0, 'timeouts': 0}

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        sample = text[:self.max_text_length]
        return hashlib.md5(sample.encode()).hexdigest()

    def classify(self, chunk: str) -> Tuple[DocumentType, float]:
        """
        Classify chunk type using LLM.

        Args:
            chunk: Text chunk to classify (any language)

        Returns:
            Tuple of (DocumentType, confidence)
            Confidence is 1.0 for LLM classification (high confidence)
        """
        if not chunk or not chunk.strip():
            return DocumentType.UNKNOWN, 0.0

        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(chunk)
            if cache_key in self._cache:
                self._stats['cache_hits'] += 1
                return self._cache[cache_key], 1.0

        # Truncate text for LLM
        text_sample = chunk[:self.max_text_length]
        prompt = self.CLASSIFICATION_PROMPT.format(text=text_sample)

        try:
            if self.debug:
                print(f"    [DEBUG] ChunkClassifier: starting LLM call...")
                t0 = time.time()

            self._stats['llm_calls'] += 1

            # LLM call with timeout protection
            def llm_call():
                return self.llm.invoke(prompt)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm_call)
                try:
                    response = future.result(timeout=self.timeout)
                except FuturesTimeoutError:
                    self._stats['timeouts'] += 1
                    if self.debug:
                        print(f"    [DEBUG] ChunkClassifier: TIMEOUT after {self.timeout}s, using UNKNOWN")
                    # Cache the timeout result to avoid repeated timeouts
                    if self.use_cache:
                        self._cache[cache_key] = DocumentType.UNKNOWN
                    return DocumentType.UNKNOWN, 0.0

            if self.debug:
                print(f"    [DEBUG] ChunkClassifier: LLM call took {time.time()-t0:.2f}s")

            # Handle different response types
            if isinstance(response, dict):
                response_text = response.get('result', '') or response.get('text', str(response))
            else:
                response_text = str(response)

            response_text = response_text.strip().upper()

            # Parse response - look for document type keywords
            doc_type = DocumentType.UNKNOWN
            for dt in DocumentType:
                if dt.value.upper() in response_text:
                    doc_type = dt
                    break

            # Cache result
            if self.use_cache:
                self._cache[cache_key] = doc_type

            return doc_type, 1.0

        except Exception as e:
            print(f"[LLMChunkClassifier] Error: {e}")
            return DocumentType.UNKNOWN, 0.0

    def classify_batch(self, chunks: List[str]) -> List[Tuple[DocumentType, float]]:
        """
        Classify multiple chunks in one LLM call for efficiency.

        Args:
            chunks: List of text chunks to classify

        Returns:
            List of (DocumentType, confidence) tuples, one per chunk
        """
        if not chunks:
            return []

        # Check cache for each chunk, identify which need LLM
        results: List[Optional[Tuple[DocumentType, float]]] = [None] * len(chunks)
        uncached_indices: List[int] = []

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                results[i] = (DocumentType.UNKNOWN, 0.0)
                continue

            if self.use_cache:
                cache_key = self._get_cache_key(chunk)
                if cache_key in self._cache:
                    self._stats['cache_hits'] += 1
                    results[i] = (self._cache[cache_key], 1.0)
                    continue

            uncached_indices.append(i)

        # If all cached, return early
        if not uncached_indices:
            return results  # type: ignore

        # Build batch prompt for uncached chunks
        chunks_text = ""
        for idx, i in enumerate(uncached_indices):
            text_sample = chunks[i][:200]  # Shorter for batch
            chunks_text += f'{idx + 1}. "{text_sample}..."\n'

        prompt = self.BATCH_PROMPT.format(chunks=chunks_text)

        try:
            self._stats['llm_calls'] += 1
            response = self.llm.invoke(prompt)

            if isinstance(response, dict):
                response_text = response.get('result', '') or str(response)
            else:
                response_text = str(response)

            # Parse batch response (expects "1. NARRATIVE\n2. TECHNICAL\n...")
            lines = response_text.strip().split('\n')
            parsed_types: List[DocumentType] = []

            for line in lines:
                line = line.strip().upper()
                if not line:
                    continue

                doc_type = DocumentType.UNKNOWN
                for dt in DocumentType:
                    if dt.value.upper() in line:
                        doc_type = dt
                        break
                parsed_types.append(doc_type)

            # Map parsed results back to original indices
            for idx, orig_i in enumerate(uncached_indices):
                if idx < len(parsed_types):
                    doc_type = parsed_types[idx]
                else:
                    doc_type = DocumentType.UNKNOWN

                results[orig_i] = (doc_type, 1.0)

                # Cache result
                if self.use_cache:
                    cache_key = self._get_cache_key(chunks[orig_i])
                    self._cache[cache_key] = doc_type

        except Exception as e:
            print(f"[LLMChunkClassifier] Batch error: {e}")
            # Fill remaining with UNKNOWN
            for i in uncached_indices:
                if results[i] is None:
                    results[i] = (DocumentType.UNKNOWN, 0.0)

        return results  # type: ignore

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
        self._stats = {'cache_hits': 0, 'llm_calls': 0, 'timeouts': 0}

    @property
    def cache_size(self) -> int:
        """Return number of cached classifications."""
        return len(self._cache)

    @property
    def stats(self) -> Dict:
        """Return classification statistics."""
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._stats['cache_hits'],
            'llm_calls': self._stats['llm_calls'],
            'timeouts': self._stats.get('timeouts', 0),
        }


def classify_chunk_llm(llm, chunk: str) -> Tuple[DocumentType, float]:
    """
    Quick helper to classify a chunk using LLM.

    Args:
        llm: LangChain LLM instance
        chunk: Text chunk to classify

    Returns:
        Tuple of (DocumentType, confidence)
    """
    classifier = LLMChunkClassifier(llm, use_cache=False)
    return classifier.classify(chunk)


class MultilingualChunkDetector:
    """
    Multi-language chunk type detector using pattern matching.

    Automatically detects language (French or English) and uses
    the appropriate pattern set. Fast, no LLM calls required.

    Supported languages:
    - English (en)
    - French (fr)

    Usage:
        detector = MultilingualChunkDetector()

        # Auto-detect language and classify
        chunk_type, confidence = detector.detect("Il était une fois...")
        print(chunk_type)  # DocumentType.NARRATIVE

        # Check detected language
        lang = detector.detect_language("Bonjour le monde")
        print(lang)  # 'fr'

        # Force a specific language
        detector = MultilingualChunkDetector(language='fr')
    """

    SUPPORTED_LANGUAGES = {'en': ENGLISH_PATTERNS, 'fr': FRENCH_PATTERNS}

    def __init__(self, confidence_threshold: float = 0.3,
                 language: Optional[str] = None,
                 auto_detect: bool = True):
        """
        Initialize multilingual detector.

        Args:
            confidence_threshold: Min score to assign a type (0.0-1.0)
            language: Force a specific language ('en', 'fr'). If None, auto-detect.
            auto_detect: Whether to auto-detect language per chunk (default True)
        """
        self.threshold = confidence_threshold
        self.forced_language = language
        self.auto_detect = auto_detect and language is None
        self._detected_language: Optional[str] = None
        self._compiled_patterns: Dict[str, Dict[DocumentType, List]] = {}
        self._compile_all_patterns()

    def _compile_all_patterns(self):
        """Compile patterns for all supported languages."""
        for lang, patterns in self.SUPPORTED_LANGUAGES.items():
            self._compiled_patterns[lang] = {
                DocumentType.TECHNICAL: [
                    re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in patterns['technical']
                ],
                DocumentType.CONVERSATIONAL: [
                    re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in patterns['conversational']
                ],
                DocumentType.SCIENTIFIC: [
                    re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in patterns['scientific']
                ],
                DocumentType.NARRATIVE: [
                    re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in patterns['narrative']
                ],
            }
            # Compile language detection patterns
            self._compiled_patterns[f'{lang}_detection'] = [
                re.compile(p, re.IGNORECASE)
                for p in patterns['detection']
            ]

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using common word patterns.

        Args:
            text: Text to analyze

        Returns:
            Language code ('en', 'fr') or 'en' as default
        """
        if self.forced_language:
            return self.forced_language

        scores = {}
        for lang in self.SUPPORTED_LANGUAGES.keys():
            detection_patterns = self._compiled_patterns.get(f'{lang}_detection', [])
            if detection_patterns:
                matches = sum(1 for p in detection_patterns if p.search(text))
                scores[lang] = matches / len(detection_patterns)
            else:
                scores[lang] = 0.0

        # Return language with highest score
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0.1:  # Minimum threshold
                return best_lang

        return 'en'  # Default to English

    def detect(self, chunk: str) -> Tuple[DocumentType, float]:
        """
        Detect chunk type using language-appropriate patterns.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Tuple of (DocumentType, confidence score 0.0-1.0)
        """
        if not chunk or not chunk.strip():
            return DocumentType.UNKNOWN, 0.0

        # Detect or use forced language
        lang = self.detect_language(chunk) if self.auto_detect else (self.forced_language or 'en')
        self._detected_language = lang

        # Get patterns for this language
        patterns = self._compiled_patterns.get(lang, self._compiled_patterns.get('en', {}))

        scores = {}
        for doc_type, type_patterns in patterns.items():
            if isinstance(doc_type, DocumentType):
                matches = sum(1 for p in type_patterns if p.search(chunk))
                scores[doc_type] = matches / len(type_patterns) if type_patterns else 0.0

        if not scores:
            return DocumentType.UNKNOWN, 0.0

        # Find highest score
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # If below threshold, return UNKNOWN
        if best_score < self.threshold:
            return DocumentType.UNKNOWN, best_score

        return best_type, best_score

    def detect_with_details(self, chunk: str) -> Dict:
        """
        Detect with detailed breakdown including language.

        Returns:
            Dict with language, scores, detected_type, confidence
        """
        if not chunk or not chunk.strip():
            return {
                'language': None,
                'scores': {},
                'detected_type': DocumentType.UNKNOWN,
                'confidence': 0.0
            }

        lang = self.detect_language(chunk)
        detected_type, confidence = self.detect(chunk)

        # Get all scores
        patterns = self._compiled_patterns.get(lang, {})
        scores = {}
        for doc_type, type_patterns in patterns.items():
            if isinstance(doc_type, DocumentType):
                matches = sum(1 for p in type_patterns if p.search(chunk))
                scores[doc_type.value] = matches / len(type_patterns) if type_patterns else 0.0

        return {
            'language': lang,
            'scores': scores,
            'detected_type': detected_type,
            'confidence': confidence
        }

    @property
    def detected_language(self) -> Optional[str]:
        """Return last detected language."""
        return self._detected_language

    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return list(self.SUPPORTED_LANGUAGES.keys())


def detect_chunk_multilingual(chunk: str, language: Optional[str] = None) -> Tuple[DocumentType, float]:
    """
    Quick helper to detect chunk type with multi-language support.

    Args:
        chunk: Text chunk to analyze
        language: Optional language code ('en', 'fr'). If None, auto-detect.

    Returns:
        Tuple of (DocumentType, confidence)
    """
    detector = MultilingualChunkDetector(language=language)
    return detector.detect(chunk)
