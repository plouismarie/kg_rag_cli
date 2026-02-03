"""
Document Type Classifier
Auto-detects document type for optimized KG extraction

This module provides automatic classification of documents into categories
that determine which extraction strategy to use for building knowledge graphs.

Supported types:
- NARRATIVE: Stories, literature, myths (focus on characters, events, journeys)
- TECHNICAL: Manuals, specifications, facts (focus on hierarchy, causation)
- CONVERSATIONAL: Transcripts, meetings, chats (focus on speakers, decisions)
- SCIENTIFIC: Papers, research, studies (focus on hypotheses, methods, findings)
"""
from enum import Enum
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import hashlib
import time


class DocumentType(Enum):
    """Enumeration of supported document types."""
    NARRATIVE = "narrative"           # Stories, literature, myths
    TECHNICAL = "technical"           # Manuals, specs, factual docs
    CONVERSATIONAL = "conversational" # Transcripts, meetings, chats
    SCIENTIFIC = "scientific"         # Papers, research, studies
    UNKNOWN = "unknown"               # Fallback

    @classmethod
    def from_string(cls, value: str) -> "DocumentType":
        """Convert string to DocumentType, case-insensitive."""
        value_lower = value.lower().strip()
        for doc_type in cls:
            if doc_type.value == value_lower:
                return doc_type
        return cls.UNKNOWN


class DocumentClassifier:
    """
    Auto-detect document type using LLM analysis.

    Analyzes the first portion of a document to determine its type,
    which is then used to select appropriate extraction strategies.

    Usage:
        classifier = DocumentClassifier(llm)
        doc_type = classifier.classify("Once upon a time...")
        print(doc_type)  # DocumentType.NARRATIVE
    """

    def __init__(self, llm, use_cache: bool = True, timeout: int = 30, debug: bool = False):
        """
        Initialize the document classifier.

        Args:
            llm: LangChain LLM instance for classification
            use_cache: Whether to cache classification results
            timeout: Timeout in seconds for LLM calls (default 30)
            debug: Whether to print debug timing info
        """
        self.llm = llm
        self.use_cache = use_cache
        self.timeout = timeout
        self.debug = debug
        self._cache: Dict[str, DocumentType] = {}

    def _get_cache_key(self, text: str, max_chars: int) -> str:
        """Generate cache key from text hash."""
        sample = text[:max_chars]
        return hashlib.md5(sample.encode()).hexdigest()

    def classify(self, text_sample: str, max_chars: int = 2000) -> DocumentType:
        """
        Classify document type from a text sample.

        Uses LLM to analyze the text and determine its type.
        Results are cached to avoid repeated LLM calls for same content.

        Args:
            text_sample: Text from the document (typically first portion)
            max_chars: Maximum characters to analyze (default 2000)

        Returns:
            DocumentType enum value
        """
        sample = text_sample[:max_chars]

        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(text_sample, max_chars)
            if cache_key in self._cache:
                return self._cache[cache_key]

        prompt = f'''Analyze this text and classify it into ONE of these categories:

- NARRATIVE: Stories, literature, myths, fiction, biographies
  (Has characters, plot, events, dialogue in a story context)

- TECHNICAL: Manuals, specifications, factual documents, reference materials
  (Has concepts, definitions, procedures, how-to instructions)

- CONVERSATIONAL: Transcripts, meetings, chats, interviews, dialogues
  (Has speakers identified, turn-taking, discussion topics)

- SCIENTIFIC: Research papers, studies, academic papers, journal articles
  (Has hypotheses, methods, findings, citations, abstract)

Text sample:
"""
{sample}
"""

Respond with ONLY the category name in uppercase (NARRATIVE, TECHNICAL, CONVERSATIONAL, or SCIENTIFIC):'''

        try:
            if self.debug:
                print(f"    [DEBUG] DocClassifier: starting LLM call...")
                t0 = time.time()

            # LLM call with timeout protection
            def llm_call():
                return self.llm.invoke(prompt)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm_call)
                try:
                    response = future.result(timeout=self.timeout)
                except FuturesTimeoutError:
                    if self.debug:
                        print(f"    [DEBUG] DocClassifier: TIMEOUT after {self.timeout}s, using UNKNOWN")
                    return DocumentType.UNKNOWN

            if self.debug:
                print(f"    [DEBUG] DocClassifier: LLM call took {time.time()-t0:.2f}s")

            # Handle different response types
            if isinstance(response, dict):
                response_text = response.get('result', '') or response.get('text', str(response))
            else:
                response_text = str(response)

            response_text = response_text.strip().upper()

            # Parse response - look for document type keywords
            result = DocumentType.UNKNOWN
            for doc_type in DocumentType:
                if doc_type == DocumentType.UNKNOWN:
                    continue
                if doc_type.value.upper() in response_text:
                    result = doc_type
                    break

            # Cache result
            if self.use_cache:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            print(f"[Classifier] Error during classification: {e}")
            return DocumentType.UNKNOWN

    def classify_with_confidence(self, text_sample: str,
                                  max_chars: int = 2000) -> Dict[str, Any]:
        """
        Classify with additional confidence information.

        Args:
            text_sample: Text from the document
            max_chars: Maximum characters to analyze

        Returns:
            Dict with 'type', 'confidence', and 'reasoning'
        """
        sample = text_sample[:max_chars]

        prompt = f'''Analyze this text and classify it. Provide your analysis in this exact format:

TYPE: [NARRATIVE|TECHNICAL|CONVERSATIONAL|SCIENTIFIC]
CONFIDENCE: [HIGH|MEDIUM|LOW]
REASONING: [Brief explanation of why you chose this type]

Categories:
- NARRATIVE: Stories, literature, myths, fiction (characters, plot, events)
- TECHNICAL: Manuals, specifications, factual docs (concepts, definitions)
- CONVERSATIONAL: Transcripts, meetings, chats (speakers, discussions)
- SCIENTIFIC: Research papers, studies (hypotheses, methods, findings)

Text sample:
"""
{sample}
"""

Analysis:'''

        try:
            response = self.llm.invoke(prompt)

            if isinstance(response, dict):
                response_text = response.get('result', '') or str(response)
            else:
                response_text = str(response)

            # Parse structured response
            result = {
                'type': DocumentType.UNKNOWN,
                'confidence': 'LOW',
                'reasoning': ''
            }

            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.upper().startswith('TYPE:'):
                    type_str = line.split(':', 1)[1].strip().upper()
                    for doc_type in DocumentType:
                        if doc_type.value.upper() in type_str:
                            result['type'] = doc_type
                            break
                elif line.upper().startswith('CONFIDENCE:'):
                    conf = line.split(':', 1)[1].strip().upper()
                    if conf in ['HIGH', 'MEDIUM', 'LOW']:
                        result['confidence'] = conf
                elif line.upper().startswith('REASONING:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()

            return result

        except Exception as e:
            print(f"[Classifier] Error: {e}")
            return {
                'type': DocumentType.UNKNOWN,
                'confidence': 'LOW',
                'reasoning': f'Classification failed: {e}'
            }

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached classifications."""
        return len(self._cache)


# Quick classification helper function
def classify_document(llm, text: str) -> DocumentType:
    """
    Quick helper to classify a document.

    Args:
        llm: LangChain LLM instance
        text: Document text (first 2000 chars used)

    Returns:
        DocumentType enum value
    """
    classifier = DocumentClassifier(llm, use_cache=False)
    return classifier.classify(text)
