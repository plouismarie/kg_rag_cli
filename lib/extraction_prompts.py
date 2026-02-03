"""
Type-Specific Extraction Prompts

Different relationship schemas and extraction prompts for each document type.
Each type focuses on extracting the most meaningful relationships for that domain.

Document Types:
- NARRATIVE: Characters, actions, events, emotions
- TECHNICAL: Definitions, hierarchies, causation
- CONVERSATIONAL: Speakers, decisions, actions
- SCIENTIFIC: Hypotheses, methods, findings, citations

Modes:
- Standard: Strict entity filtering (proper nouns only, 1-3 words)
- Verbose: Comprehensive extraction (descriptive phrases, concepts, all entities)
"""
from typing import Dict, List
from .doc_classifier import DocumentType


# ============================================================================
# EXTRACTION PROMPTS
# ============================================================================

EXTRACTION_PROMPTS: Dict[DocumentType, str] = {

    DocumentType.NARRATIVE: '''Extract knowledge graph triples from this narrative text.

ENTITY CONSTRAINTS:
- Extract ONLY core entities: proper nouns (names of people, places) and important objects/concepts
- Keep entities short: 1-3 words maximum (e.g., "George Hadley", not "George Hadley's concern")
- DO NOT extract: pronouns (He, She, They), generic terms (Father, Speaker, Person)
- DO NOT extract: full sentences or clauses as entities
- Focus on: WHO (characters), WHERE (locations), WHAT (objects, events)

Focus on extracting:
- Character relationships (FATHER_OF, MOTHER_OF, MARRIED_TO, SIBLING_OF, FRIEND_OF, ENEMY_OF)
- Actions between characters (HELPS, FIGHTS, BETRAYS, SAVES, LOVES, HATES)
- Character-location relationships (LIVES_IN, TRAVELS_TO, RETURNS_TO, RULES, BORN_IN)
- Events and plot (CAUSES, LEADS_TO, RESULTS_IN, HAPPENS_BEFORE, HAPPENS_AFTER)
- Character attributes (IS_A, HAS_TRAIT, KNOWN_FOR, SKILLED_IN)
- Motivations (DESIRES, FEARS, SEEKS, PROTECTS)

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract concrete, factual relationships only.

Text:
{text}

Triples:''',

    DocumentType.TECHNICAL: '''Extract knowledge graph triples from this technical text.

ENTITY CONSTRAINTS:
- Extract ONLY: technical terms, component names, system names, concepts
- Keep entities short: 1-3 words maximum
- DO NOT extract: pronouns, full descriptions, generic terms
- Focus on: WHAT (components, systems), HOW (processes, methods)

Focus on extracting:
- Definitions (IS_A, IS_TYPE_OF, IS_DEFINED_AS, IS_ALSO_KNOWN_AS)
- Hierarchies (PART_OF, CONTAINS, BELONGS_TO, INCLUDES, COMPONENT_OF)
- Properties (HAS_PROPERTY, HAS_VALUE, HAS_ATTRIBUTE, MEASURES)
- Causation (CAUSES, ENABLES, REQUIRES, DEPENDS_ON, PREVENTS)
- Procedures (FOLLOWS, PRECEDES, STEP_IN, USED_FOR, USED_BY)
- Comparisons (SIMILAR_TO, DIFFERENT_FROM, ALTERNATIVE_TO, REPLACES)

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract concrete, factual relationships only.

Text:
{text}

Triples:''',

    DocumentType.CONVERSATIONAL: '''Extract knowledge graph triples from this conversation/transcript.

ENTITY CONSTRAINTS:
- Extract ONLY: speaker names, topics, decisions, action items
- Keep entities short: 1-3 words maximum
- DO NOT extract: pronouns, generic terms (Someone, Anyone)
- Focus on: WHO (speakers), WHAT (topics, decisions)

Focus on extracting:
- Speaker-topic relationships (DISCUSSES, MENTIONS, ASKS_ABOUT, EXPLAINS, CLARIFIES)
- Decisions (DECIDES, AGREES_TO, DISAGREES_WITH, PROPOSES, REJECTS, APPROVES)
- Action items (ASSIGNED_TO, RESPONSIBLE_FOR, WILL_DO, COMMITTED_TO, DEADLINE_FOR)
- References (REFERS_TO, CITES, RESPONDS_TO, FOLLOWS_UP_ON, RELATED_TO)
- Opinions (SUPPORTS, OPPOSES, QUESTIONS, CONCERNS_ABOUT, RECOMMENDS)
- Participants (ATTENDED_BY, LED_BY, PRESENTED_BY, INVITED)

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Include speaker names when clear from context.

Text:
{text}

Triples:''',

    DocumentType.SCIENTIFIC: '''Extract knowledge graph triples from this scientific text.

ENTITY CONSTRAINTS:
- Extract ONLY: author names, study names, concepts, methods, findings
- Keep entities short: 1-3 words maximum
- DO NOT extract: pronouns, full descriptions
- Focus on: WHO (researchers), WHAT (findings, methods)

Focus on extracting:
- Claims (HYPOTHESIZES, CONCLUDES, PROVES, DISPROVES, SUGGESTS, ARGUES)
- Methods (USES_METHOD, MEASURES, ANALYZES, TESTS, EVALUATES, COMPARES)
- Findings (FINDS, OBSERVES, DISCOVERS, SHOWS, DEMONSTRATES, REVEALS)
- Correlations (CORRELATES_WITH, ASSOCIATED_WITH, AFFECTS, INFLUENCES)
- Citations (CITES, EXTENDS, CONTRADICTS, SUPPORTS, BUILDS_ON, REFERENCES)
- Concepts (DEFINES, INTRODUCES, PROPOSES, MODELS, REPRESENTS)

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Include author names and study references when available.

Text:
{text}

Triples:''',

    DocumentType.UNKNOWN: '''Extract knowledge graph triples from this text.
The text may contain mixed content types. Extract ALL relevant relationships.

ENTITY CONSTRAINTS:
- Extract ONLY core entities: proper nouns, important objects/concepts
- Keep entities short: 1-3 words maximum
- DO NOT extract: pronouns (He, She), generic terms (Person, Thing)
- DO NOT extract: full sentences or clauses as entities
- Focus on concrete, named entities only

CHARACTER/NARRATIVE relationships:
- Family: FATHER_OF, MOTHER_OF, SIBLING_OF, MARRIED_TO, CHILD_OF
- Actions: HELPS, FIGHTS, LOVES, BETRAYS, SAVES, KILLS
- Movement: JOURNEYS_TO, LIVES_IN, RETURNS_TO, TRAVELS_TO, RULES

TECHNICAL relationships:
- Hierarchy: IS_A, PART_OF, CONTAINS, BELONGS_TO, COMPONENT_OF
- Causation: CAUSES, ENABLES, REQUIRES, DEPENDS_ON, PREVENTS
- Properties: HAS_PROPERTY, HAS_VALUE, HAS_ATTRIBUTE, MEASURES

CONVERSATIONAL relationships:
- Discussion: DISCUSSES, MENTIONS, PROPOSES, EXPLAINS, CLARIFIES
- Decisions: DECIDES, AGREES_TO, ASSIGNED_TO, RESPONSIBLE_FOR

SCIENTIFIC relationships:
- Research: HYPOTHESIZES, PROVES, CITES, FINDS, DISCOVERS
- Methods: USES_METHOD, ANALYZES, TESTS, MEASURES

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Use the most appropriate relationship type for each fact.
Extract concrete, factual relationships only.

Text:
{text}

Triples:'''
}


# ============================================================================
# VERBOSE EXTRACTION PROMPTS (for comprehensive/verbose mode)
# ============================================================================

VERBOSE_EXTRACTION_PROMPTS: Dict[DocumentType, str] = {

    DocumentType.NARRATIVE: '''Extract ALL knowledge graph triples from this narrative text.

Be VERBOSE and COMPREHENSIVE:
- Extract EVERY meaningful entity: characters, objects, locations, concepts, descriptions
- Include descriptive phrases as entities (e.g., "terrible green-yellow eyes", "mechanical genius")
- Include emotional states, actions, sounds, and sensory details
- Do NOT limit entity length - capture the full context
- Do NOT skip abstract concepts or descriptive elements

ENTITY TYPES TO EXTRACT:
- Characters (names, titles, roles, descriptions of characters)
- Locations (rooms, places, settings, environmental details)
- Objects (machines, items, animals, physical things)
- Concepts (ideas, concerns, feelings, abstract notions)
- Descriptions (visual details, sounds, smells, textures)
- Events (actions, occurrences, happenings)
- Quotes and exclamations ("Watch out!", screams, dialogue)
- Attributes (qualities, characteristics, states)

RELATIONSHIP TYPES:
- Physical: CONTAINS, LOCATED_IN, PART_OF, NEAR, INSIDE
- Character: FATHER_OF, MOTHER_OF, MARRIED_TO, PARENT_OF, CHILD_OF
- Action: DOES, SAYS, HEARS, SEES, FEELS, THINKS, WANTS
- Descriptive: HAS_FEATURE, DESCRIBED_AS, CHARACTERIZED_BY, LOOKS_LIKE
- Emotional: FEARS, WORRIES_ABOUT, CONCERNED_ABOUT, LOVES, HATES, FEELS
- Causal: CAUSES, LEADS_TO, RESULTS_IN, TRIGGERS, PRODUCES
- Temporal: HAPPENS_BEFORE, HAPPENS_AFTER, DURING, WHILE
- Possession: OWNS, HAS, POSSESSES, BELONGS_TO

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract as many triples as possible - more is better.
Include every detail mentioned in the text.

Text:
{text}

Triples:''',

    DocumentType.TECHNICAL: '''Extract ALL knowledge graph triples from this technical text.

Be VERBOSE and COMPREHENSIVE:
- Extract EVERY technical term, component, concept, and relationship
- Include detailed descriptions and specifications
- Do NOT limit entity length
- Include all properties, values, and attributes mentioned

ENTITY TYPES TO EXTRACT:
- Components (systems, subsystems, parts, modules)
- Concepts (definitions, terms, ideas)
- Properties (attributes, values, measurements)
- Processes (procedures, methods, steps)
- Relationships (dependencies, connections)
- Specifications (requirements, constraints)

RELATIONSHIP TYPES:
- Hierarchy: IS_A, PART_OF, CONTAINS, BELONGS_TO, COMPONENT_OF, SUBSYSTEM_OF
- Definition: IS_DEFINED_AS, MEANS, REFERS_TO, KNOWN_AS
- Properties: HAS_PROPERTY, HAS_VALUE, MEASURES, SPECIFIES
- Causation: CAUSES, ENABLES, REQUIRES, DEPENDS_ON, PREVENTS, BLOCKS
- Procedures: FOLLOWS, PRECEDES, STEP_IN, USED_FOR, USED_BY, IMPLEMENTS
- Comparison: SIMILAR_TO, DIFFERENT_FROM, ALTERNATIVE_TO, REPLACES

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract as many triples as possible.

Text:
{text}

Triples:''',

    DocumentType.CONVERSATIONAL: '''Extract ALL knowledge graph triples from this conversation/transcript.

Be VERBOSE and COMPREHENSIVE:
- Extract EVERY speaker, topic, statement, and relationship
- Include all opinions, decisions, and action items
- Do NOT limit entity length
- Capture the full context of discussions

ENTITY TYPES TO EXTRACT:
- Speakers (names, roles, titles)
- Topics (subjects discussed, themes)
- Decisions (conclusions, agreements, resolutions)
- Action items (tasks, responsibilities, commitments)
- Opinions (views, positions, arguments)
- Questions (inquiries, concerns raised)
- References (documents, people, events mentioned)

RELATIONSHIP TYPES:
- Discussion: DISCUSSES, MENTIONS, ASKS_ABOUT, EXPLAINS, CLARIFIES, DESCRIBES
- Decisions: DECIDES, AGREES_TO, DISAGREES_WITH, PROPOSES, REJECTS, APPROVES
- Actions: ASSIGNED_TO, RESPONSIBLE_FOR, WILL_DO, COMMITTED_TO, PROMISED
- References: REFERS_TO, CITES, RESPONDS_TO, FOLLOWS_UP_ON, RELATED_TO
- Opinions: SUPPORTS, OPPOSES, QUESTIONS, CONCERNS_ABOUT, RECOMMENDS, SUGGESTS
- Attribution: SAID_BY, STATED_BY, CLAIMED_BY, ASKED_BY

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract as many triples as possible.

Text:
{text}

Triples:''',

    DocumentType.SCIENTIFIC: '''Extract ALL knowledge graph triples from this scientific text.

Be VERBOSE and COMPREHENSIVE:
- Extract EVERY author, study, method, finding, and concept
- Include all citations, correlations, and conclusions
- Do NOT limit entity length
- Capture all experimental details and results

ENTITY TYPES TO EXTRACT:
- Authors (researchers, scientists, institutions)
- Studies (papers, experiments, trials)
- Methods (techniques, procedures, approaches)
- Findings (results, observations, discoveries)
- Concepts (theories, hypotheses, models)
- Data (measurements, statistics, values)
- Citations (references, prior work)

RELATIONSHIP TYPES:
- Claims: HYPOTHESIZES, CONCLUDES, PROVES, DISPROVES, SUGGESTS, ARGUES, CLAIMS
- Methods: USES_METHOD, MEASURES, ANALYZES, TESTS, EVALUATES, COMPARES, APPLIES
- Findings: FINDS, OBSERVES, DISCOVERS, SHOWS, DEMONSTRATES, REVEALS, INDICATES
- Correlations: CORRELATES_WITH, ASSOCIATED_WITH, AFFECTS, INFLUENCES, PREDICTS
- Citations: CITES, EXTENDS, CONTRADICTS, SUPPORTS, BUILDS_ON, REFERENCES, REPLICATES
- Concepts: DEFINES, INTRODUCES, PROPOSES, MODELS, REPRESENTS, DESCRIBES

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract as many triples as possible.

Text:
{text}

Triples:''',

    DocumentType.UNKNOWN: '''Extract ALL knowledge graph triples from this text.

Be VERBOSE and COMPREHENSIVE:
- Extract EVERY entity, concept, and relationship mentioned
- Include descriptive phrases and abstract concepts
- Do NOT limit entity length
- Capture all details, no matter how small

ENTITY TYPES TO EXTRACT:
- People (names, roles, descriptions)
- Places (locations, settings, environments)
- Things (objects, items, entities)
- Concepts (ideas, themes, abstractions)
- Events (actions, occurrences, happenings)
- Descriptions (qualities, attributes, characteristics)
- Quotes (statements, exclamations, dialogue)

ALL RELATIONSHIP TYPES:
- Physical: CONTAINS, LOCATED_IN, PART_OF, NEAR
- Personal: FATHER_OF, MARRIED_TO, KNOWS, WORKS_WITH
- Action: DOES, SAYS, SEES, FEELS, THINKS
- Descriptive: HAS_FEATURE, DESCRIBED_AS, IS_A
- Causal: CAUSES, LEADS_TO, RESULTS_IN
- Temporal: HAPPENS_BEFORE, HAPPENS_AFTER, DURING

Format each triple as: (Subject, RELATIONSHIP, Object)
Use UPPERCASE for relationship names.
Extract as many triples as possible.

Text:
{text}

Triples:'''
}


# ============================================================================
# INVERSE RELATIONSHIP MAPPINGS
# ============================================================================

INVERSE_MAPPINGS: Dict[DocumentType, Dict[str, str]] = {

    DocumentType.NARRATIVE: {
        # Family
        'FATHER_OF': 'CHILD_OF',
        'MOTHER_OF': 'CHILD_OF',
        'CHILD_OF': 'PARENT_OF',
        'PARENT_OF': 'CHILD_OF',
        'SIBLING_OF': 'SIBLING_OF',  # symmetric
        'MARRIED_TO': 'MARRIED_TO',  # symmetric

        # Actions
        'HELPS': 'HELPED_BY',
        'FIGHTS': 'FIGHTS',  # symmetric
        'SAVES': 'SAVED_BY',
        'BETRAYS': 'BETRAYED_BY',
        'LOVES': 'LOVED_BY',
        'HATES': 'HATED_BY',

        # Location
        'LIVES_IN': 'HOME_OF',
        'TRAVELS_TO': 'VISITED_BY',
        'RULES': 'RULED_BY',
        'BORN_IN': 'BIRTHPLACE_OF',

        # Events
        'CAUSES': 'CAUSED_BY',
        'LEADS_TO': 'RESULT_OF',
    },

    DocumentType.TECHNICAL: {
        # Hierarchy
        'PART_OF': 'CONTAINS',
        'CONTAINS': 'PART_OF',
        'BELONGS_TO': 'HAS_MEMBER',
        'COMPONENT_OF': 'HAS_COMPONENT',

        # Definition
        'IS_A': 'EXAMPLE_OF',
        'IS_TYPE_OF': 'HAS_TYPE',

        # Causation
        'CAUSES': 'CAUSED_BY',
        'ENABLES': 'ENABLED_BY',
        'REQUIRES': 'REQUIRED_BY',
        'DEPENDS_ON': 'DEPENDENCY_OF',
        'PREVENTS': 'PREVENTED_BY',

        # Procedures
        'FOLLOWS': 'PRECEDES',
        'PRECEDES': 'FOLLOWS',
        'USED_FOR': 'USES',
        'USED_BY': 'USES',
    },

    DocumentType.CONVERSATIONAL: {
        # Speaker actions
        'DISCUSSES': 'DISCUSSED_BY',
        'MENTIONS': 'MENTIONED_BY',
        'PROPOSES': 'PROPOSED_BY',
        'DECIDES': 'DECIDED_BY',

        # Assignments
        'ASSIGNED_TO': 'ASSIGNED_BY',
        'RESPONSIBLE_FOR': 'RESPONSIBILITY_OF',

        # References
        'RESPONDS_TO': 'RESPONSE_FROM',
        'REFERS_TO': 'REFERENCED_BY',
    },

    DocumentType.SCIENTIFIC: {
        # Claims
        'PROVES': 'PROVEN_BY',
        'DISPROVES': 'DISPROVEN_BY',
        'HYPOTHESIZES': 'HYPOTHESIS_OF',
        'CONCLUDES': 'CONCLUSION_OF',

        # Methods
        'USES_METHOD': 'METHOD_USED_BY',
        'ANALYZES': 'ANALYZED_BY',

        # Findings
        'FINDS': 'FOUND_BY',
        'DISCOVERS': 'DISCOVERED_BY',
        'SHOWS': 'SHOWN_BY',

        # Citations
        'CITES': 'CITED_BY',
        'SUPPORTS': 'SUPPORTED_BY',
        'CONTRADICTS': 'CONTRADICTED_BY',
        'EXTENDS': 'EXTENDED_BY',
    },

    DocumentType.UNKNOWN: {
        # Generic fallbacks
        'RELATES_TO': 'RELATES_TO',
        'CONNECTED_TO': 'CONNECTED_TO',
    }
}


# ============================================================================
# COMMON INVERSE MAPPINGS (shared across all types)
# ============================================================================

COMMON_INVERSES: Dict[str, str] = {
    # Symmetric relationships
    'RELATED_TO': 'RELATED_TO',
    'SIMILAR_TO': 'SIMILAR_TO',
    'DIFFERENT_FROM': 'DIFFERENT_FROM',
    'CONNECTED_TO': 'CONNECTED_TO',
    'ASSOCIATED_WITH': 'ASSOCIATED_WITH',

    # Common directional
    'HAS': 'BELONGS_TO',
    'OWNS': 'OWNED_BY',
    'CREATES': 'CREATED_BY',
    'PRODUCES': 'PRODUCED_BY',
    'CAUSES': 'CAUSED_BY',
}


def get_extraction_prompt(doc_type: DocumentType, verbose: bool = False) -> str:
    """
    Get the extraction prompt for a document type.

    Args:
        doc_type: DocumentType enum value
        verbose: If True, use verbose prompts that extract more entities/relationships

    Returns:
        Extraction prompt template string
    """
    if verbose:
        return VERBOSE_EXTRACTION_PROMPTS.get(doc_type, VERBOSE_EXTRACTION_PROMPTS[DocumentType.NARRATIVE])
    return EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS[DocumentType.UNKNOWN])


def get_inverse_mapping(doc_type: DocumentType) -> Dict[str, str]:
    """
    Get inverse relationship mapping for a document type.

    Args:
        doc_type: DocumentType enum value

    Returns:
        Dict mapping relationship -> inverse relationship
    """
    # Combine common inverses with type-specific
    mapping = COMMON_INVERSES.copy()
    type_mapping = INVERSE_MAPPINGS.get(doc_type, {})
    mapping.update(type_mapping)
    return mapping


def get_inverse_relation(relation: str, doc_type: DocumentType) -> str:
    """
    Get the inverse of a relationship for a given document type.

    Args:
        relation: Relationship name (e.g., 'FATHER_OF')
        doc_type: Document type for context

    Returns:
        Inverse relationship name, or 'INVERSE_OF_{relation}' if unknown
    """
    mapping = get_inverse_mapping(doc_type)
    relation_upper = relation.upper()

    if relation_upper in mapping:
        return mapping[relation_upper]

    # Fallback: generate inverse name
    return f"INVERSE_OF_{relation_upper}"


def get_relationship_examples(doc_type: DocumentType) -> List[str]:
    """
    Get example relationships for a document type.

    Useful for debugging and understanding what each type extracts.

    Args:
        doc_type: DocumentType enum value

    Returns:
        List of example relationship names
    """
    examples = {
        DocumentType.NARRATIVE: [
            'FATHER_OF', 'LOVES', 'FIGHTS', 'JOURNEYS_TO', 'SAVES', 'BETRAYS'
        ],
        DocumentType.TECHNICAL: [
            'IS_A', 'PART_OF', 'CAUSES', 'REQUIRES', 'CONTAINS', 'USED_FOR'
        ],
        DocumentType.CONVERSATIONAL: [
            'DISCUSSES', 'DECIDES', 'ASSIGNED_TO', 'PROPOSES', 'AGREES_TO'
        ],
        DocumentType.SCIENTIFIC: [
            'HYPOTHESIZES', 'PROVES', 'CITES', 'FINDS', 'USES_METHOD', 'CORRELATES_WITH'
        ],
        DocumentType.UNKNOWN: [
            'RELATES_TO', 'CONNECTED_TO', 'HAS', 'IS_A'
        ]
    }
    return examples.get(doc_type, examples[DocumentType.UNKNOWN])
