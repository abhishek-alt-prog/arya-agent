"""
Year 3 (UK) / Grade 2 (US) curriculum skeleton.
Defines the topics, units, and learning objectives for a 7-year-old.
"""
from __future__ import annotations

from .models import Subject

# ─────────────────────────────────────────────────────────────────────
# Each subject is a list of units. Each unit has a name and topics.
# Topics are ordered by increasing difficulty within a unit.
# ─────────────────────────────────────────────────────────────────────

CURRICULUM: dict[Subject, list[dict]] = {
    Subject.SCIENCE: [
        {
            "name": "Plants",
            "topics": [
                "Parts of a plant",
                "What plants need to grow",
                "How water travels through plants",
                "Life cycle of a flowering plant",
                "Seeds and how they spread",
            ],
        },
        {
            "name": "Animals Including Humans",
            "topics": [
                "What animals eat — herbivore, carnivore, omnivore",
                "Skeletons and bones",
                "Muscles and movement",
                "The five senses",
                "Staying healthy — exercise and food",
            ],
        },
        {
            "name": "Rocks and Soils",
            "topics": [
                "Different types of rocks",
                "How rocks are formed",
                "Comparing rocks — hard vs soft",
                "What is soil made of?",
                "Fossils — clues from the past",
            ],
        },
        {
            "name": "Light and Shadows",
            "topics": [
                "Sources of light",
                "Light and dark",
                "How shadows are made",
                "Changing the size of shadows",
                "Reflections and mirrors",
            ],
        },
        {
            "name": "Forces and Magnets",
            "topics": [
                "Pushes and pulls",
                "What are magnets?",
                "Magnetic and non-magnetic materials",
                "Poles of a magnet — attract and repel",
                "Everyday uses of magnets",
            ],
        },
    ],
    Subject.MATHS: [
        {
            "name": "Number and Place Value",
            "topics": [
                "Counting in 2s, 5s, and 10s",
                "Hundreds, tens, and ones",
                "Comparing and ordering numbers to 1000",
                "Number lines and sequences",
                "Reading and writing numbers in words",
            ],
        },
        {
            "name": "Addition and Subtraction",
            "topics": [
                "Adding two-digit numbers",
                "Subtracting two-digit numbers",
                "Column addition",
                "Column subtraction with borrowing",
                "Word problems — addition and subtraction",
            ],
        },
        {
            "name": "Multiplication and Division",
            "topics": [
                "The 2, 5, and 10 times tables",
                "The 3 and 4 times tables",
                "Multiplying by 1 and 0",
                "Sharing equally — intro to division",
                "Word problems — multiply and divide",
            ],
        },
        {
            "name": "Fractions",
            "topics": [
                "What is a fraction?",
                "Halves, quarters, and thirds",
                "Finding fractions of amounts",
                "Comparing simple fractions",
                "Fractions on a number line",
            ],
        },
        {
            "name": "Measurement",
            "topics": [
                "Measuring length — cm and m",
                "Measuring mass — g and kg",
                "Measuring capacity — ml and l",
                "Telling the time — hours and minutes",
                "Money — adding coins and giving change",
            ],
        },
        {
            "name": "Geometry",
            "topics": [
                "2D shapes — sides and corners",
                "3D shapes — faces, edges, vertices",
                "Lines of symmetry",
                "Turns — quarter, half, three-quarter, full",
                "Position and direction — left, right, above, below",
            ],
        },
    ],
    Subject.ENGLISH: [
        {
            "name": "Reading Comprehension",
            "topics": [
                "Reading a short story and answering questions",
                "Finding the main idea",
                "Understanding characters and feelings",
                "Predicting what happens next",
                "Non-fiction — finding facts in a text",
            ],
        },
        {
            "name": "Phonics and Spelling",
            "topics": [
                "Common spelling patterns — igh, tion, ous",
                "Silent letters — kn, wr, gn",
                "Homophones — there, their, they're",
                "Prefixes — un, dis, re",
                "Suffixes — ly, ful, less, ness",
            ],
        },
        {
            "name": "Grammar",
            "topics": [
                "Nouns, verbs, and adjectives",
                "Using capital letters and full stops",
                "Question marks and exclamation marks",
                "Conjunctions — and, but, because, so",
                "Past tense and present tense",
            ],
        },
        {
            "name": "Writing",
            "topics": [
                "Writing a simple sentence",
                "Planning a story — beginning, middle, end",
                "Describing words — adjectives and adverbs",
                "Writing instructions",
                "Writing a letter or postcard",
            ],
        },
        {
            "name": "Vocabulary",
            "topics": [
                "Synonyms — words that mean the same",
                "Antonyms — words that are opposite",
                "Compound words",
                "Using a dictionary",
                "Word families",
            ],
        },
    ],
}


def get_all_subjects() -> list[Subject]:
    """Return all subjects in the curriculum."""
    return list(CURRICULUM.keys())


def get_units_for_subject(subject: Subject) -> list[dict]:
    """Return all units for a given subject."""
    return CURRICULUM.get(subject, [])


def get_total_topics() -> int:
    """Return the total number of topics across all subjects."""
    total = 0
    for units in CURRICULUM.values():
        for unit in units:
            total += len(unit["topics"])
    return total
