"""
Augmentation utilities for experiment 010-obfuscation.

Two obfuscation strategies:
1. Homoglyph replacement: swap characters with unicode look-alikes + ZWJ insertion
2. Synonym substitution: replace words with WordNet synonyms

Both are applied to AI-labeled training texts to force the model to learn
semantic features rather than surface-level character/word patterns.
"""

import random
import re
import numpy as np
from pathlib import Path

# Set up NLTK with a project-local data directory
_NLTK_DIR = str(Path(__file__).resolve().parent / ".nltk_data")
import nltk
nltk.data.path.insert(0, _NLTK_DIR)

# Download NLTK data to project-local directory
for pkg in ("wordnet", "omw-1.4", "stopwords"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=_NLTK_DIR, quiet=True)

from nltk.corpus import wordnet, stopwords

# Fallback stop words if NLTK data is unavailable
_FALLBACK_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now",
}
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = _FALLBACK_STOP_WORDS

# Zero-width joiner character
ZWJ = "\u200d"


def apply_homoglyph(text: str, h_prob: float = 0.05, zwj_prob: float = 0.05) -> str:
    """
    Apply homoglyph augmentation to text.

    For each character:
    - With probability h_prob, replace with a visually similar unicode character
    - With probability zwj_prob, insert a zero-width joiner after the character

    Uses the `confusable_homoglyphs` library for character replacement,
    matching the mdok approach from experiment 008.
    """
    from confusable_homoglyphs import confusables

    result = []
    for char in text:
        # Skip whitespace and newlines
        if char.isspace():
            result.append(char)
            continue

        # Homoglyph swap
        if random.random() < h_prob:
            confusable = confusables.is_confusable(char, greedy=True, preferred_aliases=[])
            if confusable:
                # Pick a random confusable character
                alternatives = []
                for entry in confusable:
                    for homo in entry.get("homoglyphs", []):
                        alternatives.append(homo["c"])
                if alternatives:
                    char = random.choice(alternatives)

        result.append(char)

        # ZWJ insertion
        if random.random() < zwj_prob:
            result.append(ZWJ)

    return "".join(result)


def _get_synonyms(word: str) -> list[str]:
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)


def apply_synonym_replacement(text: str, syn_prob: float = 0.20) -> str:
    """
    Replace words with their synonyms using WordNet.

    For each word (skipping stop words, short words, and non-alphabetic tokens):
    - With probability syn_prob, replace with a random WordNet synonym

    This simulates paraphrasing attacks that preserve meaning but change
    surface-level word choices.
    """
    # Simple word tokenization preserving punctuation
    tokens = re.findall(r"\b\w+\b|[^\w\s]|\s+", text)

    result = []
    for token in tokens:
        # Only consider replacing actual words (alphabetic, not stop words, length > 3)
        if (
            token.isalpha()
            and token.lower() not in STOP_WORDS
            and len(token) > 3
            and random.random() < syn_prob
        ):
            synonyms = _get_synonyms(token.lower())
            if synonyms:
                replacement = random.choice(synonyms)
                # Preserve original capitalization
                if token[0].isupper():
                    replacement = replacement.capitalize()
                if token.isupper():
                    replacement = replacement.upper()
                result.append(replacement)
                continue

        result.append(token)

    return "".join(result)


def augment_dataframe(
    df,
    homoglyph_frac: float = 0.10,
    synonym_frac: float = 0.10,
    h_prob: float = 0.05,
    zwj_prob: float = 0.05,
    syn_prob: float = 0.20,
    seed: int = 42,
):
    """
    Augment a DataFrame by applying obfuscation to AI-labeled texts.

    Selects non-overlapping subsets of AI-labeled texts:
    - homoglyph_frac of AI texts get homoglyph augmentation
    - synonym_frac of AI texts get synonym replacement

    The augmented texts replace the originals in-place.

    Args:
        df: DataFrame with 'text' and 'label' columns (label=1 for AI)
        homoglyph_frac: fraction of AI texts for homoglyph augmentation
        synonym_frac: fraction of AI texts for synonym replacement
        h_prob: per-character homoglyph swap probability
        zwj_prob: per-character ZWJ insertion probability
        syn_prob: per-word synonym replacement probability
        seed: random seed for reproducibility

    Returns:
        Augmented DataFrame (modified in-place, also returned for convenience)
    """
    rng = np.random.RandomState(seed)

    # Get indices of AI-labeled samples
    ai_indices = df[df["label"] == 1].index.tolist()
    n_ai = len(ai_indices)

    # Shuffle and split into non-overlapping subsets
    shuffled = rng.permutation(ai_indices)
    n_homoglyph = int(n_ai * homoglyph_frac)
    n_synonym = int(n_ai * synonym_frac)

    homoglyph_indices = shuffled[:n_homoglyph]
    synonym_indices = shuffled[n_homoglyph : n_homoglyph + n_synonym]

    print(f"Augmenting {n_homoglyph} AI texts with homoglyphs "
          f"({homoglyph_frac*100:.0f}% of {n_ai} AI samples)")
    print(f"Augmenting {n_synonym} AI texts with synonyms "
          f"({synonym_frac*100:.0f}% of {n_ai} AI samples)")

    # Apply homoglyph augmentation
    random.seed(seed)
    for idx in homoglyph_indices:
        df.at[idx, "text"] = apply_homoglyph(df.at[idx, "text"], h_prob, zwj_prob)

    # Apply synonym replacement
    random.seed(seed + 1)
    for idx in synonym_indices:
        df.at[idx, "text"] = apply_synonym_replacement(df.at[idx, "text"], syn_prob)

    print(f"Augmentation complete. Total augmented: {n_homoglyph + n_synonym}")
    return df
