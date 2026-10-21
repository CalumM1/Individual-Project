import pandas as pd


def significant_word_predictor(topics) -> pd.DataFrame:
    """
    Hypothesis: queries with a greater ratio of 'significant' to 'insignificant' words are more performant 
    """
    insignificant_words = [
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "of", "to", "in", "on", "at", "by", "for", "from", "with",
    "about", "as", "into", "over", "under", "between", "through",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "shall", "should", "may",
    "might", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "where", "when",
    "why", "how",
    "not", "no", "nor", "so", "too", "very", "just", "also",
    "only", "own", "same", "than", "such", "both", "each",
    "every", "all", "any", "few", "more", "most", "other",
    "some", "once", "here", "there",
    "up", "down", "out", "off", "again", "further",
    "because", "while", "although", "though", "until", "unless",
    "after", "before", "during", "as", "per", "via"
    ]
    
    def score(query):
        significant = 0
        insignificant = 0
        for word in query.split():
            if word.lower() in insignificant_words:
                insignificant += 1
            else:
                significant += 1
        if insignificant == 0:
            return significant
        return significant / (insignificant + 1)  # + 1 prevents ties when there is only one insig word

    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["query"].apply(score),
    })