VALID_VARIANTS = (*tuple(f"s{i}" for i in range(1, 14)), "center_dynamic")
UNTIED_VARIANTS = frozenset({"s2", "s8", "s9", "s10"})


def variant_tie_word_embeddings(variant):
    return variant.lower() not in UNTIED_VARIANTS
