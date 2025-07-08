import re

def router(user_query):
    """
    Routes user queries to either the recommender system or semantic search based on keyword detection.
    Args:
        user_query: The user's input query
    Returns:
        Dictionary containing route, query, and matched keywords
    """
    normalized_query = user_query.lower().strip()
    
    recommender_keywords = [
        "recommend", "suggest", "advise", "propose",
        "find me", "show me", "give me", "i want", "i need",
        "looking for", "search for", "help me find",
        "portfolio", "investment", "stocks for", "best stocks",
        "top stocks", "good stocks", "suitable stocks",
        "better than", "alternatives to", "similar to",
        "should i", "what to", "which stocks"
    ]
    
    recommender_patterns = [
        r"\b(recommend|suggest|advise|propose)\b",
        r"\b(find|show|give)\s+me\b",
        r"\bi\s+(want|need|looking)\b",
        r"\b(best|top|good)\s+stocks?\b",
        r"\bstocks?\s+for\b",
        r"\bportfolio\b",
        r"\bshould\s+i\b",
        r"\bwhat\s+to\b",
        r"\bwhich\s+stocks?\b"
    ]
    
    matched_keywords = []
    pattern_matches = []
    
    for keyword in recommender_keywords:
        if keyword in normalized_query:
            matched_keywords.append(keyword)
    
    for pattern in recommender_patterns:
        if re.search(pattern, normalized_query):
            pattern_matches.append(pattern)
    
    total_matches = len(matched_keywords) + len(pattern_matches)
    
    if total_matches > 0:
        route = "recommender"
    else:
        route = "semantic_search"
    
    factual_patterns = [
        r"what\s+is\s+the\s+\w+\s+of\s+[A-Z]+",
        r"[A-Z]{1,5}\s+(price|return|yield|ratio)",
        r"show\s+me\s+[A-Z]{1,5}",
    ]
    
    for pattern in factual_patterns:
        if re.search(pattern, user_query):
            route = "semantic_search"
            matched_keywords = ["factual_query_pattern"]
            break
    
    return {
        "route": route,
        "query": user_query,
        "matched_keywords": matched_keywords,
        "pattern_matches": pattern_matches
    }