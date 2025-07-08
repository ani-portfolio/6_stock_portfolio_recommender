import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from sentence_transformers import SentenceTransformer
import json
import re

def get_api_key(key_name):
    """
    Get API key from environment variables or Streamlit secrets.
    Args:
        key_name: Name of the API key
    Returns:
        API key value or empty string if not found
    """
    env_value = os.getenv(key_name)
    if env_value:
        return env_value
    
    try:
        return st.secrets.get(key_name, "")
    except:
        return ""
    
def extract_preferences_with_groq(query, all_columns, numerical_columns, non_numerical_columns, groq_client):
    """
    Extract user preferences from natural language using Groq LLM.
    Args:
        query: Natural language query string
        all_columns: List of all available columns
        numerical_columns: List of numerical feature columns
        non_numerical_columns: List of non-numerical columns
        groq_client: Groq client instance
    Returns:
        Dictionary containing extracted preferences
    """
    categorical_columns = [col for col in non_numerical_columns 
                          if col.lower() in ['sector', 'industry', 'country', 'company_name', 'ticker']]

    categorical_filters = {}
    for col in categorical_columns:
        if col.lower() in ['sector', 'industry', 'country']:
            categorical_filters[col] = "value or null"
    
    prompt = f"""
    Analyze this stock investment query and extract preferences: "{query}"
    
    Available stock data columns: {all_columns}
    Available numerical columns for analysis: {numerical_columns}
    Available categorical columns for filtering: {non_numerical_columns}
    
    Extract preferences and return ONLY a valid JSON object with this exact structure:
    {{
        "categorical_filters": {json.dumps(categorical_filters, indent=12)},

        "numerical_preferences": {{
            "risk_level": "low/medium/high or null",
            "return_preference": "low/medium/high or null",
            "sentiment_preference": "extreme_negative/negative/neutral/positve/extreme_positive or null",
            "market_cap_preference": "small/medium/large or null",
            "dividend_preference": "low/medium/high or null",
            "volatility_preference": "low/medium/high or null",
            "growth_preference": "low/medium/high or null",
            "valuation_preference": "undervalued/fairly_valued/overvalued or null"
        }},

        "feature_weights": {{}},

        "investment_style": "value_investing/growth_investing/dividend_investing/momentum_investing/income_investing or null",
        
        "time_horizon": "short_term/medium_term/long_term or null"
    }}
    
    Guidelines:
    - Only extract explicitly mentioned preferences from the query
    - Use null for unmentioned criteria
    - For categorical_filters: match query terms to available categorical columns
    - For numerical_preferences: interpret investment terms (low risk, high return, etc.)
    - For feature_weights: assign higher weights (0.7-1.0) to numerical columns that are important for the query
    - Risk preferences: low risk = prefer stability, lower volatility
    - Return preferences: high return = prefer higher growth/returns
    - Be conservative - only extract clear, explicit preferences
    - Do not include any explanatory text, only return the JSON object
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a financial analyst. Extract investment preferences from queries and return only valid JSON. No explanations, just JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            preferences = json.loads(json_str)
            return preferences
        else:
            print("No valid JSON found in Groq response")
            print(f"Response content: {content}")
            return {}
            
    except Exception as e:
        print(f"Error extracting preferences with Groq: {e}")
        return {}

def filter_stocks_by_categories(df, preferences):
    """
    Filter stocks based on categorical preferences.
    Args:
        df: DataFrame containing stock data
        preferences: Dictionary containing user preferences
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    categorical_filters = preferences.get('categorical_filters', {})
    
    for column, value in categorical_filters.items():
        if value and value.lower() != 'null' and column in df.columns:
            mask = filtered_df[column].str.contains(value, case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Filtered by {column}='{value}': {len(filtered_df)} stocks remaining")
    
    return filtered_df

def create_user_preference_vector_with_embeddings(user_query, preferences, numerical_columns, embedding_model):
    """
    Create user preference vector using finance embeddings and extracted preferences.
    Args:
        user_query: Original user query string
        preferences: Dictionary containing extracted preferences
        numerical_columns: List of numerical feature columns
        embedding_model: SentenceTransformer embedding model
    Returns:
        Numpy array representing user preference vector
    """
    financial_contexts = []
    numerical_prefs = preferences.get('numerical_preferences', {})
    investment_style = preferences.get('investment_style')
    time_horizon = preferences.get('time_horizon')
    
    base_context = f"Investment query: {user_query}"
    financial_contexts.append(base_context)
    
    for pref_type, pref_value in numerical_prefs.items():
        if pref_value and pref_value.lower() != 'null':
            context = f"{pref_type.replace('_', ' ')}: {pref_value}"
            financial_contexts.append(context)
    
    if investment_style:
        financial_contexts.append(f"Investment style: {investment_style}")
    
    if time_horizon:
        financial_contexts.append(f"Time horizon: {time_horizon}")
    
    combined_context = ". ".join(financial_contexts)
    
    query_embedding = embedding_model.encode([combined_context])[0]
    
    feature_embeddings = []
    for feature in numerical_columns:
        feature_context = f"Financial metric: {feature}. {combined_context}"
        feature_emb = embedding_model.encode([feature_context])[0]
        feature_embeddings.append(feature_emb)
    
    feature_similarities = []
    for feature_emb in feature_embeddings:
        similarity = cosine_similarity([query_embedding], [feature_emb])[0][0]
        feature_similarities.append(similarity)
    
    user_vector = np.array(feature_similarities)
    
    user_vector = (user_vector - user_vector.min()) / (user_vector.max() - user_vector.min() + 1e-8)
    
    return user_vector

def calculate_similarity_and_recommend(df, user_vector, numerical_columns, top_n):
    """
    Calculate cosine similarity between user preferences and stocks.
    Args:
        df: DataFrame containing stock data
        user_vector: User preference vector
        numerical_columns: List of numerical feature columns
        top_n: Number of recommendations to return
    Returns:
        DataFrame with top N recommendations
    """
    df_clean = df.dropna(subset=numerical_columns)
    
    if df_clean.empty:
        print("No stocks left after removing NaN values")
        return pd.DataFrame()
    
    print(f"Calculating similarity for {len(df_clean)} stocks")
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_clean[numerical_columns])
    
    user_vector_df = pd.DataFrame(user_vector.reshape(1, -1), columns=numerical_columns)
    user_vector_scaled = scaler.transform(user_vector_df)[0]
    
    similarities = cosine_similarity([user_vector_scaled], scaled_features)[0]
    
    df_with_scores = df_clean.copy()
    df_with_scores['Similarity_Score'] = similarities
    
    recommendations = df_with_scores.nlargest(top_n, 'Similarity_Score')
    
    output_columns = ['Ticker', 'Company_Name', 'Sector', 'Industry', 'Similarity_Score'] + numerical_columns
    available_columns = [col for col in output_columns if col in recommendations.columns]
    
    print(f"Top {top_n} recommendations found with similarity scores: {recommendations['Similarity_Score'].values}")
    
    return recommendations[available_columns].round(4)

def generate_recommendation_justification(user_query, recommendations_df, user_preferences, groq_client):
    """
    Generate a justification for the stock recommendations using Groq LLM.
    Args:
        user_query: Original user query
        recommendations_df: DataFrame containing recommended stocks
        user_preferences: Extracted user preferences
        groq_client: Groq client instance
    Returns:
        Justification string for the recommendations
    """
    stock_info = []
    for idx, row in recommendations_df.iterrows():
        stock_name = row.get('Company_Name', row.get('Ticker', f'Stock {idx}'))
        sector = row.get('Sector', 'N/A')
        industry = row.get('Industry', 'N/A')
        
        stock_summary = f"- {stock_name} (Sector: {sector}, Industry: {industry})"
        stock_info.append(stock_summary)
    
    stock_list = "\n".join(stock_info)
    
    justification_prompt = f"""
    Based on the user's investment query: "{user_query}"
    
    The following stocks have been recommended (in bullet point list form):
    {stock_list}
    
    User preferences extracted: {user_preferences}
    
    Please provide a clear, concise justification (2-3 sentences) explaining why these stocks are good matches for the user's requirements. Focus on:
    1. How the stocks align with their specified criteria
    2. Key strengths of the selected stocks
    3. Why they form a good portfolio for their needs
    
    Keep the response professional and investment-focused.
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial advisor providing clear, concise investment justifications."
                },
                {
                    "role": "user",
                    "content": justification_prompt
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=200
        )
        
        justification = chat_completion.choices[0].message.content.strip()
        return justification
        
    except Exception as e:
        print(f"Error generating justification: {e}")
        return f"These stocks were selected based on their strong alignment with your criteria: {', '.join(user_preferences.keys()) if user_preferences else 'your investment preferences'}. They represent a diversified selection that matches your risk profile and investment objectives."

def recommend_stocks_from_query(df, user_query, numerical_columns, non_numerical_columns, top_n=5, groq_api_key=None):
    """
    Recommend stocks based on natural language query using Groq LLM and Hugging Face finance embeddings.
    
    Args:
        df: DataFrame containing stock data
        user_query: Natural language query (e.g., "I want 5 stocks that are low risk, high return, in retail")
        numerical_columns: List of numerical feature columns for similarity calculation
        non_numerical_columns: List of non-numerical columns for categorical filtering
        top_n: Number of stocks to recommend
        groq_api_key: Groq API key
        
    Returns:
        Dict[str, Any]: Dictionary containing recommended stock names and justification
    """
    
    # Initialize Groq client
    if not groq_api_key:
        raise ValueError("groq_api_key is required")
    
    groq_client = Groq(api_key=groq_api_key)
    
    # Initialize finance embeddings model
    print("Loading finance embeddings model...")
    embedding_model = SentenceTransformer('FinLang/finance-embeddings-investopedia')
    
    # Get all column names
    all_columns = numerical_columns + non_numerical_columns
    
    # Step 1: Extract preferences using Groq LLM
    user_preferences = extract_preferences_with_groq(user_query, all_columns, 
                                                    numerical_columns, non_numerical_columns, groq_client)
    
    if not user_preferences:
        print("Could not extract valid preferences from query")
        return {
            "recommended_stocks": [],
            "justification": "Could not extract valid preferences from your query. Please try rephrasing your request."
        }
    
    # Step 2: Filter stocks based on categorical preferences
    filtered_df = filter_stocks_by_categories(df, user_preferences)
    
    if filtered_df.empty:
        print("No stocks match the categorical criteria")
        return {
            "recommended_stocks": [],
            "justification": "No stocks match your specified criteria. Please try adjusting your requirements."
        }
    
    # Step 3: Create user preference vector using embeddings
    user_vector = create_user_preference_vector_with_embeddings(
        user_query, user_preferences, numerical_columns, embedding_model
    )
    
    # Step 4: Calculate similarity and recommend
    recommendations_df = calculate_similarity_and_recommend(filtered_df, user_vector, numerical_columns, top_n)
    
    # Step 5: Extract stock names and generate justification
    if recommendations_df.empty:
        return {
            "recommended_stocks": [],
            "justification": "No suitable stock recommendations could be generated based on your criteria."
        }
    
    # Get recommended stock names
    recommended_stocks = []
    if 'Company_Name' in recommendations_df.columns:
        recommended_stocks = recommendations_df['Company_Name'].tolist()
    elif 'Ticker' in recommendations_df.columns:
        recommended_stocks = recommendations_df['Ticker'].tolist()
    else:
        # Fallback to index if no company name or ticker columns
        recommended_stocks = recommendations_df.index.tolist()
    
    # Step 6: Generate justification using Groq LLM
    justification = generate_recommendation_justification(
        user_query, recommendations_df, user_preferences, groq_client
    )
    
    return {
        "answer": justification,
        "context": recommended_stocks,
        "query": user_query,
        "num_sources": top_n,
        "success": True
    }