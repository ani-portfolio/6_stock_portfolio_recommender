import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from transformers import pipeline
import numpy as np
from parameters import *

sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

def get_news_articles(ticker_symbol, api_key, base_url, company_name=None, days_back=7, page_size=100):
    """
    Fetch news articles for a given stock ticker.
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        api_key (str): News API key
        base_url (str): API base URL
        company_name (str): Company name for better search results
        days_back (int): Number of days to look back for news
        page_size (int): Number of articles to fetch (max 100)
    Returns:
        List[Dict]: List of news articles
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    query = f'"{ticker_symbol}" OR "{company_name}"' if company_name else f'"{ticker_symbol}"'
    
    params = {
        'q': query,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': min(page_size, 100),
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', []) if data['status'] == 'ok' else []
    except:
        return []

def analyze_text_sentiment(text, sentiment_analyzer):
    """
    Analyze sentiment of a single text using HuggingFace model.
    Args:
        text (str): Text to analyze
        sentiment_analyzer: HuggingFace sentiment analysis pipeline
    Returns:
        Dict: Sentiment analysis results with granular labels
    """
    try:
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        result = sentiment_analyzer(text)[0]
        original_label = result['label'].upper()
        confidence_score = result['score']
        
        is_positive = 'POSITIVE' in original_label or 'POS' in original_label
        is_negative = 'NEGATIVE' in original_label or 'NEG' in original_label
        
        def get_granular_sentiment(score):
            if score <= 0.25:
                return 'extremely_negative'
            elif score <= 0.45:
                return 'negative'
            elif score <= 0.55:
                return 'neutral'
            elif score <= 0.75:
                return 'positive'
            else:
                return 'extremely_positive'
        
        if is_negative:
            adjusted_score = 1.0 - confidence_score
        elif is_positive:
            adjusted_score = confidence_score
        else:
            adjusted_score = 0.5
        
        granular_sentiment = get_granular_sentiment(adjusted_score)
        
        return {
            'granular_sentiment': granular_sentiment,
            'score': adjusted_score
        }
        
    except:
        return {
            'granular_sentiment': 'neutral',
            'score': 0.5
        }

def analyze_articles_sentiment(articles, sentiment_analyzer):
    """
    Analyze sentiment for a list of news articles.
    Args:
        articles (List[Dict]): List of news articles
        sentiment_analyzer: HuggingFace sentiment analysis pipeline
    Returns:
        Dict: Aggregated sentiment analysis
    """
    if not articles:
        return {'Sentiment': 'neutral'}
    
    sentiment_scores = []
    
    for article in articles:
        text_to_analyze = ""
        if article.get('title'):
            text_to_analyze += article['title'] + " "
        if article.get('description'):
            text_to_analyze += article['description']
        
        if text_to_analyze.strip():
            sentiment_result = analyze_text_sentiment(text_to_analyze, sentiment_analyzer)
            sentiment_scores.append(sentiment_result['score'])
    
    if not sentiment_scores:
        return {'Sentiment': 'neutral'}
    
    overall_score = np.mean(sentiment_scores)
    
    def get_overall_sentiment(score):
        if score <= 0.25:
            return 'extremely_negative'
        elif score <= 0.45:
            return 'negative'
        elif score <= 0.55:
            return 'neutral'
        elif score <= 0.75:
            return 'positive'
        else:
            return 'extremely_positive'
    
    return {'Sentiment': get_overall_sentiment(overall_score)}

def get_stock_sentiment(ticker_symbol, api_key, base_url):
    """
    Get complete sentiment analysis for a stock.
    Args:
        ticker_symbol (str): Stock ticker symbol
    Returns:
        Dict: Complete sentiment analysis results
    """
    articles = get_news_articles(ticker_symbol, api_key, base_url, days_back=14, page_size=100)
    sentiment_results = analyze_articles_sentiment(articles, sentiment_analyzer)
    sentiment_results.update({'Ticker': ticker_symbol})
    return sentiment_results

def analyze_multiple_stocks(tickers, api_key, base_url):
    """
    Analyze sentiment for multiple stock tickers.
    Args:
        tickers (List[str]): List of stock ticker symbols
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    results = []
    for ticker in tickers:
        try:
            sentiment_results = get_stock_sentiment(ticker, api_key, base_url)
            results.append(sentiment_results)
            time.sleep(0.1)
        except:
            results.append({'Ticker': ticker, 'Sentiment': 'neutral'})
    
    return pd.DataFrame(results)