def highlight_risk(value):
    """
    Apply background color highlighting based on risk level values.
    Args:
        value: Risk level string ('High', 'Medium', 'Low')
    Returns:
        str: CSS background-color style string
    """
    risk_colors = {
        'High': '#FFB6C1',
        'Medium': '#FFFFE0', 
        'Low': '#90EE90'
    }
    return f'background-color: {risk_colors.get(value, "")}'

def highlight_sentiment(value):
    """
    Apply background color highlighting based on sentiment values.
    Args:
        value: Sentiment string ('Extremely_Negative', 'Negative', 'Neutral', 'Positive', 'Extremely_Positive')
    Returns:
        str: CSS background-color style string
    """
    sentiment_colors = {
        'Extremely_Negative': '#FF6B6B',
        'Negative': '#FFB6C1',
        'Neutral': '#FFFFE0',
        'Positive': '#90EE90',
        'Extremely_Positive': '#32CD32'
    }
    return f'background-color: {sentiment_colors.get(value, "")}'