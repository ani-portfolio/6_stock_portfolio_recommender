import pandas as pd

def create_summary_chart(df):
    """
    Create a summary chart with top 25 stocks by market cap, sorted by risk and market cap.
    Args:
        df: DataFrame containing stock data with required columns
    Returns:
        DataFrame with top 25 stocks sorted by risk level and market cap
    """
    df_result = df.copy()
    
    def determine_risk_signal(row):
        """
        Determine risk level based on price metrics.
        Args:
            row: DataFrame row containing stock metrics
        Returns:
            Risk level string ('Low', 'Medium', 'High')
        """
        all_time_high_condition = row['Percent_From_All_Time_High'] < -10
        moving_avg_condition = row['Percent_Difference_200_Day_Moving_Average'] < 0
        
        if row['Percent_From_All_Time_High'] >= -10:
            return 'High'
        
        if all_time_high_condition and moving_avg_condition:
            return 'Low'
        
        if all_time_high_condition or moving_avg_condition:
            return 'Medium'
        
        return 'High'
    
    df_result['Risk'] = df_result.apply(determine_risk_signal, axis=1)
    
    df_top_25 = df_result.sort_values('Market_Cap', ascending=False).head(25)
    
    risk_order = ['Low', 'Medium', 'High']
    df_top_25['Risk'] = pd.Categorical(df_top_25['Risk'], categories=risk_order, ordered=True)
    
    df_sorted = df_top_25.sort_values(['Risk', 'Market_Cap'], ascending=[True, False])
    
    return df_sorted[
        ['Ticker', 'Company_Name', 'Market_Cap', 'Closing_Price', 'All_Time_High', 'Risk', 'Sentiment',
         'Percent_From_All_Time_High', 'Percent_Difference_200_Day_Moving_Average', 
         '24_Hour_Percent_Change', '7_Day_Percent_Change', '30_Day_Percent_Change']
    ].reset_index(drop=True)