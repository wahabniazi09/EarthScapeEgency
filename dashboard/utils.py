import pandas as pd
import os
import numpy as np

def load_data():
    """
    Load and reshape the climate change indicators dataset from wide to long format.
    The dataset has years as columns (F1961, F1962, etc.) which need to be melted.
    """
    path = os.path.join('data', 'climate_change_indicators.csv')
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"Warning: Data file not found at {path}")
        return create_sample_data()
    
    df = pd.read_csv(path)

    # Detect year columns (starting with F)
    year_cols = [c for c in df.columns if c.startswith('F')]

    # Melt wide → long
    df_melted = df.melt(
        id_vars=['Country', 'Indicator'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )

    # Clean Year (remove 'F' prefix and convert to int)
    df_melted['Year'] = df_melted['Year'].str.replace('F', '').astype(int)
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    # Drop NaNs
    df_melted = df_melted.dropna(subset=['Value'])
    
    # Add a simplified indicator name for better display
    df_melted['Indicator_Simple'] = 'Temperature Anomaly'
    
    return df_melted


def get_countries(df=None):
    """
    Get sorted list of unique countries from the dataset
    """
    if df is None:
        df = load_data()
    countries = sorted(df['Country'].unique())
    return countries


def get_years_range(df=None):
    """
    Get min and max years available in the dataset
    """
    if df is None:
        df = load_data()
    return int(df['Year'].min()), int(df['Year'].max())


def get_indicator_stats(df, country='Global', start_year=None, end_year=None):
    """
    Calculate statistics for temperature anomalies
    
    Parameters:
    - df: DataFrame
    - country: Country name or 'Global' for all countries
    - start_year: Start year filter
    - end_year: End year filter
    
    Returns:
    - Dictionary with statistics
    """
    # Filter by country
    if country != 'Global':
        df_filtered = df[df['Country'] == country]
    else:
        df_filtered = df.copy()
    
    # Filter by year range
    if start_year:
        df_filtered = df_filtered[df_filtered['Year'] >= start_year]
    if end_year:
        df_filtered = df_filtered[df_filtered['Year'] <= end_year]
    
    if df_filtered.empty:
        return {
            'avg': 0,
            'min': 0,
            'max': 0,
            'median': 0,
            'std': 0,
            'trend': 0,
            'records': 0
        }
    
    # Calculate statistics
    grouped = df_filtered.groupby('Year')['Value'].mean()
    
    if len(grouped) > 1:
        trend_pct = ((grouped.iloc[-1] - grouped.iloc[0]) / abs(grouped.iloc[0])) * 100 if grouped.iloc[0] != 0 else 0
    else:
        trend_pct = 0
    
    return {
        'avg': round(df_filtered['Value'].mean(), 3),
        'min': round(df_filtered['Value'].min(), 3),
        'max': round(df_filtered['Value'].max(), 3),
        'median': round(df_filtered['Value'].median(), 3),
        'std': round(df_filtered['Value'].std(), 3),
        'trend': round(trend_pct, 2),
        'records': len(df_filtered)
    }


def get_temperature_data_for_country(df, country, start_year=None, end_year=None):
    """
    Get temperature anomaly data for a specific country
    
    Parameters:
    - df: DataFrame
    - country: Country name
    - start_year: Optional start year filter
    - end_year: Optional end year filter
    
    Returns:
    - DataFrame with Year and Value columns
    """
    df_country = df[df['Country'] == country].copy()
    
    if start_year:
        df_country = df_country[df_country['Year'] >= start_year]
    if end_year:
        df_country = df_country[df_country['Year'] <= end_year]
    
    return df_country[['Year', 'Value']].sort_values('Year')


def get_global_temperature_data(df, start_year=None, end_year=None):
    """
    Get global average temperature anomaly data (all countries averaged)
    
    Parameters:
    - df: DataFrame
    - start_year: Optional start year filter
    - end_year: Optional end year filter
    
    Returns:
    - DataFrame with Year and Value columns (global average)
    """
    df_global = df.copy()
    
    if start_year:
        df_global = df_global[df_global['Year'] >= start_year]
    if end_year:
        df_global = df_global[df_global['Year'] <= end_year]
    
    # Calculate global average by year
    global_avg = df_global.groupby('Year')['Value'].mean().reset_index()
    global_avg.columns = ['Year', 'Value']
    
    return global_avg


def get_warming_rate(df, country='Global', years=30):
    """
    Calculate warming rate (°C per decade) for the last N years
    
    Parameters:
    - df: DataFrame
    - country: Country name or 'Global'
    - years: Number of recent years to analyze (default: 30)
    
    Returns:
    - Dictionary with warming rate information
    """
    # Get data
    if country == 'Global':
        data = get_global_temperature_data(df)
    else:
        data = get_temperature_data_for_country(df, country)
    
    if len(data) < 2:
        return {'rate': 0, 'total_change': 0, 'period': 0}
    
    # Get last N years
    data = data.sort_values('Year').tail(years)
    
    if len(data) < 2:
        return {'rate': 0, 'total_change': 0, 'period': 0}
    
    # Calculate linear trend
    from sklearn.linear_model import LinearRegression
    X = data['Year'].values.reshape(-1, 1)
    y = data['Value'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Rate per year
    rate_per_year = model.coef_[0]
    # Rate per decade
    rate_per_decade = rate_per_year * 10
    
    total_change = data['Value'].iloc[-1] - data['Value'].iloc[0]
    
    return {
        'rate_per_year': round(rate_per_year, 4),
        'rate_per_decade': round(rate_per_decade, 4),
        'total_change': round(total_change, 3),
        'period': f"{data['Year'].iloc[0]} to {data['Year'].iloc[-1]}",
        'years_analyzed': len(data)
    }


def get_top_warming_countries(df, start_year=None, end_year=None, top_n=10):
    """
    Get top N countries with highest warming rate
    
    Parameters:
    - df: DataFrame
    - start_year: Start year for analysis
    - end_year: End year for analysis
    - top_n: Number of top countries to return
    
    Returns:
    - DataFrame with top warming countries
    """
    if start_year is None:
        start_year = df['Year'].min()
    if end_year is None:
        end_year = df['Year'].max()
    
    results = []
    countries = df['Country'].unique()
    
    for country in countries:
        data = get_temperature_data_for_country(df, country, start_year, end_year)
        
        if len(data) >= 5:  # Need at least 5 data points for meaningful trend
            from sklearn.linear_model import LinearRegression
            X = data['Year'].values.reshape(-1, 1)
            y = data['Value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            results.append({
                'Country': country,
                'Warming_Rate_per_Decade': round(model.coef_[0] * 10, 4),
                'Total_Change': round(data['Value'].iloc[-1] - data['Value'].iloc[0], 3),
                'Start_Temp': round(data['Value'].iloc[0], 3),
                'End_Temp': round(data['Value'].iloc[-1], 3)
            })
    
    # Sort by warming rate (highest first)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Warming_Rate_per_Decade', ascending=False)
    
    return results_df.head(top_n)


def create_sample_data():
    """
    Create sample temperature anomaly data if the CSV file is not found
    """
    print("Creating sample temperature anomaly data...")
    
    countries = [
        'USA', 'China', 'India', 'Germany', 'Brazil', 'United Kingdom', 
        'France', 'Japan', 'Canada', 'Australia', 'Russia', 'South Africa',
        'Mexico', 'Indonesia', 'Turkey', 'Iran', 'Thailand', 'Egypt',
        'Nigeria', 'Pakistan', 'Global'
    ]
    
    years = list(range(1961, 2023))
    
    data = []
    for country in countries:
        for year in years:
            # Generate realistic temperature anomaly data
            # Baseline around 0 with increasing trend over time
            base_temp = 0
            trend = (year - 1961) * 0.018  # ~1°C warming over 55 years
            seasonal_variation = np.sin((year - 1961) * 0.2) * 0.1
            random_noise = np.random.normal(0, 0.08)
            
            # Different warming rates for different countries
            if country in ['Russia', 'Canada', 'USA']:
                warming_multiplier = 1.5  # Higher warming in northern regions
            elif country in ['India', 'Nigeria', 'Indonesia']:
                warming_multiplier = 0.8  # Lower warming in tropical regions
            else:
                warming_multiplier = 1.0
            
            value = base_temp + (trend * warming_multiplier) + seasonal_variation + random_noise
            
            data.append({
                'Country': country,
                'Indicator': 'Temperature change with respect to a baseline climatology, corresponding to the period 1951-1980',
                'Indicator_Simple': 'Temperature Anomaly',
                'Year': year,
                'Value': round(value, 2)
            })
    
    return pd.DataFrame(data)


# For quick testing
if __name__ == "__main__":
    # Test the load_data function
    df = load_data()
    print(f"Data loaded: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Years range: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Countries: {df['Country'].nunique()}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Test statistics function
    stats = get_indicator_stats(df, country='Global')
    print(f"\nGlobal Statistics: {stats}")
    
    # Test warming rate
    warming = get_warming_rate(df, country='Global')
    print(f"\nGlobal Warming Rate: {warming}")