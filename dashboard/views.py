import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
import io, base64
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from sklearn.linear_model import LinearRegression
from .models import *
from .utils import load_data
from .ml_model import train_model
import csv
from datetime import datetime


# ---------------- REGISTER ----------------
def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        # 🔒 FORCE ROLE (NO FRONTEND TRUST)
        role = "analyst"

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
        else:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )

            # create profile with FIXED role
            UserProfile.objects.create(user=user, role=role)

            messages.success(request, "Account created! Please login.")
            return redirect('login')

    return render(request, "authentication/register.html")


# ---------------- LOGIN ----------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            try:
                profile = UserProfile.objects.get(user=user)
                request.session['role'] = profile.role
            except UserProfile.DoesNotExist:
                request.session['role'] = 'analyst'
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password")
    return render(request, "authentication/login.html")


# ---------------- LOGOUT ----------------
def logout_view(request):
    logout(request)
    return redirect('dashboard')


# ---------------- HELPER - Enhanced for Blue Theme ----------------
def plot_to_base64(fig, width=10, height=6, dpi=100):
    """Convert matplotlib figure to base64 string with better styling"""
    buf = io.BytesIO()
    
    # Close any existing figures to free memory
    plt.close('all')
    
    # Set style without seaborn if not available
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    # Custom styling for blue theme (without accessing spines before save)
    fig.patch.set_facecolor('#f8fafc')
    for ax in fig.get_axes():
        ax.set_facecolor('#ffffff')
        ax.spines['top'].set_color('#3b82f6')
        ax.spines['right'].set_color('#3b82f6')
        ax.spines['bottom'].set_color('#3b82f6')
        ax.spines['left'].set_color('#3b82f6')
    
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, facecolor='#f8fafc')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    buf.close()
    
    return img_base64

# ---------------- DASHBOARD - Updated for Temperature Data ----------------
def dashboard(request):
    df = load_data()

    # Filters from GET
    indicator = request.GET.get('indicator', 'Temperature')  # Only Temperature available
    country = request.GET.get('country', 'Global')
    start_year = request.GET.get('start_year')
    end_year = request.GET.get('end_year')

    try:
        start_year = int(start_year)
    except (TypeError, ValueError):
        start_year = df['Year'].min() if not df.empty else 1961

    try:
        end_year = int(end_year)
    except (TypeError, ValueError):
        end_year = df['Year'].max() if not df.empty else 2022

    # Filter by country
    if country != 'Global':
        df_filtered = df[df['Country'] == country]
    else:
        df_filtered = df.copy()
    
    # Filter by year range
    df_filtered = df_filtered[(df_filtered['Year'] >= start_year) & (df_filtered['Year'] <= end_year)]

    temperature_chart = None
    indicator_wise_data = []
    
    if not df_filtered.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Blue theme styling
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#f8fafc')
        
        # Plot temperature data
        grouped = df_filtered.groupby('Year')['Value'].mean()
        
        # Color for temperature (red/orange theme for temperature)
        color = '#ef4444'
        
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=3, 
                color=color, markerfacecolor=color, markersize=8, 
                markeredgecolor='white', markeredgewidth=2)
        ax.fill_between(grouped.index, grouped.values, alpha=0.2, color=color)
        
        # Calculate trend
        if len(grouped) > 1:
            trend_pct = ((grouped.iloc[-1] - grouped.iloc[0]) / abs(grouped.iloc[0])) * 100 if grouped.iloc[0] != 0 else 0
            trend_text = f"📈 +{trend_pct:.1f}% increase" if trend_pct > 0 else f"📉 {trend_pct:.1f}% decrease"
        else:
            trend_text = "Insufficient data for trend analysis"
        
        ax.set_title(f"Temperature Anomaly Analysis - {country} ({start_year}-{end_year})\n{trend_text}", 
                    fontsize=16, fontweight='bold', color='#0c4a6e', pad=20)
        
        ax.set_xlabel("Year", fontsize=12, fontweight='500', color='#1e3a8a')
        ax.set_ylabel("Temperature Anomaly (°C)", fontsize=12, fontweight='500', color='#1e3a8a')
        ax.tick_params(axis='x', rotation=45, colors='#475569')
        ax.tick_params(axis='y', colors='#475569')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#3b82f6')
        ax.spines['left'].set_color('#3b82f6')
        
        # Add horizontal line at zero (baseline)
        ax.axhline(y=0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5)
        
        temperature_chart = plot_to_base64(fig)
        
        # Add temperature statistics
        indicator_wise_data.append({
            'name': 'Temperature Anomaly',
            'avg': round(grouped.mean(), 2),
            'min': round(grouped.min(), 2),
            'max': round(grouped.max(), 2),
            'trend': round(trend_pct, 2) if len(grouped) > 1 else 0,
            'unit': '°C'
        })

    # Calculate summary statistics
    total_records = len(df_filtered)
    avg_temp = df_filtered['Value'].mean() if not df_filtered.empty else 0
    
    # Get table data for preview
    table_data = df_filtered[['Year', 'Value', 'Country']].head(15).to_dict(orient='records') if not df_filtered.empty else []
    
    # Add indicator name to table data
    for row in table_data:
        row['Indicator'] = 'Temperature Anomaly'
        row['Unit'] = '°C'
    
    context = {
        "records": total_records,
        "avg_value": round(avg_temp, 2),
        "co2_chart": temperature_chart,
        "selected_indicator": "Temperature Anomaly",
        "selected_country": country,
        "start_year": start_year,
        "end_year": end_year,
        "indicators": ['Temperature Anomaly'],
        "countries": ['Global'] + sorted(df['Country'].unique()) if not df.empty else ['Global'],
        "table_data": table_data,
        "indicator_wise_data": indicator_wise_data,
        "unique_indicators": 1,
    }
    return render(request, "dashboard/dashboard.html", context)


# ---------------- REGIONS - Updated for Temperature Data ----------------
def regions(request):
    df = load_data()
    
    if df.empty:
        return render(request, "dashboard/regions.html", {
            "country": "No Data",
            "countries": [],
            "chart": None,
            "data": []
        })
    
    country = request.GET.get("country", df['Country'].unique()[0] if len(df['Country'].unique()) > 0 else "Global")
    
    df_country = df[df['Country'] == country]

    chart = None
    if not df_country.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Blue theme styling
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#f8fafc')
        
        # Plot temperature data
        grouped = df_country.groupby('Year')['Value'].mean()
        
        # Temperature color (red/orange gradient)
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2.5, 
               label='Temperature Anomaly', color='#ef4444', markersize=6)
        ax.fill_between(grouped.index, grouped.values, alpha=0.15, color='#ef4444')
        
        ax.set_title(f"{country} - Temperature Anomaly Analysis (Baseline: 1951-1980)", 
                    fontsize=16, fontweight='bold', color='#0c4a6e', pad=20)
        ax.set_xlabel("Year", fontsize=12, fontweight='500', color='#1e3a8a')
        ax.set_ylabel("Temperature Anomaly (°C)", fontsize=12, fontweight='500', color='#1e3a8a')
        ax.tick_params(axis='x', rotation=45, colors='#475569')
        ax.tick_params(axis='y', colors='#475569')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', framealpha=0.9, facecolor='#ffffff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#3b82f6')
        ax.spines['left'].set_color('#3b82f6')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5)
        
        chart = plot_to_base64(fig, width=14, height=7)

    # Prepare data for table
    table_data = df_country[['Year', 'Value']].to_dict(orient='records') if not df_country.empty else []
    for row in table_data:
        row['Indicator'] = 'Temperature Anomaly'

    return render(request, "dashboard/regions.html", {
        "country": country,
        "countries": ['Global'] + sorted(df['Country'].unique()) if not df.empty else ['Global'],
        "indicator_filter": "Temperature",
        "chart": chart,
        "data": table_data
    })


# ---------------- PREDICTIONS - Updated for Temperature Data ----------------
def predictions(request):
    df = load_data()
    
    if df.empty:
        return render(request, "dashboard/predictions.html", {
            "chart": None,
            "future_value": None,
            "indicators": ['Temperature Anomaly'],
            "countries": [],
            "selected_indicator": "Temperature Anomaly",
            "selected_country": "No Data"
        })
    
    # GET inputs
    indicator = "Temperature Anomaly"
    country = request.GET.get('country', 'Global')
    future_year = request.GET.get('future_year')

    # Filter data
    df_ind = df.copy()

    if country != 'Global':
        df_ind = df_ind[df_ind['Country'] == country]

    # Safety check
    if df_ind.empty or len(df_ind) < 2:
        return render(request, "dashboard/predictions.html", {
            "chart": None,
            "future_value": None,
            "future_year": future_year,
            "indicators": ['Temperature Anomaly'],
            "countries": ['Global'] + sorted(df['Country'].unique()),
            "selected_indicator": indicator,
            "selected_country": country
        })

    # Sort data by year
    df_ind = df_ind.sort_values('Year')

    # Prepare model
    X = df_ind[['Year']]
    y = df_ind['Value']

    model = LinearRegression()
    model.fit(X, y)

    # Predict existing trend
    pred = model.predict(X)

    # ---------------- FUTURE PREDICTION ----------------
    future_value = None
    if future_year:
        try:
            future_year = int(future_year)
            future_pred = model.predict([[future_year]])
            future_value = round(future_pred[0], 2)
        except:
            future_value = None

    # ---------------- CHART with Blue Theme ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Blue theme styling
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    
    # Actual data points
    ax.scatter(X['Year'], y, label="Historical Temperature Anomaly", color='#3b82f6', s=80, alpha=0.7, edgecolors='#1e40af')
    
    # Trend line
    ax.plot(X['Year'], pred, color='#ef4444', linewidth=2.5, label="Trend Line", linestyle='--')
    
    # Future point
    if future_value and future_year:
        ax.scatter(future_year, future_value, color='#10b981', s=150, 
                  label=f"{future_year} Prediction: {future_value}°C", 
                  edgecolors='#047857', linewidth=2, zorder=5)
        ax.annotate(f'{future_value}°C', (future_year, future_value), 
                   xytext=(5, 10), textcoords='offset points', 
                   fontsize=10, fontweight='bold', color='#10b981')

    ax.set_title(f"Temperature Anomaly Prediction - {country}", 
                fontsize=16, fontweight='bold', color='#0c4a6e', pad=20)
    ax.set_xlabel("Year", fontsize=12, fontweight='500', color='#1e3a8a')
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=12, fontweight='500', color='#1e3a8a')
    ax.tick_params(colors='#475569')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, facecolor='#ffffff', edgecolor='#3b82f6')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#3b82f6')
    ax.spines['left'].set_color('#3b82f6')
    ax.axhline(y=0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5)
    
    chart = plot_to_base64(fig)

    return render(request, "dashboard/predictions.html", {
        "chart": chart,
        "future_value": f"{future_value}°C" if future_value else None,
        "future_year": future_year,
        "indicators": ['Temperature Anomaly'],
        "countries": ['Global'] + sorted(df['Country'].unique()),
        "selected_indicator": indicator,
        "selected_country": country
    })

def analytics(request):
    df = load_data()

    # ---------------- DEFAULTS ----------------
    selected_regions = []
    heatmap_base64 = None
    start_year = 1961
    end_year = 2022

    if df.empty:
        messages.error(request, "No data available")
        return render(request, "dashboard/analytics.html", {
            "heatmap": None,
            "selected_regions": [],
            "start_year": start_year,
            "end_year": end_year
        })

    # ---------------- YEARS ----------------
    try:
        if request.GET.get('start_year'):
            start_year = int(request.GET.get('start_year'))
        if request.GET.get('end_year'):
            end_year = int(request.GET.get('end_year'))
    except:
        start_year = 1961
        end_year = 2022

    # ---------------- FIXED REGION HANDLING ----------------
    regions_param = request.GET.get('regions')

    if regions_param:
        selected_regions = [r.strip() for r in regions_param.split(',') if r.strip()]
    elif not request.GET:
        # Default regions that exist in the data
        available_countries = df['Country'].unique()
        default_regions = []
        for region in ['USA', 'China', 'India']:
            if region in available_countries:
                default_regions.append(region)
        selected_regions = default_regions[:3] if default_regions else ['Global']

    print("FINAL SELECTED REGIONS:", selected_regions)

    # ---------------- HEATMAP ----------------
    if len(selected_regions) >= 2:
        # Filter by year range
        df_filtered = df[
            (df['Year'] >= start_year) &
            (df['Year'] <= end_year)
        ].copy()

        # Prepare data for correlation
        correlation_data_dict = {}

        for region in selected_regions:
            if region == 'Global':
                # Calculate global average (mean of all countries for each year)
                global_data = df_filtered.groupby('Year')['Value'].mean()
                if not global_data.empty:
                    correlation_data_dict[region] = global_data
            else:
                # Get specific country data
                country_data = df_filtered[df_filtered['Country'] == region]
                if not country_data.empty:
                    # Group by year and get mean value (in case of multiple entries per year)
                    yearly_data = country_data.groupby('Year')['Value'].mean()
                    if len(yearly_data) >= 3:  # Need at least 3 years for correlation
                        correlation_data_dict[region] = yearly_data
                else:
                    messages.warning(request, f"No data available for {region}")

        # Create DataFrame from collected data
        if len(correlation_data_dict) >= 2:
            # Create a DataFrame with years as index and regions as columns
            correlation_df = pd.DataFrame(correlation_data_dict)
            
            # Drop any rows with NaN values
            correlation_df = correlation_df.dropna()
            
            print(f"Correlation DataFrame shape: {correlation_df.shape}")
            print(f"Years available: {len(correlation_df)}")
            print(f"Regions: {list(correlation_df.columns)}")
            
            if correlation_df.shape[1] >= 2 and correlation_df.shape[0] >= 3:
                # Calculate correlation matrix
                corr = correlation_df.corr()
                
                print("Correlation Matrix:")
                print(corr)
                
                # ---------------- CREATE HEATMAP ----------------
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Apply blue theme styling
                plt.style.use('seaborn-v0_8-darkgrid')
                fig.patch.set_facecolor('#f8fafc')
                ax.set_facecolor('#ffffff')
                
                # Create heatmap
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    linewidths=1.5,
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                    ax=ax,
                    annot_kws={'size': 10, 'weight': 'bold'}
                )
                
                ax.set_title(
                    f"Temperature Anomaly Correlation Matrix\n{', '.join(selected_regions[:3])}{'...' if len(selected_regions) > 3 else ''} ({start_year}-{end_year})",
                    fontsize=14,
                    fontweight='bold',
                    color='#0c4a6e',
                    pad=20
                )
                
                ax.set_xlabel("Regions", fontsize=11, fontweight='500', color='#1e3a8a')
                ax.set_ylabel("Regions", fontsize=11, fontweight='500', color='#1e3a8a')
                
                # Style the heatmap
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                # Convert to base64
                heatmap_base64 = plot_to_base64(fig, width=10, height=8)
                
                messages.success(request, f"Correlation heatmap generated successfully for {len(correlation_df)} years!")
            else:
                if correlation_df.shape[1] < 2:
                    messages.warning(request, "Need at least 2 regions with overlapping data")
                else:
                    messages.warning(request, f"Need at least 3 years of overlapping data (found {correlation_df.shape[0]} years)")
        else:
            messages.warning(request, "Could not find enough overlapping data for correlation analysis")
    else:
        if len(selected_regions) == 0:
            messages.info(request, "Please add at least 2 regions for correlation analysis")
        elif len(selected_regions) == 1:
            messages.info(request, f"Please add one more region (currently have {selected_regions[0]})")

    # Get available countries for the dropdown
    available_countries = sorted(df['Country'].unique())
    
    return render(request, "dashboard/analytics.html", {
        "heatmap": heatmap_base64,
        "selected_regions": selected_regions,
        "start_year": start_year,
        "end_year": end_year,
        "available_countries": available_countries
    })


@login_required(login_url='login')
def feedback(request):
    

    # 🔥 Only analyst allowed
    if request.session.get('role') != 'analyst':
        return redirect('unauthorized')

    message = None
    recent_feedback = []

    # ---------------- SAVE FEEDBACK ----------------
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        rating = request.POST.get("rating")
        comments = request.POST.get("comments")

        Feedback.objects.create(
            user=request.user,
            name=name,
            email=email,
            rating=rating,
            comments=comments
        )

        message = "Thank you for your valuable feedback! 🌟"

    # ---------------- ONLY CURRENT USER DATA ----------------
    feedback_qs = Feedback.objects.filter(user=request.user).order_by('-created_at')[:5]

    recent_feedback = [
        {
            "name": fb.name,
            "email": fb.email,
            "rating": fb.rating,
            "comments": fb.comments,
            "created_at": fb.created_at
        }
        for fb in feedback_qs
    ]

    return render(request, "dashboard/feedback.html", {
        "message": message,
        "recent_feedback": recent_feedback
    })
    
    
    