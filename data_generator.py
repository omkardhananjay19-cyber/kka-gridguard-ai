"""
Synthetic Power Grid Outage Data Generator
Creates realistic power outage data with weather, load, equipment, and maintenance features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_grid_data(n_zones=30, n_samples=2000, random_seed=42):
    """
    Generate synthetic power grid data with realistic patterns.
    
    Features:
    - electricity_load: Peak/off-peak patterns
    - rainfall: Seasonal weather patterns
    - temperature: Seasonal variation
    - equipment_age: Transformer age in years
    - maintenance_frequency: Times serviced in last year
    - past_outages: Historical outage count (12 months)
    - grid_complexity: Complexity score (more equipment = more risk)
    """
    
    np.random.seed(random_seed)
    
    data = []
    
    for zone_id in range(1, n_zones + 1):
        # Zone-specific characteristics
        base_risk = np.random.uniform(0.3, 0.8)  # Inherent zone riskiness
        equipment_age = np.random.uniform(5, 35)  # 5-35 years old
        maintenance_frequency = np.random.randint(1, 12)  # 1-12 times per year
        grid_complexity = np.random.uniform(0.3, 0.95)  # Equipment complexity
        
        for day in range(n_samples):
            # Time-based features
            date = datetime(2022, 1, 1) + timedelta(days=day)
            month = date.month
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Load patterns: Higher on weekdays, peak in evening
            base_load = 60 + (20 * (1 - is_weekend))  # 60-80 range
            peak_hour_multiplier = 1.3 if 17 <= date.hour <= 21 else 0.8
            electricity_load = base_load * (1 + np.random.normal(0, 0.1)) * peak_hour_multiplier
            electricity_load = np.clip(electricity_load, 30, 100)
            
            # Weather patterns: Seasonal rainfall, monsoon effect
            if month in [6, 7, 8, 9]:  # Monsoon season
                rainfall = np.random.exponential(scale=15) + np.random.uniform(0, 30)
            else:
                rainfall = np.random.exponential(scale=5) + np.random.uniform(0, 10)
            rainfall = np.clip(rainfall, 0, 100)
            
            # Temperature with seasonal variation
            base_temp = 20 + 15 * np.sin(2 * np.pi * month / 12)
            temperature = base_temp + np.random.normal(0, 3)
            temperature = np.clip(temperature, -5, 50)
            
            # Maintenance history (decreases as maintenance increases)
            maintenance_quality = maintenance_frequency / 12.0
            equipment_health = 1 - (equipment_age / 40.0) * (1 - maintenance_quality)
            equipment_health = np.clip(equipment_health, 0.2, 0.95)
            
            # Past outages: Last 12 months history
            past_outages = np.random.poisson(lam=base_risk * 2)
            
            # Humidity (correlated with rainfall)
            humidity = 50 + (rainfall / 100) * 40 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 100)
            
            # Voltage drop indicator (higher = worse equipment condition)
            voltage_stability = equipment_health + (1 - humidity/100) * 0.2
            voltage_stability = np.clip(voltage_stability, 0.3, 1.0)
            
            # **OUTAGE TARGET** (Logistic regression of risk factors)
            # Higher risk from: high load, heavy rain, old equipment, low maintenance, high past outages
            risk_score = (
                0.3 * (electricity_load / 100) +           # Load contribution
                0.25 * (rainfall / 100) +                  # Weather contribution
                0.2 * (1 - equipment_health) +             # Equipment age/condition
                0.15 * (1 - maintenance_quality) +         # Maintenance gap
                0.1 * min(past_outages / 5, 1.0)          # Historical pattern
            )
            
            # Add random noise and monsoon boost
            if month in [6, 7, 8, 9]:
                risk_score *= 1.3  # Monsoon season amplification
            
            risk_score += np.random.normal(0, 0.08)
            risk_score = np.clip(risk_score, 0, 1)
            
            # Binary outage: Probabilistic based on risk score
            outage = 1 if np.random.random() < risk_score else 0
            
            data.append({
                'zone_id': zone_id,
                'date': date.strftime('%Y-%m-%d'),
                'month': month,
                'day_of_week': day_of_week,
                'electricity_load': round(electricity_load, 2),
                'rainfall': round(rainfall, 2),
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'equipment_age': round(equipment_age, 2),
                'maintenance_frequency': maintenance_frequency,
                'equipment_health': round(equipment_health, 3),
                'past_outages': past_outages,
                'grid_complexity': round(grid_complexity, 3),
                'voltage_stability': round(voltage_stability, 3),
                'outage': outage
            })
    
    df = pd.DataFrame(data)
    return df

def generate_zone_metadata(n_zones=30, random_seed=42):
    """
    Generate metadata for each grid zone (location, capacity, etc.)
    """
    np.random.seed(random_seed)
    
    zones = []
    base_lat, base_lon = 17.3850, 78.4867  # Hyderabad as center
    
    for zone_id in range(1, n_zones + 1):
        latitude = base_lat + np.random.uniform(-0.5, 0.5)
        longitude = base_lon + np.random.uniform(-0.5, 0.5)
        
        zones.append({
            'zone_id': zone_id,
            'zone_name': f'Zone-{zone_id}',
            'latitude': round(latitude, 4),
            'longitude': round(longitude, 4),
            'population': np.random.randint(50000, 500000),
            'infrastructure_type': np.random.choice(['Residential', 'Commercial', 'Industrial', 'Mixed']),
            'transformer_count': np.random.randint(5, 50)
        })
    
    return pd.DataFrame(zones)

if __name__ == "__main__":
    print("🔌 Generating synthetic power grid data...")
    
    # Generate data
    df = generate_grid_data(n_zones=30, n_samples=2000)
    metadata = generate_zone_metadata(n_zones=30)
    
    # Save to CSV
    df.to_csv('outage_data.csv', index=False)
    metadata.to_csv('zone_metadata.csv', index=False)
    
    print(f"✅ Generated {len(df)} records across {df['zone_id'].nunique()} zones")
    print(f"📊 Outage rate: {df['outage'].mean()*100:.2f}%")
    print(f"💾 Saved to outage_data.csv and zone_metadata.csv")
    print(f"\n📋 Data shape: {df.shape}")
    print(f"\n🎯 Class distribution:\n{df['outage'].value_counts()}")
