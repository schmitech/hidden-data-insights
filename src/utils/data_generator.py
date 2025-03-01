import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class DataGenerator:
    """
    Generates synthetic datasets with hidden patterns for demonstration purposes.
    """
    
    @staticmethod
    def generate_ecommerce_data(num_records=1000, seed=42):
        """
        Generate an e-commerce dataset with hidden patterns in customer behavior,
        product performance, and sales trends.
        
        Hidden patterns include:
        1. Seasonal trends with peaks on weekends
        2. Price sensitivity varies by customer segment
        3. Product category performance correlates with marketing spend
        4. Customer lifetime value is influenced by first purchase category
        5. Certain product combinations have higher than expected co-purchase rates
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Date range for the past year
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        date_range = [start_date + timedelta(days=i) for i in range(366)]
        
        # Customer segments
        segments = ['Budget', 'Mainstream', 'Premium', 'Luxury']
        segment_weights = [0.4, 0.3, 0.2, 0.1]
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books']
        
        # Marketing spend by category (hidden correlation)
        marketing_spend = {
            'Electronics': 5000,
            'Clothing': 3500,
            'Home': 2800,
            'Beauty': 4200,
            'Sports': 3000,
            'Books': 1500
        }
        
        # Generate data
        data = []
        
        for _ in range(num_records):
            # Select date with weekend bias (hidden pattern 1)
            date = random.choice(date_range)
            is_weekend = date.weekday() >= 5
            date_weight = 1.5 if is_weekend else 1.0
            
            # Select customer segment
            segment = np.random.choice(segments, p=segment_weights)
            
            # Customer ID (consistent for repeat customers)
            customer_id = f"CUST-{random.randint(1, num_records // 3)}"
            
            # First purchase category for this customer (hidden pattern 4)
            if random.random() < 0.3:  # 30% chance this is a first purchase
                first_category = random.choice(categories)
            else:
                first_category = None
            
            # Select product category with marketing influence (hidden pattern 3)
            category_weights = [marketing_spend[cat] / sum(marketing_spend.values()) for cat in categories]
            category = np.random.choice(categories, p=category_weights)
            
            # Product ID
            product_id = f"{category[:3].upper()}-{random.randint(100, 999)}"
            
            # Base price by category
            base_prices = {
                'Electronics': 300,
                'Clothing': 60,
                'Home': 120,
                'Beauty': 40,
                'Sports': 80,
                'Books': 25
            }
            
            # Price with segment sensitivity (hidden pattern 2)
            segment_multipliers = {
                'Budget': 0.7,
                'Mainstream': 1.0,
                'Premium': 1.5,
                'Luxury': 2.5
            }
            
            base_price = base_prices[category]
            price_sensitivity = {
                'Budget': 0.9,      # Very sensitive
                'Mainstream': 0.5,  # Moderately sensitive
                'Premium': 0.3,     # Less sensitive
                'Luxury': 0.1       # Barely sensitive
            }
            
            # Apply price sensitivity (hidden pattern 2)
            price = base_price * segment_multipliers[segment]
            
            # Quantity purchased
            quantity = max(1, int(np.random.normal(3, 2)))
            
            # Seasonal adjustment (hidden pattern 1)
            month = date.month
            seasonal_factor = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_factor = 1.4
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.2
            elif month in [1, 2]:  # Post-holiday slump
                seasonal_factor = 0.8
            
            # Weekend boost (hidden pattern 1)
            if is_weekend:
                seasonal_factor *= 1.3
            
            # Calculate revenue with all factors
            revenue = price * quantity * seasonal_factor
            
            # Discount rate - higher for budget, lower for luxury (part of hidden pattern 2)
            discount_rate = max(0, min(0.5, np.random.normal(
                0.3 - (segments.index(segment) * 0.07), 0.1)))
            
            # Apply discount
            final_price = price * (1 - discount_rate)
            final_revenue = final_price * quantity * seasonal_factor
            
            # Customer lifetime value influence (hidden pattern 4)
            if first_category:
                ltv_multipliers = {
                    'Electronics': 1.8,
                    'Clothing': 1.4,
                    'Home': 1.6,
                    'Beauty': 1.5,
                    'Sports': 1.3,
                    'Books': 1.2
                }
                customer_ltv = random.uniform(100, 500) * ltv_multipliers[first_category]
            else:
                customer_ltv = random.uniform(100, 500)
            
            # Co-purchase pattern (hidden pattern 5)
            has_copurchase = random.random() < 0.4  # 40% chance of co-purchase
            if has_copurchase:
                # Certain combinations are more likely
                if category == 'Electronics':
                    copurchase_category = np.random.choice(['Books', 'Home'], p=[0.7, 0.3])
                elif category == 'Beauty':
                    copurchase_category = 'Clothing'
                elif category == 'Sports':
                    copurchase_category = np.random.choice(['Clothing', 'Electronics'], p=[0.8, 0.2])
                else:
                    copurchase_category = np.random.choice(
                        [c for c in categories if c != category])
                
                copurchase_product_id = f"{copurchase_category[:3].upper()}-{random.randint(100, 999)}"
            else:
                copurchase_category = None
                copurchase_product_id = None
            
            # Add record
            record = {
                'date': date,
                'customer_id': customer_id,
                'customer_segment': segment,
                'product_category': category,
                'product_id': product_id,
                'base_price': base_price,
                'discount_rate': discount_rate,
                'final_price': final_price,
                'quantity': quantity,
                'revenue': final_revenue,
                'is_weekend': is_weekend,
                'month': month,
                'customer_ltv': customer_ltv,
                'copurchase_category': copurchase_category,
                'copurchase_product_id': copurchase_product_id,
                'marketing_spend': marketing_spend[category]
            }
            
            data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    @staticmethod
    def generate_financial_data(num_records=1000, seed=42):
        """
        Generate a financial dataset with hidden patterns in stock prices,
        trading volumes, and market indicators.
        
        Hidden patterns include:
        1. Certain technical indicators predict price movements
        2. Volume spikes precede major price changes
        3. Sector correlations during market stress
        4. Seasonal patterns in specific sectors
        5. Options activity correlates with future volatility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Date range for the past 4 years (to include multiple market cycles)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=4*365)
        date_range = [start_date + timedelta(days=i) for i in range(4*365)]
        trading_dates = [d for d in date_range if d.weekday() < 5]  # Exclude weekends
        
        # Sectors
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        
        # Companies (multiple per sector)
        companies = []
        for sector in sectors:
            for i in range(5):  # 5 companies per sector
                companies.append({
                    'name': f"{sector[:3].upper()}-{i+1}",
                    'sector': sector,
                    'base_volatility': random.uniform(0.01, 0.03),
                    'market_beta': random.uniform(0.7, 1.3)
                })
        
        # Generate market data first
        market_data = []
        market_price = 1000.0
        market_trend = 0.0001  # Slight upward bias
        
        for date in trading_dates:
            # Seasonal component (hidden pattern 4)
            month = date.month
            seasonal_factor = 1.0
            if month in [11, 12]:  # Year-end rally
                seasonal_factor = 1.002
            elif month in [5, 6]:  # "Sell in May" effect
                seasonal_factor = 0.999
            
            # Market movement
            daily_return = np.random.normal(market_trend, 0.01) * seasonal_factor
            market_price *= (1 + daily_return)
            
            # Market stress periods (hidden pattern 3)
            is_stress_period = random.random() < 0.05  # 5% of days are "stress" days
            
            # Market volume with spikes before big moves (hidden pattern 2)
            base_volume = 1000000
            if abs(daily_return) > 0.02 or is_stress_period:
                volume_multiplier = random.uniform(1.5, 3.0)
            else:
                volume_multiplier = random.uniform(0.8, 1.2)
            
            market_volume = base_volume * volume_multiplier
            
            # Technical indicators (hidden pattern 1)
            rsi = random.uniform(30, 70)
            # RSI extremes tend to precede reversals
            if rsi < 35:
                next_day_bias = 0.005  # Bullish bias for next day
            elif rsi > 65:
                next_day_bias = -0.005  # Bearish bias for next day
            else:
                next_day_bias = 0
            
            # Options activity (hidden pattern 5)
            options_volume = base_volume * random.uniform(0.1, 0.3)
            put_call_ratio = random.uniform(0.7, 1.3)
            
            # Higher put/call ratio often precedes higher volatility
            if put_call_ratio > 1.1:
                volatility_bias = 0.005
            else:
                volatility_bias = 0
            
            market_data.append({
                'date': date,
                'market_price': market_price,
                'market_return': daily_return,
                'market_volume': market_volume,
                'is_stress_period': is_stress_period,
                'rsi': rsi,
                'next_day_bias': next_day_bias,
                'options_volume': options_volume,
                'put_call_ratio': put_call_ratio,
                'volatility_bias': volatility_bias
            })
        
        # Convert to DataFrame
        market_df = pd.DataFrame(market_data)
        
        # Generate individual stock data
        stock_data = []
        
        for company in companies:
            # Initialize price
            price = random.uniform(20, 200)
            
            for i, row in market_df.iterrows():
                date = row['date']
                
                # Sector-specific seasonal patterns (hidden pattern 4)
                seasonal_factor = 1.0
                if company['sector'] == 'Consumer' and date.month in [11, 12]:
                    seasonal_factor = 1.003  # Holiday boost for consumer stocks
                elif company['sector'] == 'Energy' and date.month in [1, 2]:
                    seasonal_factor = 1.002  # Winter boost for energy
                elif company['sector'] == 'Technology' and date.month in [4, 5]:
                    seasonal_factor = 1.001  # Spring tech rally
                
                # Market correlation (stronger during stress periods - hidden pattern 3)
                if row['is_stress_period']:
                    beta = company['market_beta'] * 1.5
                else:
                    beta = company['market_beta']
                
                # Stock-specific volatility
                stock_specific_return = np.random.normal(0, company['base_volatility'])
                
                # Technical indicator effect (hidden pattern 1)
                tech_effect = row['next_day_bias'] * random.uniform(0.5, 1.5)
                
                # Options activity effect on volatility (hidden pattern 5)
                vol_effect = row['volatility_bias'] * random.uniform(0.5, 1.5)
                
                # Combined daily return
                daily_return = (
                    beta * row['market_return'] +  # Market component
                    stock_specific_return +        # Stock-specific component
                    tech_effect +                  # Technical indicator effect
                    vol_effect                     # Options activity effect
                ) * seasonal_factor                # Seasonal effect
                
                # Update price
                price *= (1 + daily_return)
                
                # Volume with spikes before big moves (hidden pattern 2)
                base_volume = random.randint(100000, 500000)
                if abs(daily_return) > 0.03:
                    volume_multiplier = random.uniform(2.0, 4.0)
                else:
                    volume_multiplier = random.uniform(0.7, 1.3)
                
                volume = int(base_volume * volume_multiplier)
                
                # Add record
                record = {
                    'date': date,
                    'company': company['name'],
                    'sector': company['sector'],
                    'price': price,
                    'return': daily_return,
                    'volume': volume,
                    'market_return': row['market_return'],
                    'market_price': row['market_price'],
                    'is_stress_period': row['is_stress_period'],
                    'rsi': row['rsi'],
                    'options_volume': row['options_volume'] * random.uniform(0.5, 1.5),
                    'put_call_ratio': row['put_call_ratio'] * random.uniform(0.9, 1.1)
                }
                
                stock_data.append(record)
        
        # Convert to DataFrame and sample to get requested number of records
        df = pd.DataFrame(stock_data)
        if len(df) > num_records:
            df = df.sample(num_records, random_state=seed)
        
        return df
    
    @staticmethod
    def generate_healthcare_data(num_records=1000, seed=42):
        """
        Generate a healthcare dataset with hidden patterns in patient outcomes,
        treatment efficacy, and hospital operations.
        
        Hidden patterns include:
        1. Certain combinations of conditions lead to longer hospital stays
        2. Treatment efficacy varies by patient demographics
        3. Readmission rates correlate with discharge time of day
        4. Staffing levels impact patient outcomes
        5. Medication interactions affect recovery time
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Patient demographics
        age_groups = ['18-30', '31-45', '46-60', '61-75', '76+']
        age_weights = [0.15, 0.25, 0.3, 0.2, 0.1]
        
        genders = ['Male', 'Female']
        
        # Medical conditions
        conditions = ['Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 
                      'Cancer', 'Stroke', 'Obesity', 'Arthritis']
        
        # Treatments
        treatments = ['Medication A', 'Medication B', 'Medication C', 'Surgery', 
                      'Physical Therapy', 'Radiation', 'Counseling']
        
        # Hospitals
        hospitals = ['General Hospital', 'University Medical Center', 'Community Hospital', 
                     'Regional Medical Center', 'Specialty Clinic']
        
        # Generate data
        data = []
        
        for _ in range(num_records):
            # Patient demographics
            age_group = np.random.choice(age_groups, p=age_weights)
            gender = random.choice(genders)
            
            # Assign numeric age within group
            if age_group == '18-30':
                age = random.randint(18, 30)
            elif age_group == '31-45':
                age = random.randint(31, 45)
            elif age_group == '46-60':
                age = random.randint(46, 60)
            elif age_group == '61-75':
                age = random.randint(61, 75)
            else:  # 76+
                age = random.randint(76, 95)
            
            # Assign conditions (patients may have multiple)
            num_conditions = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            patient_conditions = random.sample(conditions, num_conditions)
            
            # Primary condition
            primary_condition = patient_conditions[0]
            
            # Hospital
            hospital = random.choice(hospitals)
            
            # Staffing level (hidden pattern 4)
            staffing_level = random.uniform(0.7, 1.3)  # Relative to optimal
            
            # Admission details
            admission_date = datetime.now().date() - timedelta(days=random.randint(1, 365))
            admission_hour = random.randint(0, 23)
            admission_time = f"{admission_hour:02d}:00"
            
            # Treatment
            primary_treatment = random.choice(treatments)
            
            # Secondary treatments
            num_secondary = random.randint(0, 2)
            secondary_treatments = []
            if num_secondary > 0:
                available_treatments = [t for t in treatments if t != primary_treatment]
                secondary_treatments = random.sample(available_treatments, num_secondary)
            
            # Hidden pattern 1: Certain combinations lead to longer stays
            base_stay_days = random.randint(1, 7)
            if 'Diabetes' in patient_conditions and 'Heart Disease' in patient_conditions:
                stay_multiplier = random.uniform(1.5, 2.0)  # Much longer stay
            elif 'Hypertension' in patient_conditions and 'Stroke' in patient_conditions:
                stay_multiplier = random.uniform(1.3, 1.8)  # Longer stay
            else:
                stay_multiplier = random.uniform(0.8, 1.2)  # Normal variation
            
            # Hidden pattern 4: Staffing impacts outcomes
            if staffing_level < 0.9:
                stay_multiplier *= random.uniform(1.1, 1.3)  # Understaffing extends stay
            
            # Calculate length of stay
            length_of_stay = max(1, int(base_stay_days * stay_multiplier))
            
            # Discharge details
            discharge_date = admission_date + timedelta(days=length_of_stay)
            discharge_hour = random.randint(8, 20)  # Discharges typically happen during day
            discharge_time = f"{discharge_hour:02d}:00"
            
            # Hidden pattern 3: Late discharges correlate with readmissions
            readmission_risk = 0.05  # Base readmission risk
            if discharge_hour >= 17:  # Late discharge (after 5 PM)
                readmission_risk *= 1.8
            
            # Hidden pattern 2: Treatment efficacy varies by demographics
            treatment_efficacy = 0.8  # Base efficacy
            
            # Age affects efficacy
            if age > 75 and primary_treatment == 'Medication A':
                treatment_efficacy *= 0.7  # Less effective in elderly
            elif age < 45 and primary_treatment == 'Physical Therapy':
                treatment_efficacy *= 1.3  # More effective in younger patients
            
            # Gender affects efficacy (simplified example)
            if gender == 'Female' and primary_treatment == 'Medication B':
                treatment_efficacy *= 1.2  # More effective in females
            elif gender == 'Male' and primary_treatment == 'Medication C':
                treatment_efficacy *= 1.1  # Slightly more effective in males
            
            # Hidden pattern 5: Medication interactions
            if 'Medication A' in secondary_treatments and primary_treatment == 'Medication B':
                treatment_efficacy *= 0.6  # Negative interaction
                readmission_risk *= 1.5  # Higher readmission risk
            elif 'Medication C' in secondary_treatments and primary_treatment == 'Physical Therapy':
                treatment_efficacy *= 1.3  # Positive interaction
            
            # Outcome metrics
            recovery_score = random.uniform(0, 10) * treatment_efficacy
            
            # Readmission (based on risk factors)
            was_readmitted = random.random() < readmission_risk
            
            # Cost (affected by length of stay, treatments, and hospital)
            base_cost = {
                'General Hospital': 1000,
                'University Medical Center': 1500,
                'Community Hospital': 800,
                'Regional Medical Center': 1200,
                'Specialty Clinic': 2000
            }[hospital]
            
            treatment_cost = {
                'Medication A': 200,
                'Medication B': 300,
                'Medication C': 500,
                'Surgery': 5000,
                'Physical Therapy': 150,
                'Radiation': 3000,
                'Counseling': 100
            }[primary_treatment]
            
            secondary_cost = sum([{
                'Medication A': 200,
                'Medication B': 300,
                'Medication C': 500,
                'Surgery': 5000,
                'Physical Therapy': 150,
                'Radiation': 3000,
                'Counseling': 100
            }[t] for t in secondary_treatments])
            
            total_cost = base_cost * length_of_stay + treatment_cost + secondary_cost
            
            # Add record
            record = {
                'patient_id': f"P-{random.randint(10000, 99999)}",
                'age': age,
                'age_group': age_group,
                'gender': gender,
                'primary_condition': primary_condition,
                'secondary_conditions': ','.join([c for c in patient_conditions if c != primary_condition]),
                'hospital': hospital,
                'staffing_level': staffing_level,
                'admission_date': admission_date,
                'admission_time': admission_time,
                'discharge_date': discharge_date,
                'discharge_time': discharge_time,
                'length_of_stay': length_of_stay,
                'primary_treatment': primary_treatment,
                'secondary_treatments': ','.join(secondary_treatments),
                'treatment_efficacy': treatment_efficacy,
                'recovery_score': recovery_score,
                'was_readmitted': was_readmitted,
                'total_cost': total_cost
            }
            
            data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df 