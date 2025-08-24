"""
🚗 Car Price Prediction - Machine Learning Regression Project

This project demonstrates various regression techniques for predicting car prices based on features like:
- Brand, Model, Year
- Engine specifications (displacement, horsepower, fuel efficiency)
- Physical attributes (length, width, height, weight)
- Market factors (mileage, condition, fuel type)

Features:
- Multiple regression algorithms (Linear, Ridge, Lasso, Random Forest, XGBoost)
- Feature engineering and preprocessing
- Hyperparameter tuning with GridSearchCV
- Comprehensive model evaluation and comparison
- Interactive prediction interface
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CarPricePredictor:
    """Car Price Prediction using multiple regression algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price'
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic car data for demonstration"""
        print("🚗 Generating sample car data...")
        
        # Car brands and models
        brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Nissan']
        models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander'],
            'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot'],
            'Ford': ['Focus', 'Fusion', 'Escape', 'Explorer'],
            'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
            'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE'],
            'Audi': ['A4', 'A6', 'Q5', 'Q7'],
            'Volkswagen': ['Golf', 'Passat', 'Tiguan', 'Atlas'],
            'Nissan': ['Sentra', 'Altima', 'Rogue', 'Murano']
        }
        
        fuel_types = ['Gasoline', 'Diesel', 'Hybrid', 'Electric']
        conditions = ['Excellent', 'Good', 'Fair', 'Poor']
        
        data = []
        
        for _ in range(n_samples):
            brand = np.random.choice(brands)
            model = np.random.choice(models[brand])
            year = np.random.randint(2010, 2025)
            mileage = np.random.randint(1000, 150000)
            condition = np.random.choice(conditions, p=[0.2, 0.4, 0.3, 0.1])
            fuel_type = np.random.choice(fuel_types, p=[0.6, 0.2, 0.15, 0.05])
            
            # Engine specifications
            displacement = np.random.uniform(1.0, 5.0)
            horsepower = int(displacement * 80 + np.random.normal(0, 20))
            fuel_efficiency = np.random.uniform(20, 40)
            
            # Physical attributes
            length = np.random.uniform(4.0, 5.5)
            width = np.random.uniform(1.7, 2.1)
            height = np.random.uniform(1.4, 1.8)
            weight = np.random.uniform(1200, 2500)
            
            # Base price calculation with some randomness
            base_price = 15000 + (year - 2010) * 1000
            brand_multiplier = {
                'Toyota': 1.0, 'Honda': 1.1, 'Ford': 0.9,
                'BMW': 1.8, 'Mercedes': 2.0, 'Audi': 1.7,
                'Volkswagen': 1.2, 'Nissan': 1.0
            }
            
            condition_multiplier = {
                'Excellent': 1.2, 'Good': 1.0, 'Fair': 0.8, 'Poor': 0.6
            }
            
            fuel_multiplier = {
                'Gasoline': 1.0, 'Diesel': 1.1, 'Hybrid': 1.3, 'Electric': 1.5
            }
            
            # Calculate final price
            price = (base_price * brand_multiplier[brand] * 
                    condition_multiplier[condition] * fuel_multiplier[fuel_type] *
                    (1 - mileage/200000) * (1 + np.random.normal(0, 0.1)))
            
            data.append({
                'brand': brand,
                'model': model,
                'year': year,
                'mileage': mileage,
                'condition': condition,
                'fuel_type': fuel_type,
                'displacement': displacement,
                'horsepower': horsepower,
                'fuel_efficiency': fuel_efficiency,
                'length': length,
                'width': width,
                'height': height,
                'weight': weight,
                'price': max(5000, price)
            })
        
        self.data = pd.DataFrame(data)
        print(f"✅ Generated {n_samples} car records")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("🔧 Preprocessing data...")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Encode categorical variables
        categorical_columns = ['brand', 'model', 'condition', 'fuel_type']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Select numerical features
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        # Split features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train, X_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print(f"✅ Data preprocessed: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"✅ Features: {len(self.feature_columns)} numerical features")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """Train multiple regression models"""
        print("🤖 Training regression models...")
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"   Training {name}...")
            
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
            
            self.models[name] = model
        
        print("✅ All models trained successfully")
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("📊 Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'RMSE': rmse
            }
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(results).T
        print("✅ Model evaluation completed")
        
        return self.results_df
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for a specific model"""
        print(f"🎯 Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
        else:
            print(f"❌ Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            grid_search.fit(self.X_train_scaled, self.y_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {-grid_search.best_score_:.2f}")
        
        # Update the model with best parameters
        self.models[f'{model_name} (Tuned)'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def predict_price(self, car_features):
        """Predict car price using the best performing model"""
        # Find best model based on R² score
        best_model_name = self.results_df['R²'].idxmax()
        best_model = self.models[best_model_name]
        
        # Prepare features
        features = np.array(car_features).reshape(1, -1)
        
        # Scale features if needed
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            features_scaled = self.scaler.transform(features)
            prediction = best_model.predict(features_scaled)[0]
        else:
            prediction = best_model.predict(features)[0]
        
        return prediction, best_model_name
    
    def plot_results(self):
        """Create visualization plots"""
        print("📈 Creating visualization plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚗 Car Price Prediction - Model Performance Analysis', fontsize=16)
        
        # 1. Model Performance Comparison
        metrics = ['R²', 'RMSE', 'MAE']
        x = np.arange(len(self.models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [self.results_df.loc[model, metric] for model in self.models.keys()]
            if metric == 'R²':
                axes[0, 0].bar(x + i*width, values, width, label=metric, alpha=0.8)
            else:
                axes[0, 0].bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(list(self.models.keys()), rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[0, 1].set_yticks(range(len(feature_importance)))
            axes[0, 1].set_yticklabels(feature_importance['feature'])
            axes[0, 1].set_xlabel('Feature Importance')
            axes[0, 1].set_title('Random Forest Feature Importance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted (Best Model)
        best_model_name = self.results_df['R²'].idxmax()
        best_model = self.models[best_model_name]
        
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            y_pred = best_model.predict(self.X_test_scaled)
        else:
            y_pred = best_model.predict(self.X_test)
        
        axes[1, 0].scatter(self.y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Predicted Price')
        axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals Plot
        residuals = self.y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualization plots created successfully")
    
    def generate_sample_prediction(self):
        """Generate a sample prediction with random car features"""
        print("\n🎯 Sample Car Price Prediction:")
        
        # Create sample car features
        sample_car = {
            'brand': 'Toyota',
            'model': 'Camry',
            'year': 2020,
            'mileage': 50000,
            'condition': 'Good',
            'fuel_type': 'Gasoline',
            'displacement': 2.5,
            'horsepower': 200,
            'fuel_efficiency': 30,
            'length': 4.9,
            'width': 1.8,
            'height': 1.5,
            'weight': 1600
        }
        
        # Encode categorical features
        features = []
        for col in self.feature_columns:
            if col in ['brand', 'model', 'condition', 'fuel_type']:
                le = self.label_encoders[col]
                features.append(le.transform([sample_car[col]])[0])
            else:
                features.append(sample_car[col])
        
        # Make prediction
        predicted_price, best_model = self.predict_price(features)
        
        print(f"🚗 Car: {sample_car['year']} {sample_car['brand']} {sample_car['model']}")
        print(f"📊 Features: {sample_car['mileage']} miles, {sample_car['condition']} condition")
        print(f"💰 Predicted Price: ${predicted_price:,.2f}")
        print(f"🤖 Best Model: {best_model}")
        
        return predicted_price, best_model

def main():
    """Main function to run the car price prediction project"""
    print("🚗 Car Price Prediction - Machine Learning Regression Project")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CarPricePredictor()
    
    # Generate sample data
    data = predictor.generate_sample_data(n_samples=1000)
    
    # Display data info
    print(f"\n📊 Dataset Overview:")
    print(f"   Shape: {data.shape}")
    print(f"   Features: {len(data.columns) - 1}")
    print(f"   Target: {predictor.target_column}")
    
    # Show sample data
    print(f"\n📋 Sample Data:")
    print(data.head())
    
    # Show data statistics
    print(f"\n📈 Data Statistics:")
    print(data.describe())
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test = predictor.preprocess_data()
    
    # Train models
    models = predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Display results
    print(f"\n🏆 Model Performance Results:")
    print(results.round(4))
    
    # Find best model
    best_model = results['R²'].idxmax()
    print(f"\n🥇 Best Performing Model: {best_model}")
    print(f"   R² Score: {results.loc[best_model, 'R²']:.4f}")
    print(f"   RMSE: ${results.loc[best_model, 'RMSE']:,.2f}")
    
    # Hyperparameter tuning for Random Forest
    print(f"\n🎯 Hyperparameter Tuning:")
    tuned_rf = predictor.hyperparameter_tuning('Random Forest')
    
    # Re-evaluate with tuned model
    if tuned_rf:
        predictor.evaluate_models()
        print(f"\n🏆 Updated Results (with tuned Random Forest):")
        print(predictor.results_df.round(4))
    
    # Generate sample prediction
    predicted_price, best_model = predictor.generate_sample_prediction()
    
    # Create visualizations
    predictor.plot_results()
    
    print(f"\n✅ Car Price Prediction Project Completed Successfully!")
    print(f"📚 This project demonstrates:")
    print(f"   - Multiple regression algorithms")
    print(f"   - Feature engineering and preprocessing")
    print(f"   - Hyperparameter tuning")
    print(f"   - Model evaluation and comparison")
    print(f"   - Visualization and analysis")

if __name__ == "__main__":
    main()
