# 📊 Machine Learning Regression Projects

This directory contains regression projects that demonstrate various machine learning techniques for predicting continuous numerical values.

## 🚗 Car Price Prediction Project

**File:** `car_price_prediction.py`

A comprehensive regression project that predicts car prices using multiple machine learning algorithms and advanced techniques.

### 🎯 Project Overview

The Car Price Prediction project demonstrates how to build, train, and evaluate multiple regression models to predict car prices based on various features including:

- **Brand & Model**: Car manufacturer and specific model
- **Year**: Manufacturing year (2010-2025)
- **Mileage**: Distance traveled
- **Condition**: Car condition (Excellent, Good, Fair, Poor)
- **Fuel Type**: Gasoline, Diesel, Hybrid, Electric
- **Engine Specs**: Displacement, horsepower, fuel efficiency
- **Physical Attributes**: Length, width, height, weight

### 🚀 Features

#### **Multiple Regression Algorithms**
- **Linear Regression**: Basic linear relationship modeling
- **Ridge Regression**: Linear regression with L2 regularization
- **Lasso Regression**: Linear regression with L1 regularization
- **Random Forest**: Ensemble method with decision trees
- **XGBoost**: Gradient boosting with extreme gradient boosting

#### **Advanced ML Techniques**
- **Feature Engineering**: Categorical encoding, numerical scaling
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Model Comparison**: Comprehensive performance metrics

#### **Data Generation & Preprocessing**
- **Synthetic Data**: 1000+ realistic car records
- **Smart Pricing Logic**: Realistic price calculation based on features
- **Data Scaling**: StandardScaler for numerical features
- **Label Encoding**: Categorical variable transformation

#### **Evaluation & Visualization**
- **Performance Metrics**: MSE, MAE, R², RMSE
- **Model Comparison**: Side-by-side performance analysis
- **Feature Importance**: Random Forest feature ranking
- **Prediction Analysis**: Actual vs Predicted plots
- **Residual Analysis**: Error distribution analysis

### 🛠️ Installation & Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Project:**
```bash
python car_price_prediction.py
```

### 📊 Sample Output

The project generates comprehensive outputs including:

- **Dataset Overview**: 1000 car records with 13 features
- **Model Performance**: Comparison of all algorithms
- **Best Model Selection**: Automatic identification of top performer
- **Sample Prediction**: Real car price prediction example
- **Visualization Plots**: 4 comprehensive analysis charts

### 🎯 Learning Objectives

This project teaches:

1. **Data Preprocessing**: Handling mixed data types and scaling
2. **Model Selection**: Comparing different regression approaches
3. **Hyperparameter Optimization**: Finding optimal model parameters
4. **Evaluation Metrics**: Understanding model performance
5. **Feature Engineering**: Creating meaningful input features
6. **Visualization**: Creating informative data plots

### 🔧 Customization

#### **Adding New Features**
- Modify the `generate_sample_data()` method
- Add new car attributes and pricing logic
- Update feature engineering in `preprocess_data()`

#### **Adding New Models**
- Import new regression algorithms
- Add to the `models` dictionary in `train_models()`
- Update preprocessing logic if needed

#### **Modifying Data Generation**
- Change car brands, models, and specifications
- Adjust pricing algorithms and multipliers
- Modify data distribution and sample sizes

### 📈 Performance Expectations

Typical performance metrics:
- **R² Score**: 0.85-0.95 (depending on data quality)
- **RMSE**: $2,000-$5,000 (price prediction accuracy)
- **Training Time**: 10-30 seconds for 1000 samples
- **Prediction Time**: <1 second per car

### 🚀 Advanced Usage

#### **Real Data Integration**
Replace synthetic data with real car datasets:
```python
# Load real data
real_data = pd.read_csv('real_car_data.csv')
predictor.data = real_data
```

#### **Custom Predictions**
Make predictions for specific cars:
```python
car_features = [brand_encoded, model_encoded, year, mileage, ...]
predicted_price, model_used = predictor.predict_price(car_features)
```

#### **Model Persistence**
Save trained models for later use:
```python
import joblib
joblib.dump(predictor.models['Random Forest'], 'car_price_model.pkl')
```

### 🔍 Troubleshooting

#### **Common Issues**
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce sample size for large datasets
3. **Plot Display**: Use `plt.show()` in interactive environments
4. **Performance**: Adjust hyperparameter grids for faster training

#### **Performance Optimization**
- Use smaller hyperparameter grids for faster tuning
- Reduce cross-validation folds (cv=3 instead of cv=5)
- Limit tree depth in Random Forest and XGBoost

### 📚 Related Projects

- **Classification**: See `../classification/` for classification algorithms
- **Clustering**: See `../clustering/` for unsupervised learning
- **Anomaly Detection**: See `../anomaly_detection/` for outlier detection

### 🤝 Contributing

Feel free to:
- Add new regression algorithms
- Improve data generation logic
- Enhance visualization capabilities
- Optimize hyperparameter tuning
- Add new evaluation metrics

---

**Happy Learning! 🎓✨**
