import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

class AccidentInjuryPredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=None, 
                max_features=None, 
                min_samples_leaf=4, 
                min_samples_split=10,
                random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                min_samples_split=2, 
                min_samples_leaf=1, 
                max_features='log2', 
                max_depth=None,
                random_state=42
            ),
            'SVR': SVR(
                kernel='rbf', 
                C=1, 
                gamma='auto'
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300, 
                learning_rate=0.3, 
                max_depth=3, 
                min_samples_split=2, 
                subsample=1.0,
                random_state=42
            )
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = float('-inf')
    
    def load_and_preprocess_data(self, file_path):
        """
        Comprehensive data preprocessing pipeline.
        """
        # Load data
        df = pd.read_excel(file_path)
        
        # Temporal Features
        df['AccidentDate'] = pd.to_datetime(df['AccidentDate'])
        df['Month'] = df['AccidentDate'].dt.month
        df['DayOfWeek'] = df['AccidentDate'].dt.dayofweek
        df['Hour'] = df['AccidentTime '].astype(int)
        
        # Handle Numeric Features
        numeric_columns = ['PerpetratorAge', 'VictimAge', 'Deaths', 'Injured']
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df[column] = df[column].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
            if column in ['PerpetratorAge', 'VictimAge']:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mean())
        
        # Handle Categorical Features
        categorical_columns = [
            'District', 'AccidentDetails', 'PerpetratorGender', 
            'PerpetratorVehicle', 'PerpetratorBodyInjuryLevel',
            'VictimGender', 'VictimVehicle', 'VictimBodyInjuryLevel',
            'AccidentType', 'LawViolation', 'Weather'
        ]
        for column in categorical_columns:
            if df[column].dtype == 'O':
                valid_mode = df[column][~df[column].isin(['Unknown', np.nan])].mode()[0]
                df[column] = df[column].replace(['Unknown', np.nan], valid_mode)
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        # Create features and target
        feature_columns = ['District', 'Month', 'DayOfWeek', 'Hour', 'Deaths',
                          'PerpetratorAge', 'VictimAge'] + categorical_columns
        X = df[feature_columns]
        y = df['Injured']
        
        if abs(y.skew()) > 1:
            y = np.log1p(y)
            print("Applied log transformation to target variable")
        
        return X, y, feature_columns

    def analyze_features(self, X, y, feature_columns):
        """
        Comprehensive feature analysis and selection.
        """
        correlation_matrix = pd.DataFrame(X, columns=feature_columns)
        correlation_matrix['Injured'] = y
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        correlations = correlation_matrix.corr()['Injured'].sort_values(ascending=False)
        correlation_threshold = 0.2
        top_correlated = correlations[abs(correlations) > correlation_threshold].index.tolist()
        top_correlated.remove('Injured')
        
        rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=10)
        rfe.fit(X, y)
        rfe_selected = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
        
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
        plt.title('Top 10 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        rf_selected = feature_importance['Feature'].head(10).tolist()
        selected_features = list(dict.fromkeys(top_correlated + rfe_selected + rf_selected))
        print("\nFinal Selected Features:")
        for feature in selected_features:
            print(f"- {feature}")
        return selected_features

    def prepare_train_test_split(self, X, y, selected_features):
        """
        Prepare data for modeling with proper scaling.
        """
        X = X[selected_features]
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print("\nData split dimensions:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate models with pre-tuned parameters.
        """
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = r2_score(y_test, y_pred)
            
            n = X_test.shape[0]
            p = X_test.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, 
                                             cv=5, scoring='neg_mean_squared_error')).mean()
            cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
            
            results[name] = {
                'RMSE': rmse, 
                'MAE': mae, 
                'R2': r2, 
                'Adjusted_R2': adjusted_r2,
                'Train_RMSE': train_rmse, 
                'Test_RMSE': rmse, 
                'CV_RMSE': cv_rmse, 
                'CV_R2': cv_r2
            }
            
            if cv_r2 > self.best_score:
                self.best_score = cv_r2
                self.best_model = model
        
        return results
    
    def visualize_results(self, results):
        """
        Visualize model performance results.
        """
        # Core metrics comparison
        metrics = ['RMSE', 'MAE', 'R2', 'Adjusted_R2']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            x = np.arange(len(results))
            values = [results[model][metric] for model in results]
            
            axes[idx].bar(x, values)
            axes[idx].set_xlabel('Models')
            axes[idx].set_ylabel(metric)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(results.keys(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png')
        plt.close()
        
        # Training vs Test Performance
        plt.figure(figsize=(10, 6))
        x = np.arange(len(results))
        width = 0.35
        
        plt.bar(x - width/2, [results[model]['Train_RMSE'] for model in results], 
               width, label='Training RMSE')
        plt.bar(x + width/2, [results[model]['Test_RMSE'] for model in results], 
               width, label='Test RMSE')
        
        plt.xlabel('Models')
        plt.ylabel('RMSE Score')
        plt.title('Training vs Test Performance')
        plt.xticks(x, results.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('train_test_comparison.png')
        plt.close()
        
        # Cross-validation Performance
        plt.figure(figsize=(10, 6))
        plt.bar(x, [results[model]['CV_R2'] for model in results])
        plt.xlabel('Models')
        plt.ylabel('Cross-validation R² Score')
        plt.title('Cross-Validation Performance')
        plt.xticks(x, results.keys(), rotation=45)
        plt.tight_layout()
        plt.savefig('cv_performance.png')
        plt.close()

if __name__ == "__main__":
    predictor = AccidentInjuryPredictor()
    
    # Load and preprocess data
    X, y, feature_columns = predictor.load_and_preprocess_data("C:/Users/Asus/Downloads/RoadAccidentSeoul.xlsx")
    
    # Analyze and select features
    selected_features = predictor.analyze_features(X, y, feature_columns)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = predictor.prepare_train_test_split(X, y, selected_features)
    
    # Train and evaluate models
    results = predictor.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    predictor.visualize_results(results)

    # Print final results
    print("\nModel Performance Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    print(f"\nBest performing model: {type(predictor.best_model).__name__}")