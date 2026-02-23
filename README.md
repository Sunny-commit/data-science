# 🛡️ Data Science Fundamentals - Complete Guide

A **comprehensive collection** covering data science methodologies, statistical analysis, and ML fundamentals.

## 🎯 Overview

This repository includes:
- ✅ Statistical analysis
- ✅ Exploratory data analysis
- ✅ Data preprocessing
- ✅ Hypothesis testing
- ✅ Probability theory
- ✅ Statistical distributions
- ✅ Best practices

## 📊 Exploratory Data Analysis (EDA)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAToolkit:
    """Comprehensive EDA analysis"""
    
    def __init__(self, dataframe):
        self.df = dataframe
    
    def basic_statistics(self):
        """Calculate descriptive statistics"""
        stats = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes,
            'missing': self.df.isnull().sum(),
            'duplicates': self.df.duplicated().sum(),
            'describe': self.df.describe()
        }
        
        return stats
    
    def missing_data_analysis(self):
        """Analyze missing data"""
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'column': missing_pct.index,
            'missing_count': self.df.isnull().sum().values,
            'missing_percentage': missing_pct.values
        }).sort_values('missing_percentage', ascending=False)
        
        return missing_df[missing_df['missing_percentage'] > 0]
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing data"""
        df_clean = self.df.copy()
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Numerical: mean/median
        if strategy == 'mean':
            for col in numerical_cols:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in numerical_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Categorical: mode
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def outlier_detection(self):
        """Identify outliers using IQR"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers[col] = outlier_mask.sum()
        
        return outliers
    
    def correlation_analysis(self):
        """Correlation matrix and heatmap"""
        numerical_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()
        
        return correlation_matrix
    
    def distribution_analysis(self):
        """Analyze feature distributions"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 4*len(numerical_cols)))
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(self.df[col], bins=30, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
```

## 📈 Statistical Analysis

```python
from scipy import stats

class StatisticalAnalysis:
    """Statistical hypothesis testing"""
    
    @staticmethod
    def t_test(group1, group2):
        """Independent t-test"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def anova_test(groups):
        """One-way ANOVA"""
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(contingency_table):
        """Chi-square test for independence"""
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def pearson_correlation(x, y):
        """Pearson correlation test"""
        corr, p_value = stats.pearsonr(x, y)
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def normality_test(data):
        """Shapiro-Wilk normality test"""
        stat, p_value = stats.shapiro(data)
        
        return {
            'test_statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
```

## 🔄 Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class PreprocessingPipeline:
    """Complete preprocessing workflow"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def drop_duplicates(self, df):
        """Remove duplicate rows"""
        duplicates = df.duplicated()
        df_cleaned = df.drop_duplicates()
        
        print(f"Removed {duplicates.sum()} duplicate rows")
        
        return df_cleaned
    
    def drop_constant_columns(self, df):
        """Remove columns with no variation"""
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        df_cleaned = df.drop(columns=constant_cols)
        
        print(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
        
        return df_cleaned
    
    def encode_categorical(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Standardize numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def full_pipeline(self, df):
        """Complete preprocessing"""
        # Drop duplicates
        df = self.drop_duplicates(df)
        
        # Drop constant columns
        df = self.drop_constant_columns(df)
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        return df
```

## 📊 Probability Distributions

```python
from scipy.stats import norm, binom, poisson, exponential

class ProbabilityDistributions:
    """Common statistical distributions"""
    
    @staticmethod
    def normal_distribution_analysis(data):
        """Analyze normal distribution"""
        mu = np.mean(data)
        sigma = np.std(data)
        
        # Probability within 1 std: ~68%
        # Probability within 2 std: ~95%
        # Probability within 3 std: ~99.7%
        
        return {
            'mean': mu,
            'std': sigma,
            'pdf': norm.pdf(data, mu, sigma),
            'cdf': norm.cdf(data, mu, sigma)
        }
    
    @staticmethod
    def binomial_probability(n, p, k):
        """Binomial probability"""
        prob = binom.pmf(k, n, p)
        
        return prob
    
    @staticmethod
    def poisson_probability(lambda_param, k):
        """Poisson probability"""
        prob = poisson.pmf(k, lambda_param)
        
        return prob
    
    @staticmethod
    def confidence_interval(data, confidence=0.95):
        """Calculate confidence interval"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return {
            'mean': mean,
            'lower_bound': mean - margin,
            'upper_bound': mean + margin
        }
```

## 💡 Interview Talking Points

**Q: When to use t-test vs ANOVA?**
```
Answer:
- t-test: 2 groups comparison
- ANOVA: 3+ groups comparison
- Assumes normal distribution
- Independent vs paired
- Variance assumptions
```

**Q: Data preprocessing importance?**
```
Answer:
- Garbage in, garbage out (GIGO)
- Missing data handling critical
- Outlier impact significant
- Encoding essential for categorical
- Scaling for distance-based models
```

## 🌟 Portfolio Value

✅ EDA mastery
✅ Statistical testing
✅ Data preprocessing
✅ Hypothesis testing
✅ Probability knowledge
✅ Distribution analysis
✅ Best practices

---

**Technologies**: Pandas, NumPy, SciPy, Scikit-learn

