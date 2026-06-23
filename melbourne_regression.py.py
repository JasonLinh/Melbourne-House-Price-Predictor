import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ── Settings ─────────────────────────────────────
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print('=' * 55)
print('   Melbourne Housing Price Analysis')
print('=' * 55)

# ═══════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════

df = pd.read_csv('melb_data.csv')
print(f'\nOriginal dataset: {df.shape[0]} houses, {df.shape[1]} columns')

# ═══════════════════════════════════════════════════
# STEP 2: CLEAN DATA
# ═══════════════════════════════════════════════════


df = df.drop(['Address', 'SellerG', 'Method', 'Postcode', 'Propertycount'], axis=1)


df['Car'] = df['Car'].fillna(0)
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
df['CouncilArea'] = df['CouncilArea'].fillna('Unknown')
df['Bedroom2'] = df['Bedroom2'].fillna(df['Bedroom2'].median())
df['Bathroom'] = df['Bathroom'].fillna(df['Bathroom'].median())


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['SaleYear'] = df['Date'].dt.year
df['SaleMonth'] = df['Date'].dt.month
df['PropertyAge'] = df['SaleYear'] - df['YearBuilt']
df['RoomsPerBath'] = df['Rooms'] / (df['Bathroom'] + 1)
df = df.drop('Date', axis=1)


df = df[df['Price'] > 100000]
df = df[df['Landsize'] < 10000]
df = df[df['BuildingArea'] < 2000]
df = df[df['BuildingArea'] > 0]
df = df[df['Rooms'] <= 10]

print(f'After basic cleaning: {df.shape[0]} houses')


# ═══════════════════════════════════════════════════
# STEP 3: REMOVE OUTLIERS (IQR across all key columns)
# ═══════════════════════════════════════════════════

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    total_removed = 0
    for col in columns:
        before = len(df_clean)
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        removed = before - len(df_clean)
        total_removed += removed
    print(f'  Outlier removal: {total_removed} rows removed across {len(columns)} columns')
    return df_clean


outlier_cols = ['Price', 'Landsize', 'BuildingArea', 'Bathroom', 'Car']
df = remove_outliers_iqr(df, outlier_cols)
print(f'After outlier removal: {df.shape[0]} houses')
print(f'Price range: ${df["Price"].min():,.0f} — ${df["Price"].max():,.0f}')

# ═══════════════════════════════════════════════════
# STEP 4: EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════

print('\n=== KEY STATISTICS ===')
print(f'Average price:  ${df["Price"].mean():,.0f}')
print(f'Median price:   ${df["Price"].median():,.0f}')

region_prices = df.groupby('Regionname')['Price'].median().sort_values(ascending=False)
print(f'Most expensive: {region_prices.index[0]} (${region_prices.values[0]:,.0f})')
print(f'Most affordable:{region_prices.index[-1]} (${region_prices.values[-1]:,.0f})')

numeric_cols = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom',
                'Car', 'Landsize', 'BuildingArea', 'PropertyAge']
correlations = df[numeric_cols].corrwith(df['Price']).sort_values(ascending=False)

print('\n=== CORRELATIONS WITH PRICE ===')
print(correlations.round(3))

# ── Chart 1: Price Distribution ──────────────────
plt.figure()
sns.histplot(df['Price'], bins=50, color='#2196F3', alpha=0.8)
plt.axvline(df['Price'].mean(), color='red', linestyle='--', label=f'Mean:   ${df["Price"].mean():,.0f}')
plt.axvline(df['Price'].median(), color='green', linestyle='--', label=f'Median: ${df["Price"].median():,.0f}')
plt.title('Distribution of Melbourne House Prices', fontsize=15, fontweight='bold')
plt.xlabel('Price ($AUD)')
plt.ylabel('Number of Houses')
plt.legend()
plt.tight_layout()
plt.savefig('chart1_price_distribution.png', dpi=150)
plt.show()

# ── Chart 2: Price by Rooms ───────────────────────
plt.figure()
room_prices = df.groupby('Rooms')['Price'].median()
bars = plt.bar(room_prices.index, room_prices.values, color='#FF5722', edgecolor='white')
for bar, val in zip(bars, room_prices.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
             f'${val / 1e6:.2f}M', ha='center', fontsize=9, fontweight='bold')
plt.title('Median Price by Number of Rooms', fontsize=15, fontweight='bold')
plt.xlabel('Number of Rooms')
plt.ylabel('Median Price ($AUD)')
plt.ylim(0, room_prices.max() * 1.12)  # headroom so labels never clip
plt.tight_layout()
plt.savefig('chart2_price_by_rooms.png', dpi=150)
plt.show()

# ── Chart 3: Price by Region ──────────────────────
plt.figure(figsize=(14, 6))
sns.barplot(x=region_prices.values, y=region_prices.index, color='#4CAF50')
plt.title('Median House Price by Melbourne Region', fontsize=15, fontweight='bold')
plt.xlabel('Median Price ($AUD)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('chart3_price_by_region.png', dpi=150)
plt.show()

# ── Chart 4: Distance vs Price ────────────────────
plt.figure()
sample = df.sample(2000, random_state=42)
sns.regplot(data=sample, x='Distance', y='Price',
            scatter_kws={'alpha': 0.3, 's': 10},
            line_kws={'color': 'red', 'linewidth': 2},
            color='#9C27B0')
plt.title(f'Distance from CBD vs House Price\n(correlation: {df["Distance"].corr(df["Price"]):.2f})',
          fontsize=15, fontweight='bold')
plt.xlabel('Distance from CBD (km)')
plt.ylabel('Price ($AUD)')
plt.tight_layout()
plt.savefig('chart4_distance_vs_price.png', dpi=150)
plt.show()

# ── Chart 5: Boxplot by Type ──────────────────────
plt.figure()
type_labels = {'h': 'House', 'u': 'Unit', 't': 'Townhouse'}
df['TypeLabel'] = df['Type'].map(type_labels)
sns.boxplot(data=df, x='TypeLabel', y='Price',
            palette=['#2196F3', '#FF5722', '#4CAF50'],
            order=['House', 'Unit', 'Townhouse'])
plt.title('Price Distribution by Property Type', fontsize=15, fontweight='bold')
plt.xlabel('Property Type')
plt.ylabel('Price ($AUD)')
plt.tight_layout()
plt.savefig('chart5_price_by_type.png', dpi=150)
plt.show()

# ── Chart 6: Correlation heatmap ─────────────────
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_cols + ['Price']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('chart6_correlation_heatmap.png', dpi=150)
plt.show()

# ── Chart 7: Correlation bar chart ───────────────
plt.figure(figsize=(10, 6))
correlations.plot(kind='bar',
                  color=['#4CAF50' if x > 0 else '#F44336' for x in correlations])
plt.title('How Much Each Feature Correlates with Price', fontsize=15, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Correlation with Price')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart7_correlations.png', dpi=150)
plt.show()

print('\n=== DISTANCE INSIGHT ===')
far_expensive = df[df['Distance'] > 20].nlargest(5, 'Price')[['Suburb', 'Distance', 'Price']]
print('Expensive houses far from CBD:')
print(far_expensive.to_string(index=False))
close_cheap = df[df['Distance'] < 10].nsmallest(5, 'Price')[['Suburb', 'Distance', 'Price']]
print('\nCheap houses close to CBD:')
print(close_cheap.to_string(index=False))
print('\nInsight: Distance alone is a weak predictor (-0.17 correlation)')
print('because expensive suburbs exist at ALL distances from CBD.')

# ═══════════════════════════════════════════════════
# STEP 5: PREPARE DATA FOR ML
# ═══════════════════════════════════════════════════

features = [
    'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom',
    'Car', 'Landsize', 'BuildingArea', 'PropertyAge',
    'Regionname', 'CouncilArea', 'SaleYear', 'SaleMonth',
    'RoomsPerBath'
]

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_cols = ['Type', 'Regionname', 'CouncilArea']
numeric_feature_cols = [c for c in features if c not in categorical_cols]

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_feature_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

print(f'\nTraining set: {len(X_train)} houses')
print(f'Testing set:  {len(X_test)} houses')

# ═══════════════════════════════════════════════════
# STEP 6: TRAIN AND COMPARE MODELS
# ═══════════════════════════════════════════════════

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Bagging': BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ),
}

results = []
trained_models = {}

print('\nTraining models — this may take 2-3 minutes...\n')

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocess),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    trained_models[name] = pipeline

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring='r2'
    )
    cv_mean = round(cv_scores.mean() * 100, 2)
    cv_std = round(cv_scores.std() * 100, 2)

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'R² %': round(r2 * 100, 2),
        'CV Score %': cv_mean,
        'CV Std %': cv_std,
    })

    print(f'{name}')
    print(f'  MAE:        ${mae:,.0f}')
    print(f'  RMSE:       ${rmse:,.0f}')
    print(f'  R² Score:   {r2:.3f} ({round(r2 * 100, 2)}%)')
    print(f'  CV Score:   {cv_mean}% ± {cv_std}%')
    print()

# ═══════════════════════════════════════════════════
# STEP 7: COMPARE MODELS
# ═══════════════════════════════════════════════════

results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)

print('=' * 65)
print('MODEL COMPARISON')
print('=' * 65)
print(results_df[['Model', 'R² %', 'CV Score %', 'CV Std %', 'MAE', 'RMSE']].to_string(index=False))
print(f'\n🏆 Best model: {results_df.iloc[0]["Model"]}')
print(f'   R²:        {results_df.iloc[0]["R² %"]}%')
print(f'   CV Score:  {results_df.iloc[0]["CV Score %"]}%')
print(f'   MAE:       ${results_df.iloc[0]["MAE"]:,.0f}')
print(f'   RMSE:      ${results_df.iloc[0]["RMSE"]:,.0f}')

# Chart — R² comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors_r2 = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(results_df))]
bars = axes[0].bar(results_df['Model'], results_df['R2 Score'],
                   color=colors_r2, edgecolor='white')
axes[0].set_title('R² Score (higher = better)\nGreen = best', fontsize=13, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)
for bar, val in zip(bars, results_df['R2 Score']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

best_mae_idx = results_df['MAE'].values.argmin()
colors_mae = ['#4CAF50' if i == best_mae_idx else '#F44336'
              for i in range(len(results_df))]
bars2 = axes[1].bar(results_df['Model'], results_df['MAE'],
                    color=colors_mae, edgecolor='white')
axes[1].set_title('Mean Absolute Error (lower = better)\nGreen = best', fontsize=13, fontweight='bold')
axes[1].set_ylabel('MAE ($AUD)')
axes[1].tick_params(axis='x', rotation=30)
for bar, val in zip(bars2, results_df['MAE']):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                 f'${val:,.0f}', ha='center', fontweight='bold', fontsize=8)

plt.suptitle('Model Comparison: Melbourne Housing Price Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart8_model_comparison.png', dpi=150)
plt.show()

# Cross validation comparison
plt.figure(figsize=(12, 6))
x = range(len(results_df))
plt.bar(x, results_df['CV Score %'], color='#9C27B0', alpha=0.7, label='CV Score %')
plt.errorbar(x, results_df['CV Score %'], yerr=results_df['CV Std %'],
             fmt='none', color='black', capsize=5, linewidth=2)
plt.xticks(x, results_df['Model'], rotation=30, ha='right')
plt.title('Cross Validation Scores (5-Fold)\nError bars show variance — lower variance = more reliable',
          fontsize=13, fontweight='bold')
plt.ylabel('CV R² Score %')
plt.legend()
plt.tight_layout()
plt.savefig('chart9_cross_validation.png', dpi=150)
plt.show()

# ═══════════════════════════════════════════════════
# STEP 8: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════

best_model_name = results_df.iloc[0]['Model']

best_pipeline = trained_models[best_model_name]
best_model = best_pipeline.named_steps['model']

if hasattr(best_model, 'feature_importances_'):
    ohe_feature_names = (
        best_pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    all_feature_names = numeric_feature_cols + ohe_feature_names

    raw_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': best_model.feature_importances_
    })


    def get_original_name(feature):
        for cat in categorical_cols:
            if feature.startswith(cat + '_'):
                return cat
        return feature

    raw_importance_df['OriginalFeature'] = raw_importance_df['Feature'].apply(get_original_name)
    importance_df = (
        raw_importance_df
        .groupby('OriginalFeature')['Importance']
        .sum()
        .reset_index()
        .rename(columns={'OriginalFeature': 'Feature'})
        .sort_values('Importance', ascending=False)
    )

    print(f'\n=== FEATURE IMPORTANCE ({best_model_name}) ===')
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 7))
    colors_imp = ['#4CAF50' if i < 3 else '#2196F3'
                  for i in range(len(importance_df))]
    plt.barh(importance_df['Feature'], importance_df['Importance'],
             color=colors_imp, edgecolor='white')
    plt.title(f'What Drives Melbourne House Prices?\n({best_model_name} — green = top 3)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('chart10_feature_importance.png', dpi=150)
    plt.show()

# ═══════════════════════════════════════════════════
# STEP 9: ACTUAL VS PREDICTED
# ═══════════════════════════════════════════════════


best_predictions = best_pipeline.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_predictions, alpha=0.3, color='#2196F3', s=8)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.title(f'Actual vs Predicted Prices ({best_model_name})\nR² = {results_df.iloc[0]["R2 Score"]:.3f}',
          fontsize=14, fontweight='bold')
plt.xlabel('Actual Price ($AUD)')
plt.ylabel('Predicted Price ($AUD)')
plt.legend()
plt.tight_layout()
plt.savefig('chart11_actual_vs_predicted.png', dpi=150)
plt.show()

# ═══════════════════════════════════════════════════
# STEP 10: CONCLUSION
# ═══════════════════════════════════════════════════

print('\n' + '=' * 55)
print('CONCLUSION')
print('=' * 55)
print(f"""
Dataset: {len(df):,} Melbourne properties (2016-2018)
Best model: {best_model_name}
R²:          {results_df.iloc[0]['R² %']}%
CV R²:       {results_df.iloc[0]['CV Score %']}%
MAE:         ${results_df.iloc[0]['MAE']:,.0f} average prediction error
RMSE:        ${results_df.iloc[0]['RMSE']:,.0f}

Key findings:
    -South Metropolitan area is most expensive
    -Number of rooms has the strongest correlation to price
    -Distance from CBD is a surprisingly weak feature
    -XGBoost performed the best
    
Limitations:
    -Data is from 2016-2018 and due to significantly
    higher market prices cannot extrapolate model.
    -An accuarate model for current prices would need 
    current data from APIs
""")