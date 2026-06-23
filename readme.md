# Melbourne Housing Price Prediction
Predicting Melbourne House property prices from 2016-2018 using 7 regression models. Includes data set cleaning, feature engineering, trend visulisation and model comparisons.

## Results
| Model | R² | CV R² | MAE |
|---|---|---|---|
| XGBoost | 78.63% | 78.22% | $143,347 |
| Random Forest | 76.18% | 76.09% | $154,239 |
| Bagging | 76.08% | 75.77% | $154,013 |
| Gradient Boosting | 74.07% | 74.91% | $160,417 |
| Linear Regression | 63.46% | 65.39% | $201,964 |
| KNN | 61.30% | 62.53% | $200,380 |
| Decision Tree | 56.38% | 54.53% | $199,315 |

- XGBoost performed with a R² OF 78.63% an a MAE of $143,347.

## What it does
- Preprocesses raw Melbourne housing data removing missing values and outliers
- Engineers features. (Property age, rooms per bath, sale year, sale month)
- Data analysis on features correlation with housing price
- Trains and compares 7 models using sklearn pipeline
- Evaluates models based on MAE, RMSE, R² and 5 fold CV
- Plots key trends in dataset
  
## Findings
- Southern Metropolitan region is the most expensive with Western Victoria being the cheapest.
- Distance from the CBD has a unexpectedly weak correlation (-0.21) with house price.
- Regionname, type of housing and council area were the most important drivers for house prices,explaining distances weak correlation.
- XGBoost outperformed the other models with a R² of 78.63% an a MAE of $143,347.


## Limitations
- Data is from 2016-2018 and due to significantly higher market prices cannot extrapolate model.
- An accuarate model for current prices would need current data from APIs.
- 6,500 datapoints were removed potentially introducing bias.
- No school zone data

## Dataset
- Dataset - https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
- Dataset contained 13580 properties, after cleaning and removing outliers only 7096 properties remained
- Range $145,000 - $2,350,000

## Tech stack
- Pandas, Numpy
- sckit-learn
- XGBoost
- Matplotlib, Seaborn
  
## How to run?
1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/melbourne-housing-regression.git
   cd melbourne-housing-regression
   
2. pip install -r requirements.txt
   
3. Place melb_data.csv in the root folder

4. Run melbourne_regression.py
