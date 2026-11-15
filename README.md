# OIBSIP_dataanalystics_task2: Predicting House Prices with Linear Regression

## 📌 Objective
Build and evaluate a Multiple Linear Regression model to predict house prices based on property features (such as area, number of bedrooms, bathrooms, stories, parking, and amenities), developing end-to-end machine learning skills from exploratory data analysis to model interpretation.

## 🛠️ Tools & Technologies Used
* **Programming Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

## 🔍 Steps Performed
1. **Data Loading & Exploratory Data Analysis (EDA):**
   * Loaded the dataset (`545 rows x 13 columns`) and verified zero missing values and zero duplicate rows.
   * Computed descriptive statistics and visualized the target variable (`price`), observing a right-skewed distribution (skewness = `1.21`).
2. **Feature Selection & Preprocessing:**
   * Evaluated numerical and categorical predictors based on domain relevance.
   * Converted categorical variables (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`) into numerical format using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
3. **Correlation Analysis:**
   * Generated a feature correlation heatmap and ranked predictors by correlation strength with `price`, identifying `area` (`r = 0.54`), `bathrooms` (`r = 0.52`), and `airconditioning_yes` (`r = 0.45`) as the strongest positive correlates.
4. **Model Training & Evaluation:**
   * Split the dataset into **80% training (436 samples)** and **20% testing (109 samples)** sets (`random_state=42`).
   * Trained a Scikit-learn `LinearRegression` model and evaluated performance on unseen test data.
5. **Visualization & Coefficient Interpretation:**
   * Plotted Actual vs. Predicted house prices against a diagonal reference line.
   * Analyzed model coefficients to determine the features driving the largest positive and negative price impacts.

## 💡 Outcome in Brief
* **Model Performance:**
  * **R-squared ($R^2$):** `0.6529` (the model explains **65.3%** of the variance in house prices).
  * **Root Mean Squared Error (RMSE):** `1,324,506.96`.
  * **Mean Squared Error (MSE):** `1,754,318,687,330.66`.
* **Key Feature Insights:**
  * **Top Positive Price Drivers:** Number of `bathrooms`, `airconditioning_yes`, `hotwaterheating_yes`, and `prefarea_yes`.
  * **Top Negative Price Driver:** `furnishingstatus_unfurnished` (unfurnished properties sell at a significant discount compared to furnished homes).
  * 
