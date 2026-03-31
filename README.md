# AutoPrice ML: Vehicle Price Prediction using Regression Model

### Overview
AutoPrice ML is a predictive machine learning pipeline designed to estimate the market value of vehicles based on their specifications. This project demonstrates proficiency in handling continuous target variables (Regression), data imputation, and high-dimensional feature encoding.

### Business Impact
* Built a pricing engine capable of evaluating 1,000+ vehicles with complex, mixed-data-type features.
* Achieved an R-squared score of **0.7842**, successfully capturing nearly 80% of the variance in vehicle pricing logic without human intervention.
* Maintained a Mean Absolute Error (MAE) of **~$4,537**, providing a reliable baseline for automated dealership pricing algorithms.

### Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)

### Methodology
1. **Data Cleaning & Imputation:** Handled missing continuous variables (mileage) using median imputation and missing categorical variables (cylinders, doors) using mode imputation.
2. **Feature Engineering:** Transformed raw text data (make, model, trim, color) into a machine-readable format using **One-Hot Encoding**, expanding the feature space to 887 distinct binary columns.
3. **Model Training:** Deployed a Random Forest Regressor using 100 decision trees to capture non-linear relationships between vehicle specs and price.

### How to Run Locally
1. Clone this repository.
2. Install the required dependencies: `pip install pandas numpy scikit-learn`
3. Run `python 4_train_model.py` to train the regression model and output the MAE and R2 metrics.

## Author
Kamal P
GitHub: https://github.com/15-Kamal