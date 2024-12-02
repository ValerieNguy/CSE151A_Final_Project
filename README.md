# CSE151A Final Project

## Data Exploration

The dataset we are using contains 6,405,008 observations (each observation being a taxi trip record) with 18 features. 

As stated on the Kaggle dataset, the columns include:
   * VendorID: A unique identifier for the taxi vendor or service provider.
   * tpep_pickup_datetime: The date and time when the passenger was picked up.
   * tpep_dropoff_datetime: The date and time when the passenger was dropped off.
   * passenger_count: The number of passengers in the taxi.
   * trip_distance: The total distance of the trip in miles or kilometers.
   * RatecodeID: The rate code assigned to the trip, representing fare types.
   * store_and_fwd_flag: Indicates whether the trip data was stored locally and then forwarded later (Y/N).
   * PULocationID: The unique identifier for the pickup location (zone or area).
   * DOLocationID: The unique identifier for the drop-off location (zone or area).
   * payment_type: The method of payment used by the passenger (e.g., cash, card).
   * fare_amount: The base fare for the trip.
   * extra: Additional charges applied during the trip (e.g., night surcharge).
   * mta_tax: The tax imposed by the Metropolitan Transportation Authority.
   * tip_amount: The tip given to the driver, if applicable.
   * tolls_amount: The total amount of tolls charged during the trip.
   * improvement_surcharge: A surcharge imposed for the improvement of services.
   * total_amount: The total fare amount, including all charges and surcharges.
   * congestion_surcharge: An additional charge for trips taken during high traffic congestion times.

When analyzing the correlation heatmap, we noticed high correlations between the different fee amounts, which include fare_amount, total_amount, tip_amount, and toll_amount. Additionally, there is a negative correlation between extra charges and VendorID which may suggest that different pricing structures among vendor. Other correlationships include a negative relationship between payment_type and tip_amount and a negative relationship between improvement_surcharge and mta_tax. This relationships could be worth further analyzing to determine if they could impact fare.

Looking into data distribution details, we noticed potential outliers in trip_distance, fare_amount, extra, MTA_tax, total_amount, tolls_amount, and congestion_surcharge. These will be handled in data preprocessing as described below.


## Data Preprocessing

Since we got our dataset directly from Kaggle, the dataset we are using has already had an initial cleaning by the author, which includes standard formatting and handling missing or null values. However, to fully optimize the data and improve the accuracy of our predictive model, we are planning to add some more preprocessing steps are helpful towards our model. These following steps will be taken to get rid of any remaining inconsistencies, such as any missing or null datas and outliers (e.g., negative values in `fare_amount` and `trip_distance`).

### Planned Preprocessing Steps

1. **Handling Missing or Null Values**:  
   We have already found out that we have around 300,000 null datas out of 6.5 million datas  and our goal is to examine each column for any remaining miss or null values and determine an appropriate strategy to address them. Depending on the feature, we are planning to choose to drop rows with null values.

2. **Outlier Removal**:  
   Outliers in key features such as `fare_amount` and `trip_distance` can impact the model's performance negatively because of some negative datas that does not make sense For example, we found out that we have some negative values of distance and taxi fares which does not make sense. To deal with this, we will filter out extreme values by setting reasonable thresholds, such as capping `trip_distance` at 50 miles and limiting `fare_amount` to a common maximum.

3. **Normalization and Scaling**:  
   To ensure consistent weighting across features during model training, we will normalize or scale numerical columns such as `trip_distance`, `fare_amount`, and `total_amount`. Depending on the our model's design and performance, we may use standardization or min-max scaling.


## Milestone 3: Preprocessing and First Model

### Summary of Changes
1. **Data Cleaning**:
   - Dropped rows with missing values (~1% of observations).
   - Removed rows where:
     - `RatecodeID = 99` (unknown rate code, ~2% of remaining observations).
     - `payment_type` was not `1` (credit card) or `2` (cash).
     - `total_amount <= 0` (negative or zero fares).
     - `trip_distance <= 0` (negative or zero trip distances).

2. **Feature Engineering**:
   - Standardized numerical features (`passenger_count`, `trip_distance`) using `StandardScaler`.
   - Removed outliers for numerical features (`passenger_count`, `trip_distance`) based on z-scores (outside ±3), reducing the dataset by ~9%.
   - Capped `total_amount` to a maximum of $500 to handle extreme fare values.
   - Applied one-hot encoding to categorical variables (`VendorID`, `RatecodeID`, `payment_type`) to prepare for modeling.
   - Retained only relevant columns that do not directly contribute to the target variable (`total_amount`).

3. **Exclusions**:
   - Did not implement `pickup_hour` due to its cyclic nature, requiring sine/cosine transformation for compatibility with linear regression.
   - Excluded `trip_duration` as it was redundant with `trip_distance`.

4. **Model Training**:
   - Trained a baseline Linear Regression model to predict `total_amount`.
   - Used an 80/20 train-test split with `random_state=151` for reproducibility.
   - Evaluated performance using Mean Squared Error (MSE). 

### Deliverables
- All code and notebooks have been uploaded to the repository.

### Links
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers)

---

## Fitting Graph

Although our model generalized well, evidenced by a relatively small gap between the training and testing loss, our MSE is still quite high. This points to our linear regression model being on the low end of model complexity and to the left of the ideal range that balances complexity, predictive error, and generalizability.

Because of this, we plan to increase our model complexity, with the goal of decreasing our MSE while minimizing the gap between our training and testing loss. Since our dataset features some categorical attributes, we will move forward with a Decision Tree model that should help capture the non-linear relationships lost in linear regression. To mitigate the overfitting that often plagues such models, we will then further iterate on this by using a Random Forest. 

## Milestone 4: Second Model

### **New Work and Updates**
1. **Feature Selection**:
   - Used `SelectKBest` with `f_regression` to identify the top 5 features.
   - Applied the same transformation to training and testing sets.

2. **Model Implementation**:
   - Built two models to improve performance:  
     - **Decision Tree Regressor**: A simpler non-linear model to assess initial improvements.  
     - **Random Forest Regressor**: An ensemble method to enhance generalization and reduce overfitting.

3. **Decision Tree Regressor**:
   - Trained the Decision Tree model on the full dataset.
   - Manually tuned `max_depth`, `min_samples_split`, and `min_samples_leaf`.
   - **Training MSE**: 8.80  
     **Testing MSE**: 9.68  
   - Observations: We can clearly see that the Decision Tree has non-linear relationships better than Linear Regression but still suffered from overfitting based on gap between training and testing errors.

4. **Random Forest Regressor**:
   - Initially trained on a 10% sample of the dataset for hyperparameter tuning due to computational constraints.
   - Hyperparameters tuned using `GridSearchCV`:
     - Parameters: `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
     - Final hyperparameter values:
       - `n_estimators`: 100
       - `max_depth`: 10
       - `min_samples_split`: 5
       - `min_samples_leaf`: 1
   - Retrained the model using the best-found parameters on the full training dataset.
   - **Training MSE**: 5.17  
     **Testing MSE**: 5.73  
   - Observations: The Random Forest outperformed both Linear Regression and the Decision Tree models, reducing overfitting and generalizing better.

5. **Prediction Analysis (Test Data)**:
   - **Correct Predictions**: 937,500  
   - **False Positives (FP)**: 126,096  
   - **False Negatives (FN)**: 68,766  
   - A prediction was considered "correct" if it fell within ±10% of the actual value.
   - **Visualization**: A bar chart was generated to display the counts of correct predictions, false positives, and false negatives, highlighting the model's strong accuracy. However, the false positives and false negatives indicate potential areas for improvement.

6. **Code and Resources**:
   - [Notebook for Milestone 4](https://github.com/ValerieNguy/CSE151A_Final_Project/blob/main/CSE%20151A%20Milestone%204.ipynb)
   - [Dataset from Kaggle](https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers)


## Conclusions 

### **First Model: Linear Regression**

- **Performance of the Linear Regression Model**:
  - **Training MSE**: 11.18  
  - **Testing MSE**: 11.79  
  - The small difference between the training and testing MSE indicates that the model is generalizable but lacks the complexity to capture non-linear relationships effectively.

- **Insights from Coefficients**:
  - **Trip Distance** (`12.66`): The most significant positive predictor, aligning with expectations as longer trips naturally incur higher fares.
  - **Payment Type** (`-8.83` for cash payments): Suggests that trips paid with cash tend to be cheaper than those paid with credit.
  - **VendorID and RatecodeID**: Capture vendor-specific and fare-code-related differences, though their interpretability requires further analysis.

- **Model Limitations**:
  - The model is overly simplistic and does not account for non-linear relationships, leading to a relatively high MSE.
  - Excluded features like `pickup_hour` might contain predictive information if transformed into cyclic representations.
  - Lack of granularity in fare components, such as surcharges, may hinder the model's performance.

- **Next Model Ideas**:
  1. Apply cyclic transformations for features like `pickup_hour` to evaluate their contribution to fare prediction.
  2. Transition to advanced models like Ridge Regression, Random Forest, or Gradient Boosting for better handling of non-linearity.
  3. Include granular boolean features for surcharges and congestion-related details to enhance the model's robustness.

---

### **Second Model: Decision Tree and Random Forest**

- **Performance of the Decision Tree Regressor**:
  - **Training MSE**: 8.80  
  - **Testing MSE**: 9.68  
  - The Decision Tree model captured non-linear relationships better than Linear Regression but exhibited overfitting, as indicated by the significant gap between training and testing MSE. While it was an improvement over the Linear Regression model, it struggled to generalize well on unseen data.

- **Performance of the Random Forest Model**:
  - **Training MSE**: 5.17  
  - **Testing MSE**: 5.73  
  - The Random Forest model outperformed both Linear Regression and Decision Tree models. The minimal gap between training and testing MSE indicates reduced overfitting and better generalization.
  - **Prediction Analysis (Test Data)**:
    - **Correct Predictions**: 937,500  
    - **False Positives (FP)**: 126,096  
    - **False Negatives (FN)**: 68,766  
    - A prediction was considered "correct" if it fell within ±10% of the actual value. While the Random Forest model demonstrated strong accuracy with a large number of correct predictions, the false positives and false negatives highlight areas for potential improvement.
  - We believe that training the Random Forest model on the full dataset would further enhance prediction performance. Increasing the proportion of the dataset used for training is a key focus for improvement.

- **Comparison and Conclusion**:
  - After trying both the **Decision Tree Regressor** and the **Random Forest Regressor**, we concluded that the **Random Forest Model** is the better choice for this dataset.
  - The Random Forest model handles non-linear relationships more effectively while mitigating overfitting through its ensemble approach.
  - Furthermore, the prediction analysis shows that while the model achieves strong accuracy, addressing false positives and false negatives is necessary to improve reliability further.

- **Next Steps for Improvement**:
  1. Experiment with Gradient Boosting models, such as XGBoost or LightGBM, for potentially better performance.
  2. Expand training to a larger dataset, as the current model was trained on only 10% of the data due to computational constraints.
  3. Implement k-fold cross-validation to further validate the model's robustness.
