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
After plotting the actual vs. predicted values for both our training and test sets, we notice that there is a large cluster of points in the bottom left corner of the plot (which indicates the model is frequently predicting low values). Additionally, as we move towards higher values for actual values, the points are sparser and still low for the predicted values. This suggests that the model tends to underpredict in this range for both are training and test set. Based on this, we can conclude that this model is too simple for what we are attempting to predict.  

Given that our model is underfitting with the linear regression model, we may want to explore more complex models that might do a better job of accounting for non-linear relationships. We may consider switching to models such as polynomial regression or decision trees. These models have more flexibility which could improve performance for both lower and higher range values.

## Conclusions Based on First Model

- **Performance of the Linear Regression Model**:
  - **Training MSE**: 11.18  
  - **Testing MSE**: 11.79  
  - The MSE for the training data is slightly lower than for the testing data. Because the difference between the training and testing MSE isn't very large, this indicates that the model is generalizable. 

- **Insights from Coefficients**:
  - **Trip Distance** (`12.66`): Has the largest positive impact, which is expected as longer trips lead to higher fares.
  - **Payment Type** (`-8.83` for cash payments): Indicates that trips paid with cash are generally less expensive than those paid with credit.
  - **VendorID and RatecodeID**: These categorical variables capture vendor-specific and fare-code-related differences but require more exploration for interpretability.

- **Model Limitations**:
  - The relatively high MSE suggests that the model does not fully capture non-linear relationships in the dataset.
  - Some features (e.g., `pickup_hour`) were excluded due to their cyclic nature but might add predictive power with proper transformations.

- **Next Model Ideas**:
  1. Include cyclic transformations for features like `pickup_hour` to see if this feature is important in predicting total fare.
  2. Explore advanced models such as Ridge Regression, Random Forest, or Gradient Boosting to capture non-linear relationships.
  3. Add boolean indicators for surcharges to retain more granular details in the model.
