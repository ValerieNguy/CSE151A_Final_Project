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


### Notes
- Chose `total_amount` as the target variable.
- Did not implement `pickup_hour` or `trip_duration`:
  - `pickup_hour` requires sine/cosine transformation for linear regression.
  - `trip_duration` was redundant with `trip_distance`.
- MSE is relatively high but generalizes well from training to testing.
- Used an 80/20 train-test split.

### Deliverables
- All code and notebooks have been uploaded to the repository.

### Links
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers)
