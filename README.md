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

Our dataset, sourced from Kaggle, has already had an initial cleaning by the author, which includes standard formatting and handling missing values. However, to fully optimize the data and improve the accuracy of our predictive model, additional preprocessing steps are necessary. These steps will address any remaining inconsistencies, such as null data and outliers (e.g., negative values in `fare_amount` and `trip_distance`), which may still be present.

### Planned Preprocessing Steps

1. **Handling Missing or Null Values**:  
   We will examine each column for any remaining miss or null values and determine an appropriate strategy to address them. Depending on the feature, we may choose to drop rows with null values.

2. **Outlier Removal**:  
   Outliers in key features such as `fare_amount` and `trip_distance` can negatively impact the model's performance due to some negative datas that does not make sense. We will also filter out extreme values by setting reasonable thresholds, such as capping `trip_distance` at 50 miles and limiting `fare_amount` to a common maximum.

3. **Normalization and Scaling**:  
   To ensure consistent weighting across features during model training, we will normalize or scale numerical columns such as `trip_distance`, `fare_amount`, and `total_amount`. Depending on the model requirements, we may use standardization or min-max scaling.
