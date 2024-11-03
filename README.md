# CSE151A Final Project

## Data Preprocessing

Our dataset, sourced from Kaggle, has already been done a initial cleaning by the author, which includes standard formatting and handling missing values. However, to fully optimize the data and improve the accuracy of our predictive model, additional preprocessing steps are necessary. These steps will address any remaining inconsistencies, such as missing data and outliers (e.g., negative values in `fare_amount` and `trip_distance`), which may still be present.

### Planned Preprocessing Steps

1. **Handling Missing or Null Values**:  
   We will examine each column for any remaining missing values and determine an appropriate strategy to address them. Depending on the feature, we may choose to drop rows with missing values.

2. **Outlier Removal**:  
   Outliers in key features such as `fare_amount` and `trip_distance` can negatively impact the model's performance due to some negative datas that does not make sense. We will also filter out extreme values by setting reasonable thresholds, such as capping `trip_distance` at 50 miles and limiting `fare_amount` to a common maximum.

3. **Normalization and Scaling**:  
   To ensure consistent weighting across features during model training, we will normalize or scale numerical columns such as `trip_distance`, `fare_amount`, and `total_amount`. Depending on the model requirements, we may use standardization or min-max scaling.
