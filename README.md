## Introduction

Our project focuses on predicting taxi fares using the NYC Taxi Fare dataset and we choose this topic to solve real-world relevance and the potential to address a common frustration among riders espcially with unpredictable fare pricing. In Texi services like Uber and Lyft, fares often fluctuate dramatically, especially in high-demand areas like airports. This inconsistency can make it difficult for users to plan trips or trust the pricing system. By building a predictive model, we aim to reduce this unpredictability and provide riders with better insights into fare expectations.

What makes this project exciting is the opportunity to leverage machine learning to solve a tangible problem. Accurate fare predictions are not only beneficial for riders but also for service providers. For users, a predictive model offers transparency and allows them to determine whether a given fare is reasonable. For providers, the model can inform dynamic pricing algorithms, ensuring fares are competitive yet fair. On a broader scale, improving fare predictability fosters trust in transportation platforms, reduces anxiety around trip costs, and creates a more equitable system for both consumers and drivers. Ultimately, this work highlights how data-driven solutions can transform user experiences and operational strategies in transportation services.

## Methods

### Data Exploration

The NYC Taxi Fare dataset contains approximately 6.4 million observations and 18 features. The key features include:

- **VendorID**: Identifier for the taxi vendor or service provider.  
- **tpep_pickup_datetime** and **tpep_dropoff_datetime**: Timestamps for trip start and end.  
- **passenger_count**: Number of passengers in the taxi.  
- **trip_distance**: Total distance of the trip.  
- **fare_amount**: Base fare amount.  
- **total_amount**: Total fare amount, including all surcharges.  
- **payment_type**: Method of payment (e.g., cash, card).  

#### Findings
- Strong correlations were observed between fare-related features such as `fare_amount`, `total_amount`, and `tip_amount`.
- Negative correlations were noted between certain vendor identifiers (`VendorID`) and extra charges, as well as between `payment_type` and `tip_amount`.
- Potential outliers were identified in features such as `trip_distance`, `fare_amount`, `extra`, and `total_amount`.


### Data Preprocessing

To prepare the dataset for modeling, we performed the following preprocessing steps:

#### 1. Handling Missing Values
- Identified ~300,000 rows (~1% of the dataset) with missing or null values.
- Dropped rows with null values to avoid introducing bias.

#### 2. Removing Invalid Entries
- Removed rows with nonsensical or invalid data:
  - **RatecodeID = 99**: Unknown rate codes (~2% of observations).
  - **payment_type** not in `{1, 2}`: Excluded unsupported payment types.
  - **total_amount ≤ 0**: Negative or zero fare amounts.
  - **trip_distance ≤ 0**: Negative or zero trip distances.

#### 3. Outlier Removal
- Filtered extreme values:
  - **trip_distance** capped at 50 miles.
  - **total_amount** capped at $500.
- Applied z-score filtering (±3) to remove additional outliers from `trip_distance` and `passenger_count`.

#### 4. Normalization and Scaling
- Standardized numerical features (`trip_distance`, `fare_amount`, `total_amount`) using StandardScaler to ensure consistent weighting during model training.

#### 5. Categorical Encoding
- Applied one-hot encoding to categorical features (`VendorID`, `RatecodeID`, `payment_type`) to prepare the dataset for machine learning models.



### Model 1: Linear Regression

#### Overview
- Baseline model used to predict `total_amount` from selected features.
- Purpose: Establish a reference for comparison with more complex models.

#### Implementation Details
- Split the dataset into training and testing sets using an 80:20 ratio (`random_state=151` for reproducibility).
- Trained a Linear Regression model with default hyperparameters.
- Evaluated performance using Mean Squared Error (MSE) on both training and testing sets.



### Model 2: Decision Tree Regressor

#### Overview
- Introduced to capture non-linear relationships in the dataset that Linear Regression could not address.
- Purpose: Improve prediction accuracy by modeling complex interactions between features.

#### Implementation Details
- Dataset: Same training and testing sets as used for Model 1 (80:20 split, `random_state=151`).
- Hyperparameter tuning:
  - `max_depth`: Limits the depth of the tree to prevent overfitting.
  - `min_samples_split`: Minimum number of samples required to split a node.
  - `min_samples_leaf`: Minimum number of samples required to be in a leaf node.
- Evaluated using Mean Squared Error (MSE) on training and testing sets.



### Model 3: Random Forest Regressor

#### Overview
- Introduced to reduce overfitting and improve generalization compared to the Decision Tree model.
- Purpose: Leverage an ensemble approach to better capture non-linear relationships and variability in the dataset.

#### Implementation Details
- Dataset: Same training and testing sets as used for previous models (80:20 split, `random_state=151`).
- Hyperparameter tuning:
  - Used `GridSearchCV` to optimize:
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of each tree.
    - `min_samples_split`: Minimum number of samples required to split a node.
    - `min_samples_leaf`: Minimum number of samples required in a leaf node.
- Trained the final model on the full dataset using the best parameters from GridSearchCV.
- Evaluated performance using Mean Squared Error (MSE) on both training and testing sets.
