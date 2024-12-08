# NYC Taxi Fare Prediction

## Introduction
Our project focuses on predicting taxi fares using the NYC Taxi Fare dataset and we choose this topic to solve real-world relevance and the potential to address a common frustration among riders espcially with unpredictable fare pricing. In Texi services like Uber and Lyft, fares often fluctuate dramatically, especially in high-demand areas like airports. This inconsistency can make it difficult for users to plan trips or trust the pricing system. By building a predictive model, we aim to reduce this unpredictability and provide riders with better insights into fare expectations.

What makes this project exciting is the opportunity to leverage machine learning to solve a tangible problem. Accurate fare predictions are not only beneficial for riders but also for service providers. For users, a predictive model offers transparency and allows them to determine whether a given fare is reasonable. For providers, the model can inform dynamic pricing algorithms, ensuring fares are competitive yet fair. On a broader scale, improving fare predictability fosters trust in transportation platforms, reduces anxiety around trip costs, and creates a more equitable system for both consumers and drivers. Ultimately, this work highlights how data-driven solutions can transform user experiences and operational strategies in transportation services.


## Methods

### Data Exploration
The NYC Taxi Fare dataset comprises approximately 6.4 million observations and 18 features, providing comprehensive information on taxi rides in New York City. Key features include VendorID, which identifies the taxi vendor or service provider; tpep_pickup_datetime and tpep_dropoff_datetime, which record the trip start and end timestamps; passenger_count, indicating the number of passengers; trip_distance, which measures the distance traveled; fare_amount, representing the base fare amount; total_amount, which includes surcharges; and payment_type, indicating the payment method (e.g., cash or card). A detailed preliminary analysis revealed strong correlations among fare-related components such as fare_amount, total_amount, and tip_amount, suggesting consistency in fare calculations. The dataset contains approximately 327,205 missing values across various columns, including VendorID and passenger_count. Significant outliers were identified in key features, such as an exceptionally high trip distance of 210,240.07 miles and a maximum total fare amount of $4,265, both of which highlight the need for careful preprocessing before model training.



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
- Utilized z-score filtering (±3) to remove additional outliers from trip_distance and passenger_count.

#### 4. Normalization and Scaling
- Standardized numerical features (`trip_distance`, `fare_amount`, `total_amount`) using StandardScaler to ensure consistent weighting during model training.

#### 5. Feature Selection
- Retained only relevant columns: `total_amount`, `VendorID`, `RatecodeID`, `payment_type`, `passenger_count`, `trip_distance`.

#### 6. Categorical Encoding
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
  - Used `GridSearchCV` on a 10% random sample of the training data to optimize:
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of each tree.
    - `min_samples_split`: Minimum number of samples required to split a node.
    - `min_samples_leaf`: Minimum number of samples required in a leaf node.
  - Identified the best hyperparameters using 3-fold cross-validation.
- Trained the final model on the full dataset using the best parameters from GridSearchCV.
- Evaluated performance using Mean Squared Error (MSE) on both training and testing sets.


## Results

### Summary Table of Model Performance

| Model              | Training MSE | Testing MSE  |
|--------------------|--------------|--------------|
| **Linear Regression**  | 11.18        |  11.79     |
| **Decision Tree**      | 8.80         | 9.68     |
| **Random Forest**      | 5.20         | 5.73     |

### Model 1: Linear Regression
The Linear Regression model achieved an accuracy of 11.17 mean squared error on the training set and 11.79 mean squared error on the test set. Using our method for determining if a prediction was correct or not, on the test data, we got 600294 correct predictions, 318063 false positives, and 216183 false negatives. For visualization, a bar chat of the prediction counts is as follows:
![image](https://github.com/user-attachments/assets/1da7c53c-82a1-4b08-97e4-1941d0f94859)

### Model 2: Decision Tree
We decided to pivot away from the Linear Regression model and into a Decision Tree model. This model increased accuracy with a mean squared error of 8.80 on the training set and 9.68 on the test set. Decision Tree had 659422 correct predictions, 274316 false positives, and 200802 false negatives on the test data. These counts are visualized on the bar chart below:
![image](https://github.com/user-attachments/assets/800b9788-5edc-4132-8127-c00c8b9b281e)

### Model 3: Random Forest
Lastly, the Random Forest model had a mean squared error of 5.20 on the training set and 5.73 on the test set. On the test data, this model had 937495 correct predictions, 126133 false positives, and 68734 false negatives. A bar chart of these counts is displayed below for visualization:
![image](https://github.com/user-attachments/assets/650db28e-9988-4c61-b289-385e4a81033d)


## Discussion

- interpret model results
  - good/bad? believable?
  - what can be extrapolated from results?
- limitations
- fitting graph stuff


## Conclusion



## Statement of Collaboration


