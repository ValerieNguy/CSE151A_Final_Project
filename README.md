# NYC Taxi Fare Prediction

## Introduction
Our project focuses on predicting taxi fares using the NYC Taxi Fare dataset and we choose this topic to solve real-world relevance and the potential to address a common frustration among riders espcially with unpredictable fare pricing. In Texi services like Uber and Lyft, fares often fluctuate dramatically, especially in high-demand areas like airports. This inconsistency can make it difficult for users to plan trips or trust the pricing system. By building a predictive model, we aim to reduce this unpredictability and provide riders with better insights into fare expectations.

What makes this project exciting is the opportunity to leverage machine learning to solve a tangible problem. Accurate fare predictions are not only beneficial for riders but also for service providers. For users, a predictive model offers transparency and allows them to determine whether a given fare is reasonable. For providers, the model can inform dynamic pricing algorithms, ensuring fares are competitive yet fair. On a broader scale, improving fare predictability fosters trust in transportation platforms, reduces anxiety around trip costs, and creates a more equitable system for both consumers and drivers. Ultimately, this work highlights how data-driven solutions can transform user experiences and operational strategies in transportation services.


## Methods

### Data Exploration
The NYC Taxi Fare dataset comprises approximately 6.4 million observations and 18 features, providing comprehensive information on taxi rides in New York City. Key features include VendorID, which identifies the taxi vendor or service provider; tpep_pickup_datetime and tpep_dropoff_datetime, which record the trip start and end timestamps; passenger_count, indicating the number of passengers; trip_distance, which measures the distance traveled; fare_amount, representing the base fare amount; total_amount, which includes surcharges; and payment_type, indicating the payment method (e.g., cash or card). A detailed preliminary analysis revealed strong correlations among fare-related components such as fare_amount, total_amount, and tip_amount, suggesting consistency in fare calculations. The dataset contains approximately 327,205 missing values across various columns, including VendorID and passenger_count. Significant outliers were identified in key features, such as an exceptionally high trip distance of 210,240.07 miles and a maximum total fare amount of $4,265, both of which highlight the need for careful preprocessing before model training.

### Figure 1: Fare Amount vs. Trip Distance (Filtered)
![Screenshot 2024-12-08 at 10 26 43 AM](https://github.com/user-attachments/assets/5c477573-2553-4115-a229-3b6eb1034d55)




### Data Preprocessing

To prepare the dataset for modeling, several preprocessing steps were performed to ensure data quality and suitability for machine learning. First, approximately 300,000 rows (~1% of the dataset) with missing or null values were identified and dropped to maintain data integrity and avoid introducing bias. Next, invalid entries were removed, including rows with RatecodeID = 99 (unknown rate codes, ~2% of observations), unsupported payment_type values not in {1, 2}, and rows with negative or zero values in total_amount or trip_distance. To handle outliers, extreme values were capped with thresholds, limiting trip_distance to 50 miles and total_amount to $500. Additional outliers in trip_distance and passenger_count were filtered using z-score thresholds (±3). For consistency, numerical features such as trip_distance, fare_amount, and total_amount were standardized using StandardScaler. Only relevant columns, including total_amount, VendorID, RatecodeID, payment_type, passenger_count, and trip_distance, were retained for model training. Finally, categorical features like VendorID, RatecodeID, and payment_type were converted into machine-readable formats using one-hot encoding, completing the preprocessing pipeline.






### Model 1: Linear Regression

The first model implemented was a Linear Regression model, serving as a baseline to predict total_amount based on the selected features. The purpose of this model was to establish a reference point for evaluating the performance of more complex models. The dataset was split into training and testing subsets using an 80:20 ratio, with a random_state of 151 to ensure reproducibility. The model was trained with default hyperparameters and evaluated using Mean Squared Error (MSE) on both training and testing sets. This baseline provides a straightforward yet effective comparison for subsequent models.


### Model 2: Decision Tree Regressor

The Decision Tree Regressor was introduced to address non-linear relationships in the dataset that the Linear Regression model could not capture. Its purpose was to improve prediction accuracy by modeling complex interactions between features. The dataset was split into training and testing sets using the same 80:20 ratio and random_state=151 as Model 1 for consistency. Hyperparameter tuning was performed to optimize the model, focusing on parameters such as max_depth to control tree depth and prevent overfitting, min_samples_split to determine the minimum number of samples required to split a node, and min_samples_leaf to specify the minimum number of samples required in a leaf node. The model’s performance was evaluated using Mean Squared Error (MSE) on both training and testing datasets, ensuring a robust assessment of its predictive capabilities.


### Model 3: Random Forest Regressor

The Random Forest Regressor was introduced to improve generalization and reduce overfitting compared to the Decision Tree model. By leveraging an ensemble approach, this model aimed to better capture non-linear relationships and variability within the dataset. The dataset was split into training and testing subsets using the same 80:20 ratio and random_state=151 as the previous models. Hyperparameter tuning was conducted using GridSearchCV on a 10% random sample of the training data, optimizing parameters such as n_estimators (number of trees in the forest), max_depth (maximum depth of each tree), min_samples_split (minimum samples required to split a node), and min_samples_leaf (minimum samples required in a leaf node). The best hyperparameters were identified using 3-fold cross-validation. The final model was then trained on the full training dataset using these optimized parameters. Performance was evaluated on both training and testing datasets using Mean Squared Error (MSE), providing insights into the model’s ability to generalize and predict effectively.


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


