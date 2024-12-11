# NYC Taxi Fare Prediction

## Introduction
Our project focuses on predicting taxi fares using the NYC Taxi Fare dataset and we choose this topic to solve real-world relevance and the potential to address a common frustration among riders espcially with unpredictable fare pricing. In taxi services like Uber and Lyft, fares often fluctuate dramatically, especially in high-demand areas like airports. This inconsistency can make it difficult for users to plan trips or trust the pricing system. By building a predictive model, we aim to reduce this unpredictability and provide riders with better insights into fare expectations.

What makes this project exciting is the opportunity to leverage machine learning to solve a tangible problem. Accurate fare predictions are not only beneficial for riders but also for service providers. For users, a predictive model offers transparency and allows them to determine whether a given fare is reasonable. For providers, the model can inform dynamic pricing algorithms, ensuring fares are competitive yet fair. On a broader scale, improving fare predictability fosters trust in transportation platforms, reduces anxiety around trip costs, and creates a more equitable system for both consumers and drivers. Ultimately, this work highlights how data-driven solutions can transform user experiences and operational strategies in transportation services.


## Methods

The implementation of the methods discussed in this section can be viewed in full detail in [this Jupyter Notebook](https://github.com/ValerieNguy/CSE151A_Final_Project/blob/Final-Submission/CSE%20151A%20Milestone%204.ipynb).

### Data Exploration

Along with basic metadata taken from the [Kaggle listing](https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers), we also used `pandas` to load the data as a `Dataframe` to conduct simple exploration of aspects such as value ranges and missing/null values by feature. We then removed outliers from the `trip_distance` and `fare_amount` columns using thresholds of fifty miles and five hundred dollars, respectively. This filtered data was then used to create several data visualizations through `matplotlib.pyplot`, such as the [Fare Amount vs. Trip Distance](#figure-1-fare-amount-vs-trip-distance) and [Fare Amount vs. Pickup Location](#figure-2-fare-amount-vs-pickup-location) figures discussed in further detail later. We also used `seaborn` to create a [pairplot](#figure-3-pair-plot-of-selected-variables) and [heatmap](#figure-4-correlation-heatmap-of-numerical-features) of relevant columns. 

### Data Preprocessing

Working within `pandas`, we dropped null values before converting features `VendorID`, `passenger_count`, `RatecodeID`, `payment_type` from type `float64` to type `int64`, as well as ensuring that the `tpep_pickup_datetime` and `tpep_dropoff_datetime` columns were stored in `datetime` format. Unknown/placeholder values such as negative fare values were also dropped from features `RatecodeID`, `payment_type`, `total_amount`, and `trip_distance`, before pruning redundant or irrelevant columns. We then used `sklearn`'s `StandardScaler` to standardize `passenger_count` and `trip_distance` before dropping any outliers that were more than three standard deviations away from the mean, as well as dropping any total fare amount greater than five hundred dollars from the target variable `total_amount`. Lastly, nominal features `VendorID`, `RatecodeID`, and `payment_type` were one-hot-encoded using the `get_dummies` method from `pandas`.

### Model 1: Linear Regression

Working within `sklearn`, `train_test_split` was used to divide our data into training and testing sets with an 80:20 split. With columns `total_amount`, `VendorID`, `RatecodeID`, `payment_type`, `passenger_count`, and `trip_distance` as input data, a `LinearRegression` object was used to fit a linear regression model to predict `total_amount`. For [result analysis](#model-1-linear-regression-1), training and testing MSE were calculated using the native `sklearn` method. Defining a correct prediction as any prediction that fell within 10% of its actual value, we again used `pyplot` to visualize the model's correct predictions in comparison to its false positives and false negatives, as well as a fitting graph visualizing the distribution of predicted values compared to actual values.

### Model 2: Decision Tree

Once again working within `sklearn`, the data was split in the same way but passed into a `DecisionTreeRegressor` object in order to fit a decision tree to predict `total_amount`. Additionally, `GridSearchCV` was used to tune hyperparameters `max_depth`, `min_samples_split`, and `min_samples_leaf` through five-fold cross-validation with negative MSE used as the evaluation metric. The best hyperparameters were determined to be `max_depth = 10`, `min_samples_split = 4`, and `min_samples_leaf = 10`. [Result analysis](#model-2-decision-tree-1) of MSE, prediction comparison, and fitting graph was performed in the same way as described in Model 1.

### Model 3: Random Forest

Expanding upon the decision tree model, we conducted hyperparameter tuning for a `RandomForestRegressor` object with the additional hyperparameter `n_estimators`. We also used `SelectKBest` to implement feature selection, selecting the top five most relevant features with the ANOVA F-statistic as the evaluation metric. The tuning process was slightly scaled down to use three-fold cross validation on 30% of the training data, randomly subsampled using the `sample` method from `pandas`. The best hyperparameters were determined to be `max_depth = 10`, `min_samples_split = 5`, `min_samples_leaf = 2`, and `n_estimators = 100`. [Result analysis](#model-3-random-forest-1) of MSE, prediction comparison, and fitting graph was performed in the same way as described in Model 1.

## Results

### Model 1: Linear Regression
The Linear Regression model achieved an accuracy of 11.17 mean squared error on the training set and 11.79 mean squared error on the test set. Using our method for determining if a prediction was correct or not, on the test data, we got 600294 correct predictions, 318063 false positives, and 216183 false negatives. For visualization, a bar chat of the prediction counts is as follows:
![image](https://github.com/user-attachments/assets/1da7c53c-82a1-4b08-97e4-1941d0f94859)

### Model 2: Decision Tree
We decided to pivot away from the Linear Regression model and into a Decision Tree model. This model increased accuracy with a mean squared error of 8.80 on the training set and 9.68 on the test set. Decision Tree had 659422 correct predictions, 274316 false positives, and 200802 false negatives on the test data. These counts are visualized on the bar chart below:
![image](https://github.com/user-attachments/assets/800b9788-5edc-4132-8127-c00c8b9b281e)

### Model 3: Random Forest
Lastly, the Random Forest model had a mean squared error of 5.20 on the training set and 5.73 on the test set. On the test data, this model had 937495 correct predictions, 126133 false positives, and 68734 false negatives. A bar chart of these counts is displayed below for visualization:
![image](https://github.com/user-attachments/assets/650db28e-9988-4c61-b289-385e4a81033d)

### Summary Table of Model Performance

| Model              | Training MSE | Testing MSE  |
|--------------------|--------------|--------------|
| **Linear Regression**  | 11.18        |  11.79     |
| **Decision Tree**      | 8.80         | 9.68     |
| **Random Forest**      | 5.20         | 5.73     |

## Discussion

- interpret model results
  - good/bad? believable?
  - what can be extrapolated from results?
- limitations
- fitting graph stuff


### Data Exploration
The NYC Taxi Fare dataset comprises approximately 6.4 million observations and 18 features, providing comprehensive information on taxi rides in New York City. Key features include VendorID, which identifies the taxi vendor or service provider; tpep_pickup_datetime and tpep_dropoff_datetime, which record the trip start and end timestamps; passenger_count, indicating the number of passengers; trip_distance, which measures the distance traveled; fare_amount, representing the base fare amount; total_amount, which includes surcharges; and payment_type, indicating the payment method (e.g., cash or card). A detailed preliminary analysis revealed strong correlations among fare-related components such as fare_amount, total_amount, and tip_amount, suggesting consistency in fare calculations. The dataset contains approximately 327,205 missing values across various columns, including VendorID and passenger_count. Significant outliers were identified in key features, such as an exceptionally high trip distance of 210,240.07 miles and a maximum total fare amount of $4,265, both of which highlight the need for careful preprocessing before model training.

### Figure 1: Fare Amount vs. Trip Distance 
![Screenshot 2024-12-08 at 10 26 43 AM](https://github.com/user-attachments/assets/5c477573-2553-4115-a229-3b6eb1034d55)

This scatter plot illustrates the relationship between trip distance (in miles) and fare amount (in dollars) after filtering outliers. The data exhibits a generally positive trend, where longer trip distances correspond to higher fares. However, there is considerable variation, particularly for shorter trips, which could reflect differences in location, surcharges, or additional charges. Outliers in both fare amount and trip distance have been removed to focus on the core patterns.

### Figure 2: Fare Amount vs. Pickup Location 
![Screenshot 2024-12-08 at 10 26 55 AM](https://github.com/user-attachments/assets/1ca00f54-6123-4275-afc7-1c9f9c72a6ce)

This scatter plot shows fare amounts across different pickup location IDs. The data highlights clustering trends, where certain locations (e.g., high-demand areas like airports) correspond to higher average fares. The wide range of fare amounts at specific pickup locations suggests variability influenced by trip distance, time of day, or other factors.

### Figure 3: Pair Plot of Selected Variables 
![Screenshot 2024-12-08 at 11 04 46 AM](https://github.com/user-attachments/assets/92d30e49-debc-4d47-bb3f-bf773adda1ec)

This pair plot visualizes the relationships and distributions of selected variables, including trip_distance, fare_amount, PULocationID, DOLocationID, congestion_surcharge, and improvement_surcharge. It highlights patterns such as the positive relationship between trip_distance and fare_amount, as well as clustering in categorical features like PULocationID and DOLocationID. The diagonal plots show the distribution of each variable, while the off-diagonal scatter plots illustrate pairwise interactions, helping identify potential correlations and feature dependencies.

### Figure 4: Correlation Heatmap of Numerical Features
![Screenshot 2024-12-08 at 10 27 33 AM](https://github.com/user-attachments/assets/7e3e09cd-7dac-4b18-8919-55c0450c904f)

The heatmap provides a visual summary of the correlations among numerical features in the dataset. Notable relationships include a strong positive correlation between fare_amount and total_amount, indicating their dependency. Moderate correlations are also observed between trip_distance and fare-related variables, reflecting the role of distance in fare calculations. Other features, such as payment_type and VendorID, exhibit weaker or negligible correlations with fare-related attributes. This visualization helps identify the most influential features for modeling.


### Data Preprocessing

To prepare the dataset for modeling, several preprocessing steps were performed to ensure data quality and suitability for machine learning. First, approximately 300,000 rows (~1% of the dataset) with missing or null values were identified and dropped to maintain data integrity and avoid introducing bias. Next, invalid entries were removed, including rows with RatecodeID = 99 (unknown rate codes, ~2% of observations), unsupported payment_type values not in {1, 2}, and rows with negative or zero values in total_amount or trip_distance. To handle outliers, extreme values were capped with thresholds, limiting trip_distance to 50 miles and total_amount to $500. Additional outliers in trip_distance and passenger_count were filtered using z-score thresholds (±3). For consistency, numerical features such as trip_distance, fare_amount, and total_amount were standardized using StandardScaler. Only relevant columns, including total_amount, VendorID, RatecodeID, payment_type, passenger_count, and trip_distance, were retained for model training. Finally, categorical features like VendorID, RatecodeID, and payment_type were converted into machine-readable formats using one-hot encoding, completing the preprocessing pipeline.


### Model 1: Linear Regression

The first model implemented was a Linear Regression model, serving as a baseline to predict total_amount based on the selected features. The purpose of this model was to establish a reference point for evaluating the performance of more complex models. The dataset was split into training and testing subsets using an 80:20 ratio, with a random_state of 151 to ensure reproducibility. The model was trained with default hyperparameters and evaluated using Mean Squared Error (MSE) on both training and testing sets. This baseline provides a straightforward yet effective comparison for subsequent models.


### Model 2: Decision Tree Regressor

The Decision Tree Regressor was introduced to address non-linear relationships in the dataset that the Linear Regression model could not capture. Its purpose was to improve prediction accuracy by modeling complex interactions between features. The dataset was split into training and testing sets using the same 80:20 ratio and random_state=151 as Model 1 for consistency. Hyperparameter tuning was performed to optimize the model, focusing on parameters such as max_depth to control tree depth and prevent overfitting, min_samples_split to determine the minimum number of samples required to split a node, and min_samples_leaf to specify the minimum number of samples required in a leaf node. The model’s performance was evaluated using Mean Squared Error (MSE) on both training and testing datasets, ensuring a robust assessment of its predictive capabilities.


### Model 3: Random Forest Regressor

The Random Forest Regressor was introduced to improve generalization and reduce overfitting compared to the Decision Tree model. By leveraging an ensemble approach, this model aimed to better capture non-linear relationships and variability within the dataset. The dataset was split into training and testing subsets using the same 80:20 ratio and random_state=151 as the previous models. Hyperparameter tuning was conducted using GridSearchCV on a 10% random sample of the training data, optimizing parameters such as n_estimators (number of trees in the forest), max_depth (maximum depth of each tree), min_samples_split (minimum samples required to split a node), and min_samples_leaf (minimum samples required in a leaf node). The best hyperparameters were identified using 3-fold cross-validation. The final model was then trained on the full training dataset using these optimized parameters. Performance was evaluated on both training and testing datasets using Mean Squared Error (MSE), providing insights into the model’s ability to generalize and predict effectively.


## Conclusion

Although we are quite satisfied by the results that we were able to achieve, we do wish that we had time to experiment with implementing other models, such as neural networks. While the size and scope of this dataset is not suitable for a model utilizing a transformer architecture, it would be interesting to see if a transformer-based model would uncover more subtle trends within the data, perhaps with more nebulous features to take advantage of the model's enhanced interpretive capabilities. In a similar vein, testing model implementation at scale using a larger dataset and something like SDSC is also an intriguing prospect that we unfortunately opted against pursuing. 

Regardless, this endeavor served as a solid groundwork for exploring foundational machine learning models using commonly used packages, with a particular emphasis on `sklearn`. We also gained insight into the factors influencing fare pricing, which will be increasingly more valuable as rideshare services continue to become a mainstay of urban transportation. Moving forward, we hope to be able to leverage these machine learning concepts in future applications as the field develops further.

## Statement of Collaboration

**Heesoon Kang (Title):** Performed data exploration / preprocessing, helped with write up for all milestones, and contributed to models / data exploration writeup of final report

**Valerie Nguyen (Title):** writeup for data exploration/preprocessing, implemented Model 2, contributions to results section of final report.

**Nicholas Jumaoas (Title):** performed data preprocessing, implemented Model 1, organized & edited final report along with contributions to methods, discussion, and conclusion sections.
