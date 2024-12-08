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

#### Figures
*(Placeholder: Correlation heatmap between features highlighting key relationships.)*

*(Code snippet for initial data exploration)*  
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("nyc_taxi_data.csv")

# Display dataset summary
print(df.info())

# Display feature correlations
corr = df.corr()
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Feature Correlation Heatmap')
plt.show()
