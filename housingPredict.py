import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



file_path = "C:\\Users\\aswin\\PycharmProjects\\Project\\housing.csv"
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
# Display the first few rows of the DataFrame to verify the data is loaded correctly
print(df.head())
# Display basic information about the DataFrame
print(df.info())
# Display summary statistics of numerical columns
print(df.describe())

# Handling Missing Values
# Use SimpleImputer to fill missing values
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessing steps
df_preprocessed = pd.DataFrame(preprocessor.fit_transform(df))
# Display the preprocessed DataFrame
print(df_preprocessed.head())

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Visualize the distribution of numerical features
numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

# Pairplot for numerical features
sns.pairplot(df[numerical_features])
#plt.show()

# Correlation heatmap
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
#plt.show()

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['median_house_value'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
#plt.show()

# Boxplot for categorical variable 'ocean_proximity' vs 'median_house_value'
plt.figure(figsize=(12, 8))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=df)
plt.title("Boxplot of Median House Value by Ocean Proximity")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value")
#plt.show()

# Exclude non-numeric columns before calculating correlations
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_with_target = df[numeric_columns].corr()['median_house_value'].abs().sort_values(ascending=False)
selected_features = correlation_with_target[correlation_with_target.index != 'median_house_value'][correlation_with_target > 0.1].index.tolist()

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Update the list of numerical features
numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'median_house_value']

# Update the numeric_transformer in the ColumnTransformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
# Update the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessing steps
df_preprocessed = pd.DataFrame(preprocessor.fit_transform(df), columns=numeric_features + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist())
df_preprocessed['median_house_value'] = df['median_house_value']

#apply transformation to existing feature
df['log_median_income'] = np.log1p(df['median_income'])

#correlation analysis
correlation_with_target = df_preprocessed.corr()['median_house_value'].abs().sort_values(ascending=False)
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()

#SelectKBest (Statistical Tests)
k_best_selector = SelectKBest(score_func=f_regression, k=5)

# Extract features from df_preprocessed
X_selected = k_best_selector.fit_transform(df_preprocessed.drop('median_house_value', axis=1), df_preprocessed['median_house_value'])

# Get the selected feature indices
selected_feature_indices = k_best_selector.get_support(indices=True)

# Get the selected feature names from the preprocessed DataFrame columns
selected_features = df_preprocessed.drop('median_house_value', axis=1).columns[selected_feature_indices].tolist()

# Ensure 'median_house_value' is included in the selected features list
selected_features.append('median_house_value')

# Recursive Feature Elimination (RFE)
estimator = RandomForestRegressor()
rfe_selector = RFE(estimator, n_features_to_select=3, step=2)

print("Starting RFE feature selection...(please allow few minutes for this process)")
# Extract features from df_preprocessed
X_rfe = rfe_selector.fit_transform(df_preprocessed.drop('median_house_value', axis=1), df_preprocessed['median_house_value'])
print("RFE feature selection completed.")
print("Building Model...")

# Get the selected feature indices
selected_feature_indices_rfe = rfe_selector.get_support(indices=True)

# RFE feature selection
selected_features_rfe = df_preprocessed.drop('median_house_value', axis=1).columns[rfe_selector.support_]
# Append the target variable to the selected features
selected_features_rfe = selected_features_rfe.append(pd.Index(['median_house_value']))

# LASSO Regression (L1 Regularization) using features from RFE
lasso = Lasso(alpha=0.1)
lasso.fit(df_preprocessed[selected_features_rfe.drop('median_house_value')], df_preprocessed['median_house_value'])
selected_features_lasso = selected_features_rfe.drop('median_house_value').tolist()

# Split the dataset into features (X) and target variable (y)
X = df_preprocessed.drop('median_house_value', axis=1)
y = df_preprocessed['median_house_value']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
linear_reg_pred = linear_reg_model.predict(X_test)

# Decision Tree
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)
decision_tree_pred = decision_tree_model.predict(X_test)

# Random Forest
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)
random_forest_pred = random_forest_model.predict(X_test)

# Gradient Boosting
gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(X_train, y_train)
gradient_boosting_pred = gradient_boosting_model.predict(X_test)


# Function to evaluate model performance
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"{name} Model:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    print()


# Evaluate Linear Regression
evaluate_model("Linear Regression", y_test, linear_reg_pred)

# Evaluate Decision Tree
evaluate_model("Decision Tree", y_test, decision_tree_pred)

# Evaluate Random Forest
evaluate_model("Random Forest", y_test, random_forest_pred)

# Evaluate Gradient Boosting
evaluate_model("Gradient Boosting", y_test, gradient_boosting_pred)
