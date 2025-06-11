import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def regression_feature_engineering(filename):
    """
    Preprocesses the property dataset by performing the following steps:
        - Reads CSV file and drops unnecessary columns
        - Caps bedroom, bathroom, and parking counts to handle outliers
        - Derives new temporal features (e.g., year sold, year purchased)
        - Computes average time to CBD from multiple transport methods
        - Encodes region price variability and flags high/low price regions
        - Consolidates rare property types into 'Other'
        - Encodes property type variability into ordinal values
        - Computes suburb median price based on property type
        - Encodes suburb price tier based on quantiles
        - Creates additional binary flags (e.g., sold in 2021, CBD proximity)
        - Encodes presence of train station
        - Drops redundant or unused columns

    Parameters:
    filename (str): name of the input CSV file

    Returns:
    pandas.DataFrame: Preprocessed dataset ready for regression modeling
    """

    ## Function to determine suburb median price based on property type
    def get_suburb_median_price(row):
        if row["type"] == "House":
            return row["suburb_median_house_price"]
        elif "Apartment" in row["type"]:
            return row["suburb_median_apartment_price"]
        else:
            return (
                row["suburb_median_house_price"] + row["suburb_median_apartment_price"]
            ) / 2

    ## Read the dataset and drop unnecessary columns
    dataset = pd.read_csv(filename, index_col="id")
    dataset = dataset.drop(
        columns=[
            "traffic",
            "public_transport",
            "affordability_rental",
            "affordability_buying",
            "nature",
            "noise",
            "things_to_see_do",
            "family_friendliness",
            "pet_friendliness",
            "safety",
            "overall_rating",
            "highlights_attractions",
            "ideal_for",
            "ethnic_breakdown",
            "suburb_lat",
            "suburb_lng",
            "postcode",
            "suburb_population",
            "suburb_sqkm",
            "suburb_elevation",
            "suburbpopulation",
            "public_housing_pct",
            "median_house_rent_per_week",
            "median_apartment_rent_per_week",
            "suburb",
        ]
    )

    ## Cap the number of bedrooms, bathrooms, and parking spaces to handle outliers
    dataset["num_bed"] = dataset["num_bed"].apply(lambda x: min(x, 8))
    dataset["num_bath"] = dataset["num_bath"].apply(lambda x: min(x, 6))
    dataset["num_parking"] = dataset["num_parking"].apply(lambda x: min(x, 6))

    # Get new temporal features
    dataset["date_sold"] = pd.to_datetime(dataset["date_sold"])
    dataset["year_sold"] = dataset["date_sold"].dt.year
    dataset["year_purchased"] = np.floor(
        dataset["year_sold"] - dataset["avg_years_held"]
    ).astype(int)
    dataset = dataset.drop(columns=["date_sold"])

    # Calculate average time to CBD from multiple transport methods
    dataset["time_to_cbd_avg"] = dataset[
        [
            "time_to_cbd_public_transport_town_hall_st",
            "time_to_cbd_driving_town_hall_st",
        ]
    ].mean(axis=1)
    dataset = dataset.drop(
        [
            "time_to_cbd_public_transport_town_hall_st",
            "time_to_cbd_driving_town_hall_st",
        ],
        axis=1,
    )

    # Create dictionary to Map region according to price variability
    variability_map_region = {
        "Eastern Suburbs": "Very High",
        "Lower North Shore": "Very High",
        "Upper North Shore": "High",
        "North Shore": "High",
        "Inner East": "High",
        "Northern Beaches": "High",
        "Inner West": "Medium",
        "Sydney City": "Medium",
        "Northern Suburbs": "Medium",
        "Southern Suburbs": "Medium",
        "Sutherland Shire": "Low",
        "Inner South": "Low",
        "Hills Shire": "Low",
        "Western Suburbs": "Very Low",
        "South West": "Very Low",
    }

    dataset["region_price_variability"] = dataset["region"].map(variability_map_region)

    ## Encode region price variability into ordinal values
    variability_order_region = {
        "Very Low": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4,
    }
    dataset["region_price_variability_encoded"] = dataset[
        "region_price_variability"
    ].map(variability_order_region)

    ## Create binary flags for high and low price regions
    dataset["is_high_price_region"] = (
        dataset["region"].isin(["Eastern Suburbs", "Lower North Shore"]).astype(int)
    )
    dataset["is_low_price_region"] = (
        dataset["region"].isin(["Western Suburbs", "South West"]).astype(int)
    )

    dataset = dataset.drop(columns=["region", "region_price_variability"])

    ## Consolidate rare property types into 'Other'
    type_counts = dataset["type"].value_counts()

    types_to_replace = type_counts[type_counts < 10].index

    dataset["type"] = dataset["type"].replace(types_to_replace, "Other")

    # Create dictionary to Map property type according to price variability
    property_type_map = {
        "Block of Units": "High",
        "Terrace": "High",
        "House": "High",
        "Semi-Detached": "Medium",
        "Duplex": "Medium",
        "Apartment / Unit / Flat": "Medium",
        "Other": "Medium",
        "Townhouse": "Low",
        "Vacant land": "Low",
        "Villa": "Low",
    }

    dataset["property_type_variability"] = dataset["type"].map(property_type_map)

    ## Encode property type variability into ordinal values
    variability_order_property = {"Low": 0, "Medium": 1, "High": 2}
    dataset["property_type_variability_encoded"] = dataset[
        "property_type_variability"
    ].map(variability_order_property)

    dataset = dataset.drop(columns=["property_type_variability"])

    ## Create suburb median price columns based on property type
    dataset["suburb_median_price_by_type"] = dataset.apply(
        get_suburb_median_price, axis=1
    )

    dataset = dataset.drop(
        columns=["type", "suburb_median_house_price", "suburb_median_apartment_price"]
    )

    # Create dictionary to Map suburb price tier according to quantiles
    dataset["suburb_price_tier"] = pd.qcut(
        dataset["suburb_median_price_by_type"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
    )

    ## Encode suburb price tier into ordinal values
    tier_mapping = {"Low": 0, "Mid-Low": 1, "Mid-High": 2, "High": 3}

    dataset["suburb_price_tier_encoded"] = (
        dataset["suburb_price_tier"].map(tier_mapping).astype(int)
    )

    dataset = dataset.drop(columns=["suburb_price_tier"])

    ## Create binary flags for sold in 2021
    dataset["sold_in_2021"] = (dataset["year_sold"] == 2021).astype(int)

    # Create new features for CBD proximity
    dataset["inv_cbd_dist"] = 1 / (dataset["km_from_cbd"] + 1)
    dataset["is_cbd_near"] = (dataset["km_from_cbd"] < 20).astype(int)

    dataset["has_nearest_train_station"] = (
        dataset["nearest_train_station"] != "0"
    ).astype(int)
    dataset = dataset.drop(["nearest_train_station"], axis=1)

    # Return feature engineered dataset
    return dataset


def train_regression(train_df):
    """
    Trains a regression model using the HistGradientBoostingRegressor with MAE loss.

    Parameters:
    train_df (pandas.DataFrame): The training dataset including both features and the target variable ('price').

    Returns:
    HistGradientBoostingRegressor: Trained regression model
    """

    # Separate the target variable from the features
    y_train = train_df["price"]
    X_train = train_df.drop("price", axis=1)

    # Initialize the HistGradientBoostingRegressor with specified hyperparameters
    hist_model = HistGradientBoostingRegressor(
        learning_rate=0.19,
        max_iter=500,
        max_depth=3,
        loss="absolute_error",
        random_state=42,
    )

    # Train the model on the training dataset
    hist_model.fit(X_train, y_train)

    # Return the trained model
    return hist_model


def test_regression(hist_model, test_df):
    """
    Evaluates a trained regression model on the test dataset and returns predictions.

    Parameters:
    hist_model (HistGradientBoostingRegressor): The trained regression model.
    test_df (pandas.DataFrame): The test dataset including both features and the target variable ('price').

    Returns:
    numpy.ndarray: Predicted prices for the test set
    """

    # Separate the target variable from the test features
    y_test = test_df["price"]
    X_test = test_df.drop("price", axis=1)

    # Generate predictions using the trained model
    y_pred_regression = hist_model.predict(X_test)

    # Print the Mean Absolute Error (MAE) for model evaluation
    print(
        "Mean Absolute Error for Regression:",
        mean_absolute_error(y_test, y_pred_regression),
    )

    # Return the predicted values
    return y_pred_regression


def classification_feature_engineering(filename):
    """

    Preprocesses the property dataset by performing the following steps:
     -Reads CSV file and drops unnecessary columns
     -Creates new engineered features:
        - 'time_to_cbd_avg': the average of public transport and driving time to the CBD.
        - 'total_rooms': sum of number of bedrooms and bathrooms.
        - 'inv_cbd_dist': inverse of distance to the CBD for emphasizing proximity.
     -Handles missing values in newly created features.


    Parameters:
    filename (str): name of the input CSV file

    Returns:
    pandas.DataFrame: Preprocessed dataset ready for classification modeling
    """

    dataset = pd.read_csv(filename, index_col="id")
    dataset = dataset.drop(
        columns=[
            "suburb_lat",
            "suburb_elevation",
            "ethnic_breakdown",
            "suburb_elevation",
            "suburb",
            "region",
            "cash_rate",
            "suburb_lng",
            "suburb_population",
            "suburb_sqkm",
            "suburbpopulation",
            "postcode",
            "public_housing_pct",
            "nearest_train_station",
            "highlights_attractions",
            "ideal_for",
            "traffic",
            "public_transport",
            "affordability_rental",
            "affordability_buying",
            "nature",
            "noise",
            "things_to_see_do",
            "family_friendliness",
            "pet_friendliness",
            "safety",
            "overall_rating",
            "date_sold",
            "suburb_median_income",
            "property_inflation_index",
        ]
    )

    # Create a new feature: average time to CBD by combining both transport types
    dataset["time_to_cbd_avg"] = dataset[
        [
            "time_to_cbd_public_transport_town_hall_st",
            "time_to_cbd_driving_town_hall_st",
        ]
    ].mean(axis=1)

    # Remove the original columns used to calculate average CBD time
    dataset = dataset.drop(
        [
            "time_to_cbd_public_transport_town_hall_st",
            "time_to_cbd_driving_town_hall_st",
        ],
        axis=1,
    )

    # Fill any missing values in the new average CBD time with the column mean
    dataset["time_to_cbd_avg"] = dataset["time_to_cbd_avg"].fillna(
        dataset["time_to_cbd_avg"].mean()
    )

    # Create a new feature: total number of rooms (bedrooms + bathrooms)
    dataset["total_rooms"] = dataset["num_bed"] + dataset["num_bath"]

    # Create a new feature: inverse distance to CBD to give more weight to closer properties
    dataset["inv_cbd_dist"] = 1 / (dataset["km_from_cbd"])

    # Drop the original 'km_from_cbd' after using it to create the new feature
    dataset = dataset.drop(columns=["km_from_cbd"])

    # Return feature engineered dataset
    return dataset


def train_classification(train_df):
    """
    Trains a Random Forest classifier on a cleaned and preprocessed training dataset.

    This function performs the following:
    1. Encodes the target categorical variable 'type' using Label Encoding.
    2. Separates features (X) and encoded target (y) for training.
    3. Trains a Random Forest Classifier with class balancing and specified hyperparameters.

    Parameters:

    train_df : pandas.DataFrame
        The training dataset containing features and the target 'type' column.

    Returns:

    rf_model : RandomForestClassifier
        Trained Random Forest model ready for classification predictions.

    label_encoder : LabelEncoder
        Fitted LabelEncoder used to transform and inverse-transform the target labels.
    """
    # Initialize label encoder to convert categorical target into numeric labels
    label_encoder = LabelEncoder()

    # Encode the 'type' column (target variable) and store in a new column
    train_df["type_encoded"] = label_encoder.fit_transform(train_df["type"])
    train_df = train_df.drop(columns=["type"])

    # Separate the features and target variable
    y_train = train_df["type_encoded"]
    X_train = train_df.drop(columns=["type_encoded"])

    # Initialize and configure the Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=600, max_depth=15, random_state=42, class_weight="balanced"
    )

    # Train the classifier on the feature set and labels
    rf_model.fit(X_train, y_train)

    return rf_model, label_encoder


def test_classification(rf_model, label_encoder, test_df):
    """
    Tests a trained Random Forest classification model on the test dataset
    and evaluates performance using the weighted F1 score.

    This function performs the following:
    1. Encodes the target column 'type' in the test set using the provided label encoder.
    2. Separates features and labels for prediction.
    3. Generates predictions using the trained model.
    4. Calculates and prints the weighted F1 score.
    5. Returns the predicted class labels in their original string form.

    Parameters:

    rf_model : RandomForestClassifier
        The trained Random Forest classifier.

    label_encoder : LabelEncoder
        Fitted LabelEncoder used to encode and decode the target labels.

    test_df : pandas.DataFrame
        The test dataset containing features and the original target column 'type'.

    Returns:

    np.ndarray
        Predicted labels in their original (non-encoded) form.
    """

    # Encode the target column 'type' using the same encoder as used during training
    test_df["type_encoded"] = label_encoder.transform(test_df["type"])
    test_df = test_df.drop(columns=["type"])

    # Separate the features and target variable
    y_test = test_df["type_encoded"]
    X_test = test_df.drop(columns=["type_encoded"])

    # Predict the target labels using the trained model
    y_pred_classification = rf_model.predict(X_test)

    # Calculate and print the weighted F1 score for model evaluation
    print(
        "F1 score for classification:",
        f1_score(y_test, y_pred_classification, average="weighted", zero_division=1),
    )

    # Return the predicted labels in their original class names
    return label_encoder.inverse_transform(y_pred_classification)


if __name__ == "__main__":
    """ This script is designed to be run from the command line with two arguments:
    1. The name of the training CSV file
    2. The name of the testing CSV file
    """

    train_csv = str(sys.argv[1])

    test_csv = str(sys.argv[2])

    """ Load the datasets and perform feature engineering for regression """
    regression_train_dataset = regression_feature_engineering(train_csv)
    regression_test_dataset = regression_feature_engineering(test_csv)

    """Train and test regression model"""

    regression_model = train_regression(regression_train_dataset)

    regression_predictions = test_regression(regression_model, regression_test_dataset)

    """ Load the datasets and perform feature engineering for classification """
    classification_train_dataset = classification_feature_engineering(train_csv)
    classification_test_dataset = classification_feature_engineering(test_csv)

    """Train and test classification model"""

    classification_model, label_encoder = train_classification(
        classification_train_dataset
    )

    classification_predictions = test_classification(
        classification_model, label_encoder, classification_test_dataset
    )

    """Write the predictions to CSV files"""

    regression_prediction_df = pd.DataFrame(
        {"id": regression_test_dataset.index, "price": regression_predictions}
    )
    regression_prediction_df.to_csv("regression.csv", index=False)

    classification_prediction_df = pd.DataFrame(
        {"id": classification_test_dataset.index, "type": classification_predictions}
    )
    classification_prediction_df.to_csv("classification.csv", index=False)
