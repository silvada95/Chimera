import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('testbed_results.csv')

# --------------------------------------------------
# 1. Weight configurations (W1 ... W6)
# Order of features these weights correspond to:
# ["mac backbone", "zen score", "az-expressivity", "az-progressivity", "az-trainability", "meco", "param backbone"]
WEIGHT_SETS = {
    "W1": [0.4, 0,   0,    0,   0,   0.6, 0],
    "W2": [0.6, 0,   0.4,  0,   0,   0,   0],
    "W3": [0.35,0.65,0,    0,   0,   0,   0],
    "W4": [0.32,0,   0.18, 0,   0,   0.5, 0],
    "W5": [0.22,0.66,0,    0,   0,   0.18,0],
    "W6": [0.19,0.58,0.06, 0,   0,   0.17,0],
}

# The ordered feature list used by the weight vectors
WEIGHT_FEATURES = [
    "mac backbone",
    "zen score",
    "az-expressivity",
    "az-progressivity",
    "az-trainability",
    "meco",
    "param backbone"
]

# --------------------------------------------------
def add_weighted_columns(df, weight_sets=WEIGHT_SETS, feature_list=WEIGHT_FEATURES, normalize=True, prefix_as_is=True):
    """
    Creates weighted composite columns (e.g., W1..W6) based on provided weight sets.
    Normalizes each feature (Z-score) before applying weights if normalize=True.
    """
    working = df.copy()

    if normalize:
        for feat in feature_list:
            mean_val = working[feat].mean()
            std_val = working[feat].std()
            if std_val == 0:
                # Avoid division by zero; if constant, set to 0 contribution
                working[feat] = 0.0
            else:
                working[feat] = (working[feat] - mean_val) / std_val

    # Generate each weighted column
    for name, weights in weight_sets.items():
        if len(weights) != len(feature_list):
            raise ValueError(f"Weight set {name} length {len(weights)} does not match feature count {len(feature_list)}.")
        weighted_sum = np.zeros(len(working))
        for w, feat in zip(weights, feature_list):
            if w != 0:
                weighted_sum += w * working[feat]
        df[name] = weighted_sum  # add to original df

    return df

# --------------------------------------------------
def calculate_correlations(df, category_filter=None, data_filter=None):
    """
    Computes Kendall and Spearman correlations against mAP50-95 for
    base metrics + weighted composites (W1..W6).
    """
    filtered = df.copy()
    if category_filter:
        filtered = filtered[filtered['category'].isin(category_filter)]
    if data_filter:
        filtered = filtered[filtered['data'].isin(data_filter)]

    print("Filtered DataFrame Length for Correlations:", len(filtered))

    # Keys to correlate (include base metrics + W columns)
    keys_to_check = [
        "param backbone", "mac backbone",
        "zen score",
        "az-expressivity", "az-progressivity", "az-trainability",
        "meco",
        # Weighted composites:
        "W1", "W2", "W3", "W4", "W5", "W6"
    ]

    correlation_results = {
        'key': [],
        'kendall_correlation': [],
        'spearman_correlation': []
    }

    for key in keys_to_check:
        if key not in filtered.columns:
            print(f"Warning: {key} not found in DataFrame; skipping.")
            continue
        kendall_corr = filtered[key].corr(filtered['mAP50-95'], method='kendall')
        spearman_corr = filtered[key].corr(filtered['mAP50-95'], method='spearman')
        correlation_results['key'].append(key)
        correlation_results['kendall_correlation'].append(kendall_corr)
        correlation_results['spearman_correlation'].append(spearman_corr)

    return pd.DataFrame(correlation_results)

# --------------------------------------------------
def calculate_mse(df, data_filter=None, top_percentile=0.10):
    """
    For the top X% rows by mAP50-95, computes MSE between the reference
    ranking (sorted by mAP50-95) and the ranking induced by each key.
    """
    filtered = df.copy()
    if data_filter:
        filtered = filtered[filtered['data'].isin(data_filter)]

    if filtered.empty:
        raise ValueError("Filtered DataFrame is empty; cannot compute MSE.")

    threshold = filtered['mAP50-95'].quantile(1 - top_percentile)
    top_percent_df = filtered[filtered['mAP50-95'] >= threshold]

    print("Top Percent DataFrame Length for MSE:", len(top_percent_df))

    reference_mAP50_values = top_percent_df['mAP50-95'].sort_values(ascending=False).values

    keys_to_check = [
        "param backbone", "mac backbone",
        "zen score",
        "az-expressivity", "az-progressivity", "az-trainability",
        "meco",
        "W1", "W2", "W3", "W4", "W5", "W6"
    ]

    mse_results = {'key': [], 'mean_squared_error': []}

    for key in keys_to_check:
        if key not in top_percent_df.columns:
            print(f"Warning: {key} not found; skipping MSE.")
            continue

        # Sort rows by the given key descending
        sorted_df = top_percent_df.sort_values(by=key, ascending=False)
        sorted_mAP_values = sorted_df['mAP50-95'].values

        # Compare with reference ordering
        mse = np.mean((reference_mAP50_values[:len(sorted_mAP_values)] - sorted_mAP_values) ** 2)
        mse_results['key'].append(key)
        mse_results['mean_squared_error'].append(mse)

    return pd.DataFrame(mse_results)

# --------------------------------------------------
# Example usage
if __name__ == "__main__":
    category_options = ["full heterogeneous", "full mambas", "full maxvit", "full c2f", "full mlp"]
    data_options = ["pedro_shist.yaml"]

    # 1. Add weighted composite columns W1..W6
    df = add_weighted_columns(df)

    # 2. Correlations
    correlations = calculate_correlations(df, category_filter=category_options, data_filter=data_options)
    print("Correlations:")
    print(correlations)

    # 3. MSE (top 10%)
    mse = calculate_mse(df, data_filter=data_options, top_percentile=0.10)
    print("Mean Squared Error:")
    print(mse)

    # 4. Merge and save
    merged_results = correlations.merge(mse, on='key', how='left')
    print("Merged Results:")
    print(merged_results)

    merged_results.to_csv('corr_results_shist.csv', index=False)
    print("Merged results saved to 'corr_results_shist.csv'.")
