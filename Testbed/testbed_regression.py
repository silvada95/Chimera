import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys

# ---------------- Configuration ----------------
DATA_FILE = 'testbed_results.csv'
data_options = ["pedro_shist.yaml", "pedro_mdes.yaml", "pedro_taf.yaml", "pedro_vtei.yaml"]
selected_data_option = "pedro_shist.yaml"   # change as needed
target = "mAP50-95"
INCLUDE_MAE = True   # Set to False if you do NOT want MAE in the weights file
RANDOM_STATE = 42

# Define the six feature combinations (W1..W6)
feature_sets = {
    "W1": ["mac backbone", "meco"],
    "W2": ["mac backbone", "az-expressivity"],
    "W3": ["zen score", "mac backbone"],
    "W4": ["mac backbone", "az-expressivity", "meco"],
    "W5": ["zen score", "mac backbone", "meco"],
    "W6": ["zen score", "mac backbone", "meco", "az-expressivity"],
}

# Collect all unique features (keep stable readable order ? you can define a custom order if preferred)
all_features = ["mac backbone", "meco", "zen score", "az-expressivity"]

# ---------------- Load & Filter Data ----------------
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: File '{DATA_FILE}' not found.", file=sys.stderr)
    sys.exit(1)

if 'data' not in df.columns:
    print("ERROR: 'data' column not found in the CSV.", file=sys.stderr)
    sys.exit(1)

if selected_data_option not in data_options:
    print(f"WARNING: selected_data_option '{selected_data_option}' not in predefined data_options list.")

df_filtered = df[df['data'] == selected_data_option].copy()
if df_filtered.empty:
    print(f"ERROR: No rows found for data == '{selected_data_option}'.", file=sys.stderr)
    sys.exit(1)

if target not in df_filtered.columns:
    print(f"ERROR: Target column '{target}' not found in filtered DataFrame.", file=sys.stderr)
    sys.exit(1)

# ---------------- Validate Feature Availability ----------------
missing_features = [f for f in all_features if f not in df_filtered.columns]
if missing_features:
    print("ERROR: Missing required feature columns:")
    for m in missing_features:
        print(" -", m)
    sys.exit(1)

# ---------------- Prepare Data ----------------
X_all = df_filtered[all_features].values
scaler = StandardScaler()
X_all_norm = scaler.fit_transform(X_all)
y = df_filtered[target].values

feature_index_map = {f: i for i, f in enumerate(all_features)}

# ---------------- Compute Weights per Combination ----------------
weight_rows = []

for label, feats in feature_sets.items():
    indices = [feature_index_map[f] for f in feats]
    X_subset = X_all_norm[:, indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.5, random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    importances = model.feature_importances_

    # Prepare row with zeros for all features
    row = {"Combination": label}
    if INCLUDE_MAE:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        row["MAE"] = mae

    # Map importances
    importance_map = {f: 0.0 for f in all_features}
    for f, imp in zip(feats, importances):
        importance_map[f] = imp

    row.update(importance_map)
    weight_rows.append(row)

# ---------------- Build & Save Weights Table ----------------
weights_df = pd.DataFrame(weight_rows)

# Optional: sort by MAE if included, else by Combination
if INCLUDE_MAE:
    weights_df.sort_values(by="MAE", inplace=True)
else:
    weights_df.sort_values(by="Combination", inplace=True)

output_suffix = selected_data_option.replace('.yaml', '')
weights_outfile = f'feature_weights_{output_suffix}.csv'
weights_df.to_csv(weights_outfile, index=False)

print("\n=== Feature Weights (Importances) ===")
print(weights_df.to_string(index=False))
print(f"\nSaved weights to: {weights_outfile}")
