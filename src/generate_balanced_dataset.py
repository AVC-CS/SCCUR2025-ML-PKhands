import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def generate_balanced(samples_per_class, out_csv):
    # Load the full poker dataset
    df = pd.read_csv('../data/poker-hand-testing.data', header=None)

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Oversample
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    df_res = pd.concat([X_res, y_res], axis=1)

    # Limit per class
    df_bal = df_res.groupby(df_res.columns[-1]).head(samples_per_class)

    # Add column names
    cols = ['S1','R1','S2','R2','S3','R3','S4','R4','S5','R5','ORD']
    df_bal.columns = cols

    # Save with header
    df_bal.to_csv(out_csv, index=False)

    print(f"Saved {len(df_bal)} rows to {out_csv}")
    return df_bal

