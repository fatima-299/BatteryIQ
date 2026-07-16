"""
BatteryIQ — Step 10: Feature Engineering
==========================================
Transforms combined_soh_clean.csv into ML-ready feature matrix.

Features computed:
  1. Lag features          — SOH at N-1, N-5, N-10 cycles ago
  2. Rolling statistics    — mean, std, min SOH over last 10/20 cycles
  3. Physics features      — capacity fade rate, resistance growth rate
  4. Temporal features     — normalized cycle, lifecycle stage
  5. Imputation            — fill missing values with source medians
  6. Encoding              — one-hot encode source and chemistry

Output → data/features/feature_matrix.csv

Run from BatteryIQ root:
  python pipeline/etl/05_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
FEAT_DIR  = ROOT / "data" / "features"
FIG_DIR   = ROOT / "memoire" / "figures"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load clean combined dataset ────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    path = FEAT_DIR / "combined_soh_clean.csv"
    df   = pd.read_csv(path)
    print(f"✅ Loaded: {len(df):,} rows × {df.shape[1]} cols")
    print(f"   Sources   : {df['source'].value_counts().to_dict()}")
    print(f"   Cells     : {df['cell_id'].nunique()}")
    return df


# ── Step 1: Sort correctly ─────────────────────────────────────────────────
def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['cell_id', 'cycle_number']).reset_index(drop=True)


# ── Step 2: Impute missing values ──────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Imputing missing values ...")

    # Compute source-level medians for imputation
    source_medians = df.groupby('source').agg({
        'avg_temp_c'         : 'median',
        'avg_voltage_v'      : 'median',
        'avg_current_a'      : 'median',
        'internal_resistance': 'median',
    }).round(4)

    print("   Source medians used for imputation:")
    print(source_medians.to_string())

    for src in df['source'].unique():
        mask = df['source'] == src
        for col in ['avg_temp_c', 'avg_voltage_v',
                    'avg_current_a', 'internal_resistance']:
            median_val = source_medians.loc[src, col]
            n_filled   = df.loc[mask & df[col].isna(), col].shape[0]
            df.loc[mask & df[col].isna(), col] = median_val
            if n_filled > 0:
                print(f"   {src:10s} {col:25s}: filled {n_filled:,} with {median_val:.4f}")

    return df


# ── Step 3: Lag features ───────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  Computing lag features ...")

    lags = [1, 5, 10, 20]
    for lag in lags:
        df[f'soh_lag_{lag}'] = df.groupby('cell_id')['soh_pct'].shift(lag)
        df[f'cap_lag_{lag}'] = df.groupby('cell_id')['cycle_capacity_ah'].shift(lag)

    # SOH change from previous cycles
    df['soh_delta_1']  = df['soh_pct'] - df['soh_lag_1']
    df['soh_delta_5']  = df['soh_pct'] - df['soh_lag_5']
    df['soh_delta_10'] = df['soh_pct'] - df['soh_lag_10']

    print(f"   Added {len(lags)*2 + 3} lag/delta features")
    return df


# ── Step 4: Rolling statistics ─────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📈 Computing rolling statistics ...")

    windows = [5, 10, 20]
    for w in windows:
        grp = df.groupby('cell_id')['soh_pct']
        df[f'soh_roll_mean_{w}'] = grp.transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f'soh_roll_std_{w}']  = grp.transform(
            lambda x: x.rolling(w, min_periods=2).std().fillna(0))
        df[f'soh_roll_min_{w}']  = grp.transform(
            lambda x: x.rolling(w, min_periods=1).min())

        # Rolling internal resistance mean
        if 'internal_resistance' in df.columns:
            grp_ir = df.groupby('cell_id')['internal_resistance']
            df[f'ir_roll_mean_{w}'] = grp_ir.transform(
                lambda x: x.rolling(w, min_periods=1).mean())

    print(f"   Added {len(windows)*4} rolling features")
    return df


# ── Step 5: Physics-derived features ──────────────────────────────────────
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚛️  Computing physics-derived features ...")

    # 1. Capacity fade rate — slope of capacity over last 10 cycles
    def rolling_slope(series, window=10):
        """Compute slope of linear fit over rolling window."""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            start = max(0, i - window + 1)
            y     = series.iloc[start:i+1].values
            if len(y) < 3:
                slopes.iloc[i] = np.nan
                continue
            x     = np.arange(len(y))
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
            except Exception:
                slopes.iloc[i] = np.nan
        return slopes

    print("   Computing capacity fade rate (per cell) ...")
    fade_rates = df.groupby('cell_id')['cycle_capacity_ah'].transform(
        lambda x: rolling_slope(x, window=10)
    )
    df['capacity_fade_rate'] = fade_rates

    # 2. SOH acceleration — second derivative (rate of change of rate of change)
    df['soh_acceleration'] = df.groupby('cell_id')['soh_delta_1'].transform(
        lambda x: x.diff()
    ) if 'soh_delta_1' in df.columns else np.nan

    # 3. Resistance growth rate
    if 'internal_resistance' in df.columns:
        df['ir_growth_rate'] = df.groupby('cell_id')['internal_resistance'].transform(
            lambda x: x.diff()
        )
        # Cumulative resistance increase from start
        df['ir_cumulative_growth'] = df.groupby('cell_id')['internal_resistance'].transform(
            lambda x: x - x.iloc[0]
        )

    # 4. Normalized capacity — capacity relative to cell's own first cycle
    df['cap_normalized'] = df.groupby('cell_id')['cycle_capacity_ah'].transform(
        lambda x: x / x.iloc[0] if x.iloc[0] > 0 else x
    )

    # 5. Arrhenius feature — if temperature available
    # Degradation rate ∝ exp(-Ea/RT) → use 1/T as physics proxy
    R  = 8.314   # gas constant J/(mol·K)
    Ea = 50000   # typical activation energy for SEI growth ~50 kJ/mol
    if 'avg_temp_c' in df.columns:
        T_kelvin = df['avg_temp_c'] + 273.15
        df['arrhenius_factor'] = np.exp(-Ea / (R * T_kelvin))

    print(f"   Added capacity_fade_rate, ir_growth_rate, ir_cumulative_growth,")
    print(f"   cap_normalized, arrhenius_factor, soh_acceleration")
    return df


# ── Step 6: Temporal features ──────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🕐 Computing temporal features ...")

    # Max cycle per cell (for normalisation)
    max_cycle = df.groupby('cell_id')['cycle_number'].transform('max')
    min_cycle = df.groupby('cell_id')['cycle_number'].transform('min')

    # Normalised cycle position: 0 = start, 1 = end of observation
    df['cycle_normalized'] = (
        (df['cycle_number'] - min_cycle) /
        (max_cycle - min_cycle + 1e-8)
    )

    # Lifecycle stage
    def get_stage(norm):
        if norm <= 0.33:   return 0   # early
        elif norm <= 0.66: return 1   # mid
        else:              return 2   # late
    df['lifecycle_stage'] = df['cycle_normalized'].apply(get_stage)

    # Cycles since start (absolute)
    df['cycles_from_start'] = df['cycle_number'] - min_cycle

    print(f"   Added cycle_normalized, lifecycle_stage, cycles_from_start")
    return df


# ── Step 7: Encode categorical features ───────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔤 Encoding categorical features ...")

    # One-hot encode source
    source_dummies = pd.get_dummies(df['source'], prefix='src').astype(int)
    df = pd.concat([df, source_dummies], axis=1)

    # One-hot encode chemistry
    chem_dummies = pd.get_dummies(df['chemistry'], prefix='chem').astype(int)
    df = pd.concat([df, chem_dummies], axis=1)

    # Label encode status
    df['status_encoded'] = (df['status'] == 'end_of_life').astype(int)

    print(f"   Source dummies : {list(source_dummies.columns)}")
    print(f"   Chem dummies   : {list(chem_dummies.columns)}")
    return df


# ── Step 8: Define final feature list ─────────────────────────────────────
def select_features(df: pd.DataFrame) -> tuple:
    """
    Returns (feature_df, feature_names, target_names)
    Drops rows with too many NaN in core features.
    """

    # Core features available for all sources
    core_features = [
        # Identity
        'cell_id', 'source', 'chemistry', 'cycle_number',

        # Targets
        'soh_pct', 'rul_cycles', 'status_encoded',

        # Raw features
        'cycle_capacity_ah', 'nominal_capacity_ah',
        'avg_temp_c', 'internal_resistance',
        'avg_voltage_v', 'avg_current_a',

        # Lag features
        'soh_lag_1', 'soh_lag_5', 'soh_lag_10', 'soh_lag_20',
        'cap_lag_1', 'cap_lag_5',
        'soh_delta_1', 'soh_delta_5', 'soh_delta_10',

        # Rolling features
        'soh_roll_mean_5',  'soh_roll_std_5',  'soh_roll_min_5',
        'soh_roll_mean_10', 'soh_roll_std_10', 'soh_roll_min_10',
        'soh_roll_mean_20', 'soh_roll_std_20', 'soh_roll_min_20',
        'ir_roll_mean_5', 'ir_roll_mean_10', 'ir_roll_mean_20',

        # Physics features
        'capacity_fade_rate', 'ir_growth_rate',
        'ir_cumulative_growth', 'cap_normalized',
        'arrhenius_factor', 'soh_acceleration',

        # Temporal
        'cycle_normalized', 'lifecycle_stage', 'cycles_from_start',

        # Encoded
        'src_nasa', 'src_stanford', 'src_calce',
        'chem_NMC', 'chem_LFP', 'chem_CS2', 'chem_CX2',
    ]

    # Keep only columns that exist
    available = [c for c in core_features if c in df.columns]
    missing   = [c for c in core_features if c not in df.columns]
    if missing:
        print(f"\n   ⚠️  Missing columns (will skip): {missing}")

    feat_df = df[available].copy()

    # Drop rows where soh_pct (target) is null
    feat_df = feat_df.dropna(subset=['soh_pct'])

    # ML feature columns (exclude identity and targets)
    id_cols     = ['cell_id', 'source', 'chemistry', 'cycle_number']
    target_cols = ['soh_pct', 'rul_cycles', 'status_encoded']
    ml_features = [c for c in available
                   if c not in id_cols + target_cols]

    return feat_df, ml_features, target_cols


# ── Step 9: Quality report ─────────────────────────────────────────────────
def quality_report(df: pd.DataFrame, ml_features: list):
    print("\n🔍 Feature Matrix Quality Report")
    print("=" * 55)
    print(f"  Total rows          : {len(df):,}")
    print(f"  Total features      : {len(ml_features)}")
    print(f"  Cells               : {df['cell_id'].nunique()}")
    print(f"  SOH range           : {df['soh_pct'].min():.1f}% → {df['soh_pct'].max():.1f}%")
    print(f"\n  Missing values per feature (top 10):")
    miss = df[ml_features].isna().mean().sort_values(ascending=False)
    for feat, pct in miss.head(10).items():
        if pct > 0:
            bar = '█' * int(pct * 20)
            print(f"    {feat:35s}: {pct*100:.1f}%  {bar}")

    print(f"\n  Feature groups:")
    print(f"    Raw features       : cycle_capacity_ah, avg_temp_c, internal_resistance, ...")
    print(f"    Lag features       : soh_lag_1/5/10/20, soh_delta_1/5/10")
    print(f"    Rolling features   : soh_roll_mean/std/min (5,10,20 cycles)")
    print(f"    Physics features   : capacity_fade_rate, ir_growth_rate, arrhenius_factor")
    print(f"    Temporal features  : cycle_normalized, lifecycle_stage")
    print(f"    Encoded features   : src_*, chem_*")

    print(f"\n  Sample of feature correlations with SOH:")
    num_feats = [f for f in ml_features
                 if df[f].dtype in [float, int] and f in df.columns]
    corr = df[num_feats + ['soh_pct']].corr()['soh_pct'].drop('soh_pct')
    corr = corr.abs().sort_values(ascending=False)
    for feat, val in corr.head(15).items():
        bar = '█' * int(val * 20)
        print(f"    {feat:35s}: {val:.3f}  {bar}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — Feature Engineering Pipeline")
    print("=" * 55)

    # 1. Load
    df = load_data()

    # 2. Sort
    df = sort_data(df)

    # 3. Impute
    df = impute_missing(df)

    # 4. Lag features
    df = add_lag_features(df)

    # 5. Rolling features
    df = add_rolling_features(df)

    # 6. Physics features
    df = add_physics_features(df)

    # 7. Temporal features
    df = add_temporal_features(df)

    # 8. Encode categoricals
    df = encode_categoricals(df)

    # 9. Select final features
    feat_df, ml_features, target_cols = select_features(df)

    # 10. Save
    out_path = FEAT_DIR / "feature_matrix.csv"
    feat_df.to_csv(out_path, index=False)

    # 11. Quality report
    quality_report(feat_df, ml_features)

    print(f"\n  💾 Saved → {out_path}")
    print(f"     Rows     : {len(feat_df):,}")
    print(f"     Columns  : {feat_df.shape[1]}")
    print(f"     ML feats : {len(ml_features)}")

    # Save feature list for ML scripts
    feat_list_path = FEAT_DIR / "feature_list.txt"
    with open(feat_list_path, 'w') as f:
        f.write("# BatteryIQ ML Features\n")
        f.write("# Generated by 05_feature_engineering.py\n\n")
        f.write("TARGETS:\n")
        for t in target_cols:
            f.write(f"  {t}\n")
        f.write("\nML_FEATURES:\n")
        for feat in ml_features:
            f.write(f"  {feat}\n")

    print(f"     Feature list saved → feature_list.txt")
    print("\n✅ Feature engineering complete!")
    print("   Next: python pipeline/etl/06_pyspark_etl.py")


if __name__ == "__main__":
    main()
