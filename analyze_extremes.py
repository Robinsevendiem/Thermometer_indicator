import pandas as pd
import numpy as np
import glob
from scipy.optimize import minimize

# --- 1. Define Helper Functions ---
def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def pine_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = rma(up, period)
    avg_down = rma(down, period)
    rs = avg_up / avg_down
    return 100 - (100 / (1 + rs))

def calculate_components(df):
    # Source
    if all(col in df.columns for col in ['high', 'low', 'close']):
        src = (df['high'] + df['low'] + df['close'] * 2) / 4
    else:
        src = df['close']
    
    # Components
    # 1. RSI
    rsi = pine_rsi(src, 14)
    
    # 2. TSI (Correlation)
    bar_index = pd.Series(np.arange(len(df)), index=df.index)
    tsi = src.rolling(window=14).corr(bar_index)
    tsi_norm = (tsi + 1) / 2 * 100
    
    # 3. BB%B
    sma_bb = src.rolling(window=20).mean()
    std_bb = src.rolling(window=20).std()
    bb_percent = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100
    bb_percent = bb_percent.clip(0, 100)
    
    return rsi, tsi_norm, bb_percent

# --- 2. Load Data ---
csv_files = glob.glob("*.csv")
all_data = []

print(f"Loading {len(csv_files)} files...")

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df.columns = [c.lower() for c in df.columns]
        
        if '温度计' not in df.columns:
            continue
            
        rsi, tsi, bb = calculate_components(df)
        
        temp_df = pd.DataFrame({
            'rsi': rsi,
            'tsi': tsi,
            'bb': bb,
            'original': df['温度计']
        })
        
        # Drop NaNs
        temp_df = temp_df.dropna()
        all_data.append(temp_df)
        
    except Exception as e:
        print(f"Skipping {file}: {e}")

if not all_data:
    print("No valid data found.")
    exit()

full_df = pd.concat(all_data, ignore_index=True)
print(f"Total data points: {len(full_df)}")

# --- 3. Analyze Current Model Performance at Extremes ---
# Current Weights: 0.45, 0.26, 0.29
full_df['current_model'] = (full_df['rsi'] * 0.45) + (full_df['tsi'] * 0.26) + (full_df['bb'] * 0.29)
full_df['residual'] = full_df['original'] - full_df['current_model']

# Define Extremes
# High: Original > 75
# Low: Original < 25
high_mask = full_df['original'] > 75
low_mask = full_df['original'] < 25
mid_mask = (~high_mask) & (~low_mask)

mse_global = (full_df['residual']**2).mean()
mse_high = (full_df.loc[high_mask, 'residual']**2).mean()
mse_low = (full_df.loc[low_mask, 'residual']**2).mean()
mse_mid = (full_df.loc[mid_mask, 'residual']**2).mean()

print("\n--- Current Model Error (MSE) ---")
print(f"Global MSE: {mse_global:.4f}")
print(f"High (>75) MSE: {mse_high:.4f} (Count: {high_mask.sum()})")
print(f"Low  (<25) MSE: {mse_low:.4f}  (Count: {low_mask.sum()})")
print(f"Mid  (25-75) MSE: {mse_mid:.4f}")

print("\n--- Residual Bias (Mean Error) ---")
print(f"High Bias: {full_df.loc[high_mask, 'residual'].mean():.4f} (Positive means Original > Model)")
print(f"Low Bias:  {full_df.loc[low_mask, 'residual'].mean():.4f}")

# --- 4. Optimize Specifically for Extremes ---
def objective_function(weights, X, y):
    # weights = [w_rsi, w_tsi, w_bb]
    # Bias is assumed 0 for now to keep it simple, or we can optimize bias too
    pred = (X['rsi'] * weights[0]) + (X['tsi'] * weights[1]) + (X['bb'] * weights[2]) + weights[3]
    return ((y - pred)**2).mean()

# Prepare Data
X = full_df[['rsi', 'tsi', 'bb']]
y = full_df['original']

# Optimize High
print("\n--- Optimizing for HIGH Zone (>75) ---")
res_high = minimize(objective_function, [0.45, 0.26, 0.29, 0.0], 
                   args=(X[high_mask], y[high_mask]), 
                   bounds=[(0,1), (0,1), (0,1), (-10, 10)])
print(f"Optimal Weights High: RSI={res_high.x[0]:.3f}, TSI={res_high.x[1]:.3f}, BB={res_high.x[2]:.3f}, Bias={res_high.x[3]:.3f}")
print(f"New High MSE: {res_high.fun:.4f}")

# Optimize Low
print("\n--- Optimizing for LOW Zone (<25) ---")
res_low = minimize(objective_function, [0.45, 0.26, 0.29, 0.0], 
                  args=(X[low_mask], y[low_mask]), 
                  bounds=[(0,1), (0,1), (0,1), (-10, 10)])
print(f"Optimal Weights Low: RSI={res_low.x[0]:.3f}, TSI={res_low.x[1]:.3f}, BB={res_low.x[2]:.3f}, Bias={res_low.x[3]:.3f}")
print(f"New Low MSE: {res_low.fun:.4f}")

# Optimize Global with Bias
print("\n--- Optimizing Global (Check) ---")
res_global = minimize(objective_function, [0.45, 0.26, 0.29, 0.0], 
                     args=(X, y), 
                     bounds=[(0,1), (0,1), (0,1), (-10, 10)])
print(f"Optimal Weights Global: RSI={res_global.x[0]:.3f}, TSI={res_global.x[1]:.3f}, BB={res_global.x[2]:.3f}, Bias={res_global.x[3]:.3f}")

# --- 5. Check for Non-Linearity Evidence ---
# If High Weights are significantly different from Low Weights, it suggests Regime Switching.
