import pandas as pd
import numpy as np
import os
import glob

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

def nonlinear_transform(series, low_threshold=30, high_threshold=70):
    s = series.copy()
    mask_low = s < low_threshold
    s.loc[mask_low] = (s.loc[mask_low] ** 2) / 100
    mask_high = s > high_threshold
    s.loc[mask_high] = np.sqrt(s.loc[mask_high]) * 10
    return s

def calculate_thermometer(df, use_nonlinear=False):
    if all(col in df.columns for col in ['high', 'low', 'close']):
        src = (df['high'] + df['low'] + df['close'] * 2) / 4
    else:
        src = df['close']
    
    # 1. RSI Factor
    rsi = pine_rsi(src, 14)
    
    # 2. TSI Factor
    bar_index = pd.Series(np.arange(len(df)), index=df.index)
    tsi = src.rolling(window=14).corr(bar_index)
    tsi_norm = (tsi + 1) / 2 * 100
    
    # 3. BB%B Factor
    sma_bb = src.rolling(window=20).mean()
    std_bb = src.rolling(window=20).std()
    bb_percent = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100
    bb_percent = bb_percent.clip(0, 100)
    
    # --- Non-linear Transformation ---
    if use_nonlinear:
        rsi = nonlinear_transform(rsi)
        tsi_norm = nonlinear_transform(tsi_norm)
        bb_percent = nonlinear_transform(bb_percent)
    
    # 4. Weighted Composite
    thermometer = (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)
    
    return thermometer

def evaluate_models():
    csv_files = glob.glob("*.csv")
    results = []
    
    print(f"Found {len(csv_files)} CSV files.")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
            
            if '温度计' not in df.columns:
                continue
                
            # Calculate Linear Model
            linear_model = calculate_thermometer(df, use_nonlinear=False)
            
            # Calculate Non-linear Model
            nonlinear_model = calculate_thermometer(df, use_nonlinear=True)
            
            # Filter valid data for comparison
            valid_mask = df['温度计'].notnull() & linear_model.notnull() & nonlinear_model.notnull()
            
            if valid_mask.sum() < 50:
                continue
                
            original = df.loc[valid_mask, '温度计']
            linear = linear_model[valid_mask]
            nonlinear = nonlinear_model[valid_mask]
            
            r2_linear = original.corr(linear)**2
            r2_nonlinear = original.corr(nonlinear)**2
            
            results.append({
                "File": file,
                "Linear R2": r2_linear,
                "Non-linear R2": r2_nonlinear,
                "Improvement": r2_nonlinear - r2_linear
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_results = pd.DataFrame(evaluate_models())
    if not df_results.empty:
        print("\n--- Model Comparison Results ---")
        print(df_results)
        
        avg_linear = df_results['Linear R2'].mean()
        avg_nonlinear = df_results['Non-linear R2'].mean()
        print(f"\nAverage Linear R2: {avg_linear:.4f}")
        print(f"Average Non-linear R2: {avg_nonlinear:.4f}")
        
        if avg_nonlinear > avg_linear:
            print("\nResult: Non-linear model IMPROVED the fit.")
        else:
            print("\nResult: Non-linear model DECREASED the fit.")
    else:
        print("No valid data found for comparison.")
