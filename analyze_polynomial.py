import pandas as pd
import numpy as np
import glob
from scipy.optimize import curve_fit

# --- Reuse calculation logic ---
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

def calculate_base_score(df):
    if all(col in df.columns for col in ['high', 'low', 'close']):
        src = (df['high'] + df['low'] + df['close'] * 2) / 4
    else:
        src = df['close']
    
    rsi = pine_rsi(src, 14)
    bar_index = pd.Series(np.arange(len(df)), index=df.index)
    tsi = src.rolling(window=14).corr(bar_index)
    tsi_norm = (tsi + 1) / 2 * 100
    
    sma_bb = src.rolling(window=20).mean()
    std_bb = src.rolling(window=20).std()
    bb_percent = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100
    bb_percent = bb_percent.clip(0, 100)
    
    # Base Linear Model
    return (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)

# --- Load Data ---
csv_files = glob.glob("*.csv")
X_list = []
y_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df.columns = [c.lower() for c in df.columns]
        if '温度计' not in df.columns: continue
            
        base_score = calculate_base_score(df)
        valid = base_score.notnull() & df['温度计'].notnull()
        
        if valid.sum() > 0:
            X_list.append(base_score[valid].values)
            y_list.append(df.loc[valid, '温度计'].values)
    except: pass

if not X_list: 
    print("No data found")
    exit()

X = np.concatenate(X_list)
y = np.concatenate(y_list)

# --- 1. Linear Fit (Degree 1) ---
p1 = np.polyfit(X, y, 1)
pred_lin = np.polyval(p1, X)
r2_lin = 1 - (np.sum((y - pred_lin)**2) / np.sum((y - np.mean(y))**2))
print(f"Linear (Deg 1) R2: {r2_lin:.5f}")
print(f"Coeffs: {p1}")

# --- 2. Polynomial Fit (Degree 2) ---
p2 = np.polyfit(X, y, 2)
pred_poly2 = np.polyval(p2, X)
r2_poly2 = 1 - (np.sum((y - pred_poly2)**2) / np.sum((y - np.mean(y))**2))
print(f"Poly (Deg 2) R2: {r2_poly2:.5f}")
print(f"Coeffs: {p2}")

# --- 3. Polynomial Fit (Degree 3) ---
p3 = np.polyfit(X, y, 3)
pred_poly3 = np.polyval(p3, X)
r2_poly3 = 1 - (np.sum((y - pred_poly3)**2) / np.sum((y - np.mean(y))**2))
print(f"Poly (Deg 3) R2: {r2_poly3:.5f}")
print(f"Coeffs: {p3}")

# --- 4. Sigmoid Fit (Logistic) ---
def sigmoid_func(x, L, x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

try:
    # Initial guess: L=100 (max), x0=50 (midpoint), k=0.1 (slope), b=0 (min)
    p0 = [100, 50, 0.1, 0] 
    popt, pcov = curve_fit(sigmoid_func, X, y, p0=p0, maxfev=10000)
    pred_sig = sigmoid_func(X, *popt)
    r2_sig = 1 - (np.sum((y - pred_sig)**2) / np.sum((y - np.mean(y))**2))
    print(f"Sigmoid R2: {r2_sig:.5f}")
    print(f"Sigmoid Params: L={popt[0]:.4f}, x0={popt[1]:.4f}, k={popt[2]:.4f}, b={popt[3]:.4f}")
except Exception as e:
    print(f"Sigmoid fit failed: {e}")
