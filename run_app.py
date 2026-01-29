import os
import subprocess
import time
import webbrowser
import sys

def run_streamlit():
    # 1. Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "streamlit_app.py")
    
    if not os.path.exists(app_path):
        print(f"错误: 找不到 {app_path}")
        return

    print("🚀 正在启动温度计指标 Web 应用...")
    
    # 2. Command to run streamlit
    # Use sys.executable to ensure we use the same python environment
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    # 3. Start the process
    process = subprocess.Popen(cmd)
    
    # 4. Wait a bit for the server to start
    time.sleep(3)
    
    # 5. Open the browser
    url = "http://localhost:8501"
    print(f"🌐 正在自动打开浏览器: {url}")
    webbrowser.open(url)
    
    try:
        # Keep the script running to maintain the process
        process.wait()
    except KeyboardInterrupt:
        print("\n👋 正在关闭应用...")
        process.terminate()

if __name__ == "__main__":
    run_streamlit()
