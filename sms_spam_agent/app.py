"""
Streamlit Cloud Entry Point Wrapper
This wrapper allows Streamlit Cloud to run sms_spam_agent/app.py seamlessly when configured as Main file path.
"""
import os
import sys

# Ensure repository root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Change current working directory to repository root
os.chdir(root_dir)

# Execute main app script
root_app_path = os.path.join(root_dir, "app.py")
with open(root_app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, root_app_path, "exec"), globals())
