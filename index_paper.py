"""One-off: embed the capstone paper into Atlas and build the vector index.

Run from a folder containing both the .docx and a .env with MONGODB_URI +
GOOGLE_API_KEY:  python index_paper.py
"""

from dotenv import load_dotenv

# Safe to import from app.py without pulling in Streamlit: the UI code in
# app.py only runs inside `if __name__ == "__main__"`, so this import just
# picks up the plain function.
from app import index_paper

load_dotenv()
print(index_paper())
