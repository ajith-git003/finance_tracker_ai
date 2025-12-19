import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
# Expecting 'serviceAccountKey.json' in the same directory or specified via env var
KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")

def get_db():
    if not firebase_admin._apps:
        if os.path.exists(KEY_PATH):
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
            print(f"Firebase initialized with {KEY_PATH}")
        else:
            print(f"Error: {KEY_PATH} not found. Please place the file in the project root.")
            return None
    
    return firestore.client()

if __name__ == "__main__":
    db = get_db()
    if db:
        print("Firestore client created successfully.")
