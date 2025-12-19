import pandas as pd
from db_config import get_db

def sync_csv_to_firestore(csv_file):
    """
    Reads a CSV file and syncs transactions to Firestore.
    Expected CSV columns: Date, Description, Amount, Category (optional)
    """
    db = get_db()
    if not db:
        return False, "Database connection failed."

    try:
        df = pd.read_csv(csv_file)
        # Basic validation
        required_cols = ['Date', 'Description', 'Amount']
        if not all(col in df.columns for col in required_cols):
            return False, f"CSV must contain columns: {required_cols}"

        # Convert to dictionary records
        records = df.to_dict(orient='records')
        
        batch = db.batch()
        count = 0
        
        # Firestore batch has a limit of 500 operations
        # For simplicity in this demo, we'll do simple iteration or chunks
        # But to keep it robust for small files:
        collection_ref = db.collection('transactions')
        
        for record in records:
            # Create a doc reference (auto-id)
            doc_ref = collection_ref.document()
            batch.set(doc_ref, record)
            count += 1
            
            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
                
        if count % 400 != 0:
            batch.commit()
            
        return True, f"Successfully synced {count} transactions."
        
    except Exception as e:
        return False, str(e)

def get_user_transactions():
    """
    Fetches all transactions from Firestore and returns a DataFrame.
    """
    db = get_db()
    if not db:
        return pd.DataFrame()

    try:
        docs = db.collection('transactions').stream()
        data = [doc.to_dict() for doc in docs]
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
