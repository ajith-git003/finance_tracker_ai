import os
import pandas as pd
import faiss
import numpy as np
import google.generativeai as genai
import pickle
import hashlib

class RAGEngine:
    def __init__(self):
        self.embedding_model = 'models/text-embedding-004'
        self.index = None
        self.stored_data = None
        self.cache_dir = ".cache/embeddings"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None

    def _get_cache_path(self, text_list):
        content_hash = hashlib.md5("".join(text_list).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{content_hash}.pkl")

    def ingest_data(self, df: pd.DataFrame):
        if df.empty:
            return False, "DataFrame is empty."
        
        df['combined_text'] = (
            "Date: " + df['Date'].astype(str) + 
            ", Description: " + df['Description'].astype(str) + 
            ", Amount: " + df['Amount'].astype(str) + 
            ", Category: " + df['Category'].astype(str)
        )
        
        self.stored_data = df
        texts = df['combined_text'].tolist()
        cache_path = self._get_cache_path(texts)

        try:
            # Try loading from cache first
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    embeddings = pickle.load(f)
            else:
                embeddings = []
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(response['embedding'])
                
                # Save to cache
                with open(cache_path, "wb") as f:
                    pickle.dump(embeddings, f)
            
            # Initialize FAISS
            embeddings_np = np.array(embeddings).astype('float32')
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            
            return True, f"Ingested {len(df)} records."
            
        except Exception as e:
            return False, f"Ingestion Failed: {e}"

    def retrieve(self, query, k=5):
        if self.index is None or self.stored_data is None:
            return []
        
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_vec = np.array([response['embedding']]).astype('float32')
            
            distances, indices = self.index.search(query_vec, k)
            
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.stored_data):
                    results.append(self.stored_data.iloc[idx]['combined_text'])
                    
            return results
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []

    def query_llm(self, user_query):
        if not self.model:
            return "Error: Gemini model not initialized."

        context_items = self.retrieve(user_query)
        context = "\n".join(context_items) if context_items else "No data found."
            
        prompt = f"""
        You are a helpful financial assistant. 
        Transaction Data:
        {context}
        
        User Question: {user_query}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
