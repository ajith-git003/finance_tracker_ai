import os
import pandas as pd
import faiss
import numpy as np
import google.generativeai as genai

# Removing SentenceTransformer import
# from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self):
        # Use Google Embeddings instead of local SentenceTransformer
        self.embedding_model = 'models/text-embedding-004'
        self.index = None
        self.stored_data = None
        
        try:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None

    def get_embeddings(self, text_list):
        """
        Helper to get embeddings from Google API.
        Handles batching if necessary, though genai usually handles lists.
        """
        try:
            # embed_content returns a dictionary with 'embedding'
            # We need to loop if the API expects single strings or handles batches.
            # The 'batch_embed_contents' is better for lists.
            # Or we can iterate. text-embedding-004 supports batching.
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=text_list,
                task_type="retrieval_document",
                title="Transactions"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding Error: {e}")
            # Fallback or empty return
            return []

    def ingest_data(self, df: pd.DataFrame):
        if df.empty:
            return False, "DataFrame is empty."
        
        df['combined_text'] = df.apply(
            lambda x: f"Date: {x.get('Date', 'N/A')}, Description: {x.get('Description', 'N/A')}, Amount: {x.get('Amount', 'N/A')}, Category: {x.get('Category', 'N/A')}",
            axis=1
        )
        
        self.stored_data = df
        
        try:
            # Google Embeddings
            texts = df['combined_text'].tolist()
            embeddings = []
            
            # Batching to be safe (API limits)
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Note: embed_content for a list returns a list of embeddings
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
            
            # Initialize FAISS
            embeddings_np = np.array(embeddings).astype('float32')
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            
            return True, f"Ingested {len(df)} records using Gemini Embeddings."
            
        except Exception as e:
            return False, f"Ingestion Failed: {e}"

    def retrieve(self, query, k=5):
        if self.index is None or self.stored_data is None:
            return []
        
        try:
            # Embed query
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
            return "Error: Gemini model not initialized. Check GOOGLE_API_KEY in .env."

        context_items = self.retrieve(user_query)
        if not context_items:
            context = "No relevant transactions found."
        else:
            context = "\n".join(context_items)
            
        prompt = f"""
        You are a helpful financial assistant. Use the following fetched transaction data to answer the user's question.
        If the answer is not in the data, say so. Do not make up numbers.
        
        User Question: {user_query}
        
        Relevant Transactions:
        {context}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
