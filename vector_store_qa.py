# -*- coding: utf-8 -*-

# This script manages a vector store for question-answer (Q&A) retrieval
# using OpenAI embeddings to calculate text similarity.

# The script defines the VectorStoreManager class, which provides methods to:
# 1. Initialize the vector store using the Chroma library and OpenAI embeddings.
#    - The vector store is saved in a persistent directory ("./chroma_db").
# 2. Add question-answer pairs to the vector store.
#    - Each pair is indexed, enabling future similarity-based searches.
# 3. Perform similarity searches for questions within the vector store:
#    - Upon receiving a new question, it searches for similar questions with a minimum similarity score
#      (defined by a similarity threshold).
#    - Returns the original question, answer, and similarity score if a suitable match is found.
# 4. Clear all data from the vector store:
#    - Deletes all data from the current collection and creates a new empty collection.

# Additionally, the script includes error handling and logging to monitor for issues in initialization,
# search, data addition, and data deletion operations in the vector store.


from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Optional, Tuple
import os
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, api_key: str, similarity_threshold: float = 0.85):
        """
        Initialize the vector store manager.
        
        Args:
            api_key: OpenAI API key for embeddings
            similarity_threshold: Minimum similarity score to consider a match (0-1)
        """
        self.api_key = api_key  # Armazena a API key para comparação
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.similarity_threshold = similarity_threshold
        self.persist_directory = "./chroma_db"
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize or load the vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load the Chroma vector store."""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def find_similar_question(self, question: str) -> Optional[Tuple[str, str, float]]:
        """
        Search for similar questions in the vector store.
        
        Args:
            question: The question to search for
            
        Returns:
            Tuple of (question, answer, similarity_score) if found, None otherwise
        """
        try:
            # Search for similar questions
            results = self.vector_store.similarity_search_with_relevance_scores(
                question,
                k=1  # Get the most similar result
            )
            
            if not results:
                return None
                
            # Get the most similar result
            document, similarity_score = results[0]
            
            # Extract original question and answer from metadata
            original_question = document.metadata.get('question')
            answer = document.metadata.get('answer')
            
            # Check if similarity meets our threshold
            if similarity_score >= self.similarity_threshold:
                logger.info(f"Similar question found with score: {similarity_score}")
                return original_question, answer, similarity_score
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return None
    
    def add_qa_pair(self, question: str, answer: str):
        """
        Add a question-answer pair to the vector store.
        
        Args:
            question: The question
            answer: The answer
        """
        try:
            # Add the Q&A pair to the vector store
            self.vector_store.add_texts(
                texts=[question],
                metadatas=[{
                    'question': question,
                    'answer': answer
                }]
            )
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info("Q&A pair added to vector store successfully")
            
        except Exception as e:
            logger.error(f"Error adding Q&A pair to vector store: {str(e)}")
            raise

    def clear_vector_store(self):
        """Clear all data from the vector store."""
        try:
            # Ao invés de deletar o diretório e reinicializar,
            # vamos deletar todos os dados da coleção atual
            self.vector_store.delete_collection()
            
            # Criar uma nova coleção vazia
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
