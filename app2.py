# -*- coding: utf-8 -*-

# This script is a Streamlit-based web application for analyzing YouTube video content and answering questions
# about it. It integrates with OpenAI's language models and a vector database to provide responses, track usage,
# and optimize costs by caching similar queries.

# Main Functionalities:
# 1. **Token and Cost Tracking**: Calculates token usage and associated costs for input and output tokens based on
#    different OpenAI models, with a reset option for cost counters.
# 2. **YouTube Video Analysis**: Accepts a YouTube video URL, processes the video content, and displays the analysis
#    to the user. Supports downloading of analysis results in HTML format and displays a workflow diagram if available.
# 3. **Q&A Assistant with Vector Database Optimization**: Allows users to ask questions about the analyzed video content.
#    Before generating a new answer, the application checks a vector database (ChromaDB) for similar questions previously
#    asked, helping to reduce costs by retrieving answers from cache when a match is found.
# 4. **Session State Management**: Maintains session-based data such as API keys, selected model, token tracker,
#    chat history, video content, and vector store, ensuring continuity across interactions.
# 5. **Sidebar Configuration**: Provides input fields for API key and model selection, along with usage statistics
#    for tokens and cost, keeping users informed of their usage.
# 
# This script uses logging for event tracking and error handling, displaying messages to guide users in case of issues.



import streamlit as st
import os
from pathlib import Path
import app1
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from dataclasses import dataclass
from typing import Dict, Tuple
from vector_store_qa import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
logger.info("Initializing Streamlit application...")
st.set_page_config(page_title="AgentAI | YouTube Analysis", layout="wide")
st.title("YouTube Video Analysis | 'Agents AI' with LangGraph")


# Data class to define the pricing structure of a model's token usage
@dataclass
class ModelPricing:
    input_price: float
    output_price: float
    
    # Calculate input and output costs based on the number of tokens
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Tuple[float, float]:
        input_cost = (input_tokens / 1_000_000) * self.input_price
        output_cost = (output_tokens / 1_000_000) * self.output_price
        return input_cost, output_cost

# Class to track token usage and calculate associated costs for a model
class TokenCostTracker:
    def __init__(self):
        self.model_prices = {
            "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
            "gpt-4o": ModelPricing(2.5, 10.0),
            "gpt-4o-mini": ModelPricing(0.15, 0.60)
        }
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        
    # Count the tokens in the provided text using the specified model
    def count_tokens(self, text: str, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    # Count the characters in the provided text
    def count_chars(self, text: str) -> int:
        return len(text)
    
    # Add the input and output tokens and characters from a user interaction
    def add_interaction(self, input_text: str, output_text: str, model: str):
        # Count tokens
        input_tokens = self.count_tokens(input_text, model)
        output_tokens = self.count_tokens(output_text, model)
        
        # Count characters
        input_chars = self.count_chars(input_text)
        output_chars = self.count_chars(output_text)
        
        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_input_chars += input_chars
        self.total_output_chars += output_chars
    
    # Calculate and return the total cost based on the current token counts for the specified model
    def get_total_cost(self, model: str) -> Tuple[float, float, float]:
        pricing = self.model_prices.get(model)
        if not pricing:
            return 0.0, 0.0, 0.0
            
        input_cost, output_cost = pricing.calculate_cost(
            self.total_input_tokens,
            self.total_output_tokens
        )
        return input_cost, output_cost, input_cost + output_cost
    
    # Reset all token and character counts to zero
    def reset_counts(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_chars = 0
        self.total_output_chars = 0

# Check if a given URL is a valid YouTube URL
def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube video URL"""
    valid_domains = [
        'youtube.com',
        'www.youtube.com',
        'youtu.be',
        'm.youtube.com'
    ]
    try:
        # Check if URL contains video identifier
        if 'watch?v=' in url or 'youtu.be/' in url:
            # Check if domain is valid
            for domain in valid_domains:
                if domain in url:
                    return True
        return False
    except:
        return False

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_content' not in st.session_state:
    st.session_state.video_content = None
if 'token_tracker' not in st.session_state:
    st.session_state.token_tracker = TokenCostTracker()
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Setup sidebar configuration
logger.info("Setting up sidebar configuration...")
sidebar = st.sidebar

# Configure the sidebar elements
with sidebar:
    api_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.get("OPENAI_API_KEY", ""), 
        type="password"
    )
    if api_key:
        logger.info("API key updated")
        st.session_state.OPENAI_API_KEY = api_key
        
    # LLM model selector
    model_llm = st.selectbox(
        "OpenAI Model",
        options=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        index=1
    )
    logger.info(f"Selected model: {model_llm}")
    st.session_state["model_llm"] = model_llm
    
    # Usage Statistics
    st.divider()
    st.subheader("Usage Statistics")
    
    # Get costs
    input_cost, output_cost, total_cost = st.session_state.token_tracker.get_total_cost(
        st.session_state.model_llm
    )
    
    # Token statistics
    st.write("**Token Usage:**")
    st.write(f"Input Tokens: {st.session_state.token_tracker.total_input_tokens:,}")
    st.write(f"Output Tokens: {st.session_state.token_tracker.total_output_tokens:,}")
    st.write(f"Total Tokens: {st.session_state.token_tracker.total_input_tokens + st.session_state.token_tracker.total_output_tokens:,}")
    
    # Character statistics
    st.write("**Character Usage:**")
    st.write(f"Input Characters: {st.session_state.token_tracker.total_input_chars:,}")
    st.write(f"Output Characters: {st.session_state.token_tracker.total_output_chars:,}")
    st.write(f"Total Characters: {st.session_state.token_tracker.total_input_chars + st.session_state.token_tracker.total_output_chars:,}")
    
    # Cost statistics
    st.write("**Costs (USD):**")
    st.write(f"Input Cost: ${input_cost:.4f}")
    st.write(f"Output Cost: ${output_cost:.4f}")
    st.write(f"Total Cost: ${total_cost:.4f}")
    
    if st.button("Reset Counters"):
        st.session_state.token_tracker.reset_counts()
        st.rerun()

# Load the summarized content from a video analysis file
def load_video_content():
    """Load the video content from the markdown file"""
    try:
        with open("output/resumo.md", "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error loading video content: {str(e)}")
        return None

# Generate a response from an AI model based on a question and context
def get_ai_response(question: str, context: str, api_key: str, model: str) -> str:
    """Get AI response for a question about the video content"""
    try:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers questions about video content.
            Use the following context to answer the question. If you cannot answer
            the question based on the context, say so.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        
        formatted_prompt = prompt.format(
            context=context,
            question=question
        )
        
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            streaming=True
        )
        
        response = ""
        for chunk in llm.stream(formatted_prompt):
            content = chunk.content
            response += content
            yield content
        
        # Track tokens for this interaction
        st.session_state.token_tracker.add_interaction(
            formatted_prompt,
            response,
            model
        )
        
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        yield f"Error: {str(e)}"

# Clean the output directory while optionally keeping specific files
def clear_output_directory(keep_content=False):
    """
    Clean output directory while keeping specific files
    Args:
        keep_content (bool): If True, keeps files needed for Q&A
    """
    logger.info("Cleaning output directory...")
    output_dir = Path("output")
    files_to_keep = ['workflow.png']
    
    if keep_content:
        files_to_keep.extend(['resumo.md', 'resumo.html'])
    
    try:
        for file in output_dir.iterdir():
            if file.is_file():
                if file.name not in files_to_keep:
                    os.remove(file)
        logger.info("Output directory cleaned successfully")
    except Exception as e:
        logger.error(f"Error cleaning output directory: {str(e)}")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Video Analysis", "Q&A Assistant", "Contact Me"])

# Video Analysis Tab
with tab1:
    url = st.text_input("Video URL")

    if st.button("Analyze"):
        logger.info("Analysis button clicked")
        
        if not st.session_state.get("OPENAI_API_KEY"):
            logger.warning("No API key provided")
            st.error("Please enter your OpenAI API Key.")
        elif not url:
            logger.warning("No URL provided")
            st.warning("Please enter a YouTube URL.")
        elif not is_valid_youtube_url(url):
            logger.warning("Invalid YouTube URL")
            st.error("Please enter a valid YouTube video URL. Example: https://www.youtube.com/watch?v=xxxxx")
        else:
            logger.info(f"Starting analysis for URL: {url}")
            with st.spinner('Analyzing video...'):
                try:
                    if st.session_state.video_content is not None:
                        st.session_state.video_content = None
                        st.session_state.chat_history = []
                        clear_output_directory(keep_content=False)
                        
                        # Clear vector store when starting new video analysis
                        if st.session_state.vector_store:
                            st.session_state.vector_store.clear_vector_store()
                    
                    # Reset token counter before new analysis
                    st.session_state.token_tracker.reset_counts()
                    
                    logger.info("Initiating video processing...")
                    resultado = app1.process_video(
                        url,
                        api_key=st.session_state.OPENAI_API_KEY,
                        model_llm=st.session_state["model_llm"],
                        token_tracker=st.session_state.token_tracker
                    )
                    
                    if resultado["error"]:
                        logger.error(f"Processing error: {resultado['error']}")
                        st.error(f"Error: {resultado['error']}")
                        st.session_state.show_analysis = False
                    else:
                        logger.info("Video analysis completed successfully")
                        st.success("Analysis completed!")
                        
                        st.session_state.video_content = load_video_content()
                        st.session_state.show_analysis = True
                        st.rerun()
                        
                except Exception as e:
                    logger.error(f"Unexpected error during analysis: {str(e)}")
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state.show_analysis = False

    # Display the video content analysis if available
    if st.session_state.show_analysis and st.session_state.video_content:
        st.markdown(st.session_state.video_content)
        
        html_file_path = "output/resumo.html"
        if os.path.exists(html_file_path):
            logger.info("Loading HTML content for download...")
            try:
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                if st.download_button(
                    label="Download HTML",
                    data=html_content,
                    file_name="resumo.html",
                    mime="text/html"
                ):
                    logger.info("Download initiated")
                    if not st.session_state.video_content:
                        clear_output_directory(keep_content=False)
                    else:
                        clear_output_directory(keep_content=True)
                logger.info("HTML download button created")
            except Exception as e:
                logger.error(f"Error preparing HTML download: {str(e)}")
        
        # Display a workflow diagram if available
        if os.path.exists("output/workflow.png"):
            logger.info("Displaying workflow diagram...")
            st.image("output/workflow.png")

    # Load and display README content
    logger.info("Attempting to load README.md...")
    readme_path = Path("README.md")
    if readme_path.is_file():
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            st.markdown(readme_content)
            logger.info("README.md loaded and displayed")
        except Exception as e:
            logger.error(f"Error reading README.md: {str(e)}")
            st.warning("Error loading README.md file.")
    else:
        logger.warning("README.md file not found")
        st.warning("README.md file not found.")

# Q&A Assistant Tab
with tab2:
    st.header("Ask About the Video")
    
    if st.session_state.video_content is None:
        st.warning("Please analyze a video first in the Video Analysis tab.")
    elif not st.session_state.get("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    else:
        # Initialize vector store if not already done or if API key changed
        current_api_key = st.session_state.get("OPENAI_API_KEY")
        if (st.session_state.vector_store is None or 
            getattr(st.session_state.vector_store, 'api_key', None) != current_api_key):
            vector_store = VectorStoreManager(api_key=current_api_key)
            try:
                vector_store.clear_vector_store()
                logger.info("Vector store cleared on initialization/API key change")
            except Exception as e:
                logger.error(f"Error clearing vector store: {str(e)}")
            st.session_state.vector_store = vector_store
        
        question = st.text_input("Ask a question about the video content:")
        
        # Botões lado a lado
        col1, col2 = st.columns([4, 1])
        
        with col1:
            get_answer = st.button("Get Answer", use_container_width=True)
        
        with col2:
            if st.button("Clear Cache", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear_vector_store()
                    st.success("Vector store cache cleared!")
        
        if get_answer:
            if not question:
                st.warning("Please enter a question.")
            else:
                answer_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("Generating answer..."):
                    # Try to get answer from cache first
                    cached_result = st.session_state.vector_store.find_similar_question(question)
                    
                    if cached_result:
                        original_question, cached_answer, similarity_score = cached_result
                        full_response = cached_answer
                        answer_placeholder.markdown(full_response)
                        
                        # Show cache info
                        st.info(f"""Answer retrieved from cache!
                        Similar question: "{original_question}"
                        Similarity score: {similarity_score:.2%}""")
                    else:
                        # If not in cache, generate new answer
                        for content in get_ai_response(
                            question,
                            st.session_state.video_content,
                            st.session_state.OPENAI_API_KEY,
                            st.session_state.model_llm
                        ):
                            full_response += content
                            answer_placeholder.markdown(full_response + "▌")
                        
                        answer_placeholder.markdown(full_response)
                        
                        # Add to vector store
                        st.session_state.vector_store.add_qa_pair(question, full_response)
                        logger.info("New Q&A pair added to vector store")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": full_response,
                        "from_cache": bool(cached_result)
                    })
                    
                    # Force a page refresh to update statistics
                    st.rerun()
        
        # Display chat history if available
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in reversed(st.session_state.chat_history):
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer']}")
                if chat.get('from_cache'):
                    st.info("(Retrieved from cache)")
                st.markdown("---")

# Contact Me Tab
with tab3:
    st.subheader("Autor: Rafael G. Fernandes")
    st.subheader("Criado em: 31/10/2024")
    st.subheader("Clique no ícone abaixo para ver meu perfil no LinkedIn:")
    
    linkedin_url = "https://www.linkedin.com/in/rafael-g-fernandes/"
    st.markdown(
        f"""
        <a href="{linkedin_url}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn" width="40"/>
        </a>
        """,
        unsafe_allow_html=True
    )

# Footer section for additional information or copyright
st.markdown(
    "<hr style='margin-top: 40px;'><center>© 2024 RGF. All rights reserved.</center>",
    unsafe_allow_html=True
)

logger.info("Streamlit application initialization completed")
