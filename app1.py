# This script processes YouTube videos by downloading, transcribing, and analyzing content for insights.
# It uses a structured workflow to perform various tasks such as downloading the video, transcribing
# audio, generating summaries, analyzing sentiment, relevance, depth, and providing a final evaluation score.
# The results are saved as Markdown and HTML reports with a workflow diagram.

# Main Functionalities:
# Agent1: **Download and Metadata Extraction**: Downloads audio from a YouTube video URL and extracts metadata (title, 
#    uploader, views, etc.) using `yt-dlp`.
# Agent2: **Audio Segmentation and Transcription**: Splits the audio file into segments, transcribes each segment 
#    using OpenAI's Whisper model, and combines the results into a full transcription.
# Agent3: **Content Summarization and Validation**: Generates an initial summary of the transcription and validates 
#    it for accuracy and completeness.
# Agent4: **Analyses**:
#    - **Sentiment Analysis**: Evaluates the overall tone and emotional content.
#    - **Relevance Analysis**: Assesses the relevance based on timeliness, audience, and practical application.
#    - **Depth Analysis**: Examines the technical level, examples provided, and explanation quality.
# Agent5: **Final Evaluation**: Provides a score and recommendation based on previous analyses, suggesting the content's
#    usefulness.
# Agent6: **Summary Export**: Compiles and exports a final structured summary with metadata and analysis sections in both 
#    Markdown and HTML formats.
# Agent7: **Workflow Management and Visualization**: Configures and compiles a workflow of steps with entry points, edges,
#    and generates a visual workflow diagram for clarity.

# This script uses structured logging to track each step and handles errors gracefully, updating the state
# accordingly. It maintains a flexible workflow for processing and analyzing YouTube video content efficiently.


from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yt_dlp
import os
from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod
from openai import OpenAI
import markdown
from pydub import AudioSegment
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

output_dir = './output'
model_transcript = "whisper-1"


try:
    logger.info("Checking/Creating output directory...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory confirmed at: {output_dir}")
except Exception as e:
    logger.error(f"Error creating output directory: {str(e)}")
    raise Exception(f"Failed to create output directory: {str(e)}")


class State(TypedDict):
    url: str
    audio_path: str
    transcription: str
    initial_summary: str
    validation_feedback: str
    final_summary: str
    status: str
    error: str
    metadata: dict
    sentiment_analysis: str
    relevance_analysis: str
    depth_analysis: str
    final_evaluation: str
    api_key: str
    model_llm: str
    token_tracker: object

# Agent to download video and extract metadata
def download_agent(state: State) -> State:
    logger.info("Starting video download and metadata extraction...")
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Youtube-dl download settings
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # Use default name for audio file
            'outtmpl': os.path.join(output_dir, 'downloaded_audio.%(ext)s'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Fetching video information...")
            info = ydl.extract_info(state["url"], download=True)
            audio_path = os.path.join(output_dir, "downloaded_audio.mp3")
            
            logger.info("Extracting video metadata...")
            metadata = {
                "title": info.get("title", "Not available"),
                "channel": info.get("uploader", "Not available"),
                "upload_date": info.get("upload_date", "Not available"),
                "view_count": info.get("view_count", "Not available"),
                "categories": info.get("categories", ["Not available"])[0],
                "tags": info.get("hashtags", ["Not available"])
            }

            logger.info("Video download completed successfully")
            return state | {
                "audio_path": audio_path,
                "metadata": metadata,
                "status": "downloaded"
            }
    except Exception as e:
        logger.error(f"Download error occurred: {str(e)}")
        return state | {"error": f"Download error: {str(e)}", "status": "error"}

def split_audio(audio_path: str, segment_length_ms: int = 300000) -> list:
    """
    Split audio file into smaller segments.
    Args:
        audio_path: Path to the audio file
        segment_length_ms: Length of each segment in milliseconds (default: 5 minutes)
    Returns:
        List of AudioSegment objects
    """
    logger.info("Starting audio splitting process...")
    try:
        audio = AudioSegment.from_file(audio_path)
        segments = []
        
        # Split audio into segments
        total_segments = len(audio) // segment_length_ms + (1 if len(audio) % segment_length_ms else 0)
        logger.info(f"Audio will be split into {total_segments} segments")
        
        for i in range(0, len(audio), segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            segments.append(segment)
            logger.debug(f"Segment {len(segments)} created")
            
        logger.info("Audio splitting completed")
        return segments
    except Exception as e:
        logger.error(f"Error in audio splitting: {str(e)}")
        raise Exception(f"Error splitting audio: {str(e)}")

# Agent to transcribe audio using OpenAI API
def transcription_agent(state: State) -> State:
    """Transcript the audio using Whisper OpenAI API."""
    logger.info("Starting audio transcription process...")
    try:
        # Check if the audio file exists
        if not os.path.exists(state["audio_path"]):
            logger.error("Audio file not found")
            return state | {"error": "Audio file not found for transcription", "status": "error"}

        client = OpenAI(api_key=state["api_key"])
        
        # Split audio into segments
        try:
            logger.info("Starting audio segmentation...")
            segments = split_audio(state["audio_path"])
        except Exception as e:
            logger.error(f"Audio segmentation failed: {str(e)}")
            return state | {"error": f"Error splitting audio: {str(e)}", "status": "error"}
        
        transcriptions = []
        
        # Process each segment
        logger.info(f"Starting transcription of {len(segments)} segments...")
        for idx, segment in enumerate(segments):
            try:
                logger.info(f"Processing segment {idx + 1}/{len(segments)}")
                # Save segment temporarily
                segment_path = os.path.join(output_dir, f"temp_segment_{idx}.mp3")
                segment.export(segment_path, format="mp3", parameters=["-ac", "1"])  # Convert to mono
                
                # Transcribe segment
                logger.debug(f"Transcribing segment {idx + 1}")
                with open(segment_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model=model_transcript,
                        file=audio_file,
                        language="pt"
                    )
                    transcriptions.append(transcript.text)
                
                # Clean up temporary segment file
                os.remove(segment_path)
                logger.debug(f"Segment {idx + 1} completed and cleaned up")
                
            except Exception as e:
                logger.error(f"Error processing segment {idx}: {str(e)}")
                # Clean up on error
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                return state | {"error": f"Error processing segment {idx}: {str(e)}", "status": "error"}
        
        # Join all transcriptions
        logger.info("Combining all transcriptions...")
        full_transcription = " ".join(transcriptions)
        
        # Save full transcription
        try:
            logger.info("Saving complete transcription...")
            with open(f"{output_dir}/transcription.txt", "w", encoding="utf-8") as f:
                f.write(full_transcription)
        except Exception as e:
            logger.error(f"Error saving transcription: {str(e)}")
            return state | {"error": f"Error saving transcription: {str(e)}", "status": "error"}

        logger.info("Transcription process completed successfully")
        return state | {"transcription": full_transcription, "status": "transcribed"}
    except Exception as e:
        logger.error(f"Transcription process failed: {str(e)}")
        return state | {"error": f"Transcription error: {str(e)}", "status": "error"}

# Agent to generate initial summary of the transcription
def initial_summary_agent(state: State) -> State:
    """Generates an initial summary of the transcription."""
    logger.info("Starting initial summary generation...")
    try:
        prompt = ChatPromptTemplate.from_template(
            "Create a detailed summary of the following transcription, focusing on key points and important insights: {text}"
        )
        formatted_prompt = prompt.format(text=state["transcription"])
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Initial summary generated successfully")
        return state | {"initial_summary": response, "status": "summarized"}
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return state | {"error": f"Summary error: {str(e)}", "status": "error"}

# Agent to validate initial summary accuracy based on transcription
def validation_agent(state: State) -> State:
    """Reviews the initial summary for accuracy and completeness based on the transcription."""
    logger.info("Starting summary validation process...")
    try:
        prompt = ChatPromptTemplate.from_template(
            "Review this summary and provide feedback on its accuracy and completeness. Suggest improvements if necessary: {summary}\n\nOriginal transcription: {transcription}"
        )
        formatted_prompt = prompt.format(
            summary=state["initial_summary"],
            transcription=state["transcription"]
        )
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Summary validation completed successfully")
        return state | {"validation_feedback": response, "status": "validated"}
    except Exception as e:
        logger.error(f"Validation process failed: {str(e)}")
        return state | {"error": f"Validation error: {str(e)}", "status": "error"}

# Agent to analyze the sentiment of the transcription content
def sentiment_analysis_agent(state: State) -> State:
    """Analyzes the sentiment of the transcription content."""
    logger.info("Starting sentiment analysis...")
    try:
        prompt = ChatPromptTemplate.from_template(
            """Analyze the sentiment of the text on a scale from Very Negative to Very Positive.
            Focus on:
            1. General tone (1 line)
            2. Justification (max 3 points)
            3. Objective conclusion with final rating

            Text: {text}"""
        )
        formatted_prompt = prompt.format(text=state["transcription"])
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Sentiment analysis completed successfully")
        return state | {"sentiment_analysis": response, "status": "sentiment_analyzed"}
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return state | {"error": f"Sentiment analysis error: {str(e)}", "status": "error"}

# Agent to assess the relevance of the content
def relevance_analysis_agent(state: State) -> State:
    """Assesses the relevance of the content."""
    logger.info("Starting relevance analysis...")
    try:
        prompt = ChatPromptTemplate.from_template(
            """Objectively assess the content relevance:
            1. Timeliness (1 line)
            2. Target audience and importance (1 line)
            3. Practical application (1 line)
            4. Objective conclusion about general relevance

            Text: {text}"""
        )
        formatted_prompt = prompt.format(text=state["transcription"])
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Relevance analysis completed successfully")
        return state | {"relevance_analysis": response, "status": "relevance_analyzed"}
    except Exception as e:
        logger.error(f"Relevance analysis failed: {str(e)}")
        return state | {"error": f"Relevance analysis error: {str(e)}", "status": "error"}

# Agent to analyze technical depth of the content
def depth_analysis_agent(state: State) -> State:
    """Analyzes the technical depth of the content."""
    logger.info("Starting technical depth analysis...")
    try:
        prompt = ChatPromptTemplate.from_template(
            """Analyze the technical depth of the content:
            1. Technical level (basic/intermediate/advanced) with brief justification
            2. Quality of explanations (1 line)
            3. Practical examples (yes/no and relevance)
            4. Conclusion about overall depth

            Text: {text}"""
        )
        formatted_prompt = prompt.format(text=state["transcription"])
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Technical depth analysis completed successfully")
        return state | {"depth_analysis": response, "status": "depth_analyzed"}
    except Exception as e:
        logger.error(f"Technical depth analysis failed: {str(e)}")
        return state | {"error": f"Depth analysis error: {str(e)}", "status": "error"}

# Agent to provide a final evaluation score based on previous analyses
def final_score_agent(state: State) -> State:
    """Provides a final score for the content with a recommendation on its usefulness."""
    logger.info("Starting final evaluation...")
    try:
        prompt = ChatPromptTemplate.from_template(
            """Based on previous analyses, provide:
            1. Score (0-10)
            2. Direct recommendation (Is it worth it? Yes/No)
            3. Objective justification in up to 3 main points

            Analyses:
            Sentiment: {sentiment}
            Relevance: {relevance}
            Depth: {depth}"""
        )
        formatted_prompt = prompt.format(
            sentiment=state["sentiment_analysis"],
            relevance=state["relevance_analysis"],
            depth=state["depth_analysis"]
        )
        
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        response = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                response,
                state["model_llm"]
            )
        
        logger.info("Final evaluation completed successfully")
        return state | {"final_evaluation": response, "status": "evaluated"}
    except Exception as e:
        logger.error(f"Final evaluation failed: {str(e)}")
        return state | {"error": f"Final evaluation error: {str(e)}", "status": "error"}

# Convert Markdown text to HTML
def convert_md_to_html(md_text: str) -> str:
    logger.debug("Converting markdown to HTML")
    return markdown.markdown(md_text, extensions=['extra', 'smarty'])

# Agent to create final summary and export in Markdown and HTML formats
def final_summary_agent(state: State) -> State:
    """Compiles a final summary with metadata, analysis, and exports to Markdown and HTML."""
    logger.info("Starting final summary compilation...")
    try:
        if not state.get("transcription"):
            logger.error("No transcription found")
            return state | {"error": "No transcription found", "status": "error"}

        # Format metadata
        upload_date = state["metadata"]["upload_date"]
        formatted_date = f"{upload_date[6:8]}/{upload_date[4:6]}/{upload_date[0:4]}"
        tags_str = ", ".join(state["metadata"]["tags"]) if state["metadata"]["tags"] else "Not available"
        
        logger.info("Preparing metadata and analysis sections...")
        metadata_text = f"""# {state["metadata"]["title"]}

## Video Metadata
- **Author:** {state["metadata"]["channel"]}
- **Publication Date:** {formatted_date}
- **Views:** {state["metadata"]["view_count"]:,}
- **Category:** {state["metadata"]["categories"]}
- **Tags:** {tags_str}

## Analyses
### Sentiment Analysis
{state["sentiment_analysis"]}

### Relevance Analysis
{state["relevance_analysis"]}

### Depth Analysis
{state["depth_analysis"]}

### Final Evaluation
{state["final_evaluation"]}

## Summary
### Structured Summary

"""

        prompt = ChatPromptTemplate.from_template(
            """Based on the transcription and validation feedback, create a final structured summary:

            Transcription: {transcription}
            
            Feedback: {validation_feedback}
            
            Please structure the summary with:
            **1. Main themes and concepts covered (1 line):**
            **2. Key points discussed (up to 3 points):**
            **3. Mentioned tools or technologies (up to 3 points):**
            **4. Important conclusions (1 line):**"""
        )

        formatted_prompt = prompt.format(
            transcription=state["transcription"],
            validation_feedback=state["validation_feedback"]
        )

        logger.info("Generating final structured summary...")
        llm = ChatOpenAI(model=state["model_llm"], api_key=state["api_key"])
        final_summary = llm.invoke(formatted_prompt).content
        
        if state.get("token_tracker"):
            state["token_tracker"].add_interaction(
                formatted_prompt,
                final_summary,
                state["model_llm"]
            )
        
        complete_summary = metadata_text + final_summary
        
        logger.info("Saving results to files...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Markdown version
        logger.info("Saving Markdown file...")
        with open(f"{output_dir}/resumo.md", "w", encoding="utf-8") as f:
            f.write(complete_summary)

        # Save HTML version
        logger.info("Converting to HTML and saving...")
        html_content = convert_md_to_html(complete_summary)
        with open(f"{output_dir}/resumo.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info("Final summary process completed successfully")
        return state | {
            "final_summary": complete_summary, 
            "status": "completed",
            "html_content": html_content
        }
    except Exception as e:
        logger.error(f"Final summary process failed: {str(e)}")
        return state | {"error": f"Final summary error: {str(e)}", "status": "error"}

# Setup workflow
logger.info("Initializing workflow configuration...")
workflow = StateGraph(State)

# Add nodes to workflow
logger.info("Adding nodes to workflow...")
workflow.add_node("download", download_agent)
workflow.add_node("transcribe", transcription_agent)
workflow.add_node("summarize", initial_summary_agent)
workflow.add_node("validate", validation_agent)
workflow.add_node("sentiment", sentiment_analysis_agent)
workflow.add_node("relevance", relevance_analysis_agent)
workflow.add_node("depth", depth_analysis_agent)
workflow.add_node("score", final_score_agent)
workflow.add_node("finalize", final_summary_agent)

# Set entry point
workflow.set_entry_point("download")

# Define workflow edges
logger.info("Configuring workflow connections...")
edges = [
    ("download", "transcribe"),
    ("transcribe", "summarize"),
    ("summarize", "validate"),
    ("validate", "sentiment"),
    ("sentiment", "relevance"),
    ("relevance", "depth"),
    ("depth", "score"),
    ("score", "finalize"),
    ("finalize", END)
]

# Add edges to workflow
for start, end in edges:
    workflow.add_edge(start, end)

logger.info("Compiling workflow...")
app = workflow.compile()

logger.info("Generating workflow diagram...")
workflow_image = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)

with open(f"{output_dir}/workflow.png", "wb") as f:
    f.write(workflow_image)
logger.info("Workflow diagram saved successfully")

# Main function to process video
def process_video(url: str, api_key: str, model_llm: str, token_tracker=None) -> dict:
    """Main function to initiate the workflow with provided API key, model, and URL."""
    logger.info("Starting video processing workflow...")
    logger.info(f"Processing URL: {url}")
    logger.info(f"Using model: {model_llm}")
    
    # Initialize state
    initial_state = State(
        url=url,
        audio_path="",
        transcription="",
        initial_summary="",
        validation_feedback="",
        final_summary="",
        status="started",
        error="",
        metadata={},
        sentiment_analysis="",
        relevance_analysis="",
        depth_analysis="",
        final_evaluation="",
        api_key=api_key,
        model_llm=model_llm,
        token_tracker=token_tracker
    )
    
    try:
        logger.info("Executing workflow...")
        result = app.invoke(initial_state)
        
        if result.get("error"):
            logger.error(f"Workflow completed with error: {result['error']}")
        else:
            logger.info("Workflow completed successfully")
            
        return result
    except Exception as e:
        error_msg = f"Error in workflow execution: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "status": "error"
        }