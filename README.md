# YouTube Video Analysis with OpenAI and LangGraph

### Author
**Rafael G. Fernandes**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-g-fernandes/)

## Purpose of the Project
This project automates the analysis of YouTube videos by downloading audio, transcribing content, and applying various analyses like sentiment, relevance, and depth. It provides structured reports in Markdown and HTML formats, utilizing OpenAI's language models within a structured workflow.

## General Structure
The project is organized into a modular workflow with nine agents, each performing a specific task in the video analysis process. These agents are executed in a predefined sequence to ensure a smooth flow from data extraction to report generation:

- **Agent1: Download and Metadata Extraction**  
  Downloads the YouTube audio and extracts metadata (title, tags, uploader, etc.).

- **Agent2: Audio Segmentation and Transcription**  
  Splits the audio into segments and uses OpenAI’s Whisper model to transcribe each segment.

- **Agent3: Initial Summarization**  
  Generates an initial summary of the transcription, focusing on key points.

- **Agent4: Summary Validation**  
  Validates the initial summary for accuracy and completeness based on the transcription.

- **Agent5: Sentiment Analysis**  
  Analyzes the overall tone and sentiment of the transcribed content.

- **Agent6: Relevance Analysis**  
  Assesses the relevance of the content, considering timeliness, target audience, and practical applications.

- **Agent7: Depth Analysis**  
  Evaluates the technical depth of the content, including explanation quality and examples.

- **Agent8: Score**  
  Assigns a score based on prior analyses and provides a recommendation on the content’s usefulness.

- **Agent9: Final Summary Export**  
  Compiles a final report with metadata and analysis results, exporting it in both Markdown and HTML formats.

The workflow is managed by LangGraph’s `StateGraph`, which defines the sequence of agent execution and visually represents the workflow.

## Tab 1: Video Analysis
The **Video Analysis** tab enables users to input a YouTube video URL for processing. Key functionalities include:
- **Video URL Input**: Accepts the YouTube URL for analysis.
- **Analyze Button**: Starts the workflow, covering download, transcription, and content analysis.
- **Analysis Display**: Shows results of sentiment, relevance, and depth analyses alongside the full transcription.
- **Download Options**: Allows users to download the results as HTML and Markdown reports.
- **Workflow Visualization**: Displays a diagram of the workflow process for a clear understanding of each step.

## Tab 2: Q&A Assistant
The **Q&A Assistant** tab lets users ask questions about the analyzed video content. Key features include:
- **Ask a Question**: Users can input questions about specific content aspects.
- **Answer Generation**: Uses the video analysis data to generate concise answers.
- **Chat History**: Displays previous questions and answers, allowing users to track the session’s conversation history.

## 4 - Tab 3: Contact Me
The **Contact Me** tab displays the author's contact information. It includes:
- **Name**: Displays the author's name (e.g., "Rafael G. Fernandes").
- **LinkedIn Icon**: A clickable LinkedIn icon that links to the author's LinkedIn profile, allowing users to connect or learn more about the author.

---

This structure provides an organized, end-to-end solution for YouTube content analysis, combining automated processing with an interactive, user-friendly interface.
