# AWS Bedrock with Claude, Titan & Stable Diffusion

This repository demonstrates how to build **end-to-end Generative AI applications** on **AWS Bedrock** by integrating multiple foundation models into a unified, modular workflow. The project covers three primary capabilities:

-   **Conversational AI with Claude**  
    Uses Anthropic’s **Claude v3** models via BedrockChat for question answering, summarization, and natural conversation, with carefully designed prompt templates to enforce detailed, grounded responses.
    
-   **Semantic Search with Titan Embeddings**  
    Leverages Amazon’s **Titan Embeddings** to transform PDF documents into dense vector representations, stored efficiently using **FAISS**. A retrieval-augmented generation (RAG) pipeline powers semantic search over local knowledge bases, making answers more context-aware and reliable.
    
-   **Creative Generation with Stable Diffusion**  
    Implements **Stability AI’s Stable Diffusion** models for text-to-image generation, showcasing how Bedrock can support both productivity-focused and creative AI workloads within a single environment.
    

The codebase is organized into **modular Python scripts** (`claude.py`, `amazon_titan.py`, `stableDiffusion.py`, `app.py`) to separate model logic from application orchestration. A **Streamlit app** provides a simple UI for interactive experimentation, where users can upload PDFs, update vector indexes, and query documents using either Claude or Titan.
<br>

Technically, the project highlights:
-   **Bedrock + boto3 integration** for invoking models programmatically.
  -   **Prompt engineering** with LangChain’s `PromptTemplate` for consistency and explainability.
-   **RAG pipeline design** with recursive text splitting, embeddings, and vector search.
-   **Secure persistence** of FAISS indexes with controlled deserialization.
-   **Practical deployment patterns** for real-world scenarios such as chatbots, document assistants, and AI-assisted creative design.
 

This repository serves as a **starter kit for developers** who want to explore **multi-modal AI on AWS Bedrock** — combining **chat, retrieval, and image generation** in one cohesive framework.
<br>

Special thanks to [**Krish Naik**](https://github.com/krishnaik06), whose tutorials inspired the Bedrock setup and CLI workflows.
