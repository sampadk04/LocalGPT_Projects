# MyLocalGPT - Project Report

## Project Overview

MyLocalGPT is a privacy-focused document querying system that allows users to interact with their personal documents using natural language. The system runs entirely on local hardware, ensuring complete data privacy while leveraging state-of-the-art language models for intelligent question answering.

**Key Objective:** Enable users to query their document collections conversationally without compromising privacy by sending data to external services.

## Methodology

### Approach

The project implements a Retrieval-Augmented Generation (RAG) architecture, which combines:
1. **Information Retrieval**: Efficiently finding relevant document sections
2. **Language Generation**: Producing coherent, context-aware responses

This approach addresses the fundamental limitation of language models—their inability to access external, user-specific knowledge—by augmenting them with a retrieval mechanism.

### Core Components

#### 1. Document Processing Pipeline
- Documents are loaded from local storage (PDF, TXT, CSV formats)
- Text is extracted and split into semantically meaningful chunks
- Chunk size is optimized for both context preservation and retrieval accuracy
- Overlap between chunks ensures continuity of information

#### 2. Embedding System
- Text chunks are converted to high-dimensional vector representations
- Embeddings capture semantic meaning, allowing similarity-based matching
- Multiple embedding model options accommodate different hardware constraints
- Vectors are normalized for efficient distance computation

#### 3. Vector Storage
- ChromaDB provides persistent storage for embeddings
- Enables fast similarity search across large document collections
- Maintains metadata linking vectors back to source documents
- Supports incremental updates as new documents are added

#### 4. Retrieval Mechanism
- User queries are embedded using the same model as documents
- Vector similarity search identifies most relevant document chunks
- Retrieved chunks provide context for answer generation
- Source attribution is preserved for transparency

#### 5. Language Model Integration
- Llama2-based models generate human-like responses
- Retrieved context is injected into the generation prompt
- Model is constrained to answer based on provided context
- Conversation history enables multi-turn interactions

## System Flow

### Initial Setup Flow

```
User Documents (PDF/TXT/CSV)
    ↓
Document Loader
    ↓
Text Chunking (1000 chars, 200 overlap)
    ↓
Embedding Model (HuggingFace Instructor)
    ↓
Vector Database (ChromaDB)
    ↓
Ready for Queries
```

### Query Processing Flow

```
User Question
    ↓
Query Embedding
    ↓
Similarity Search (Vector DB)
    ↓
Retrieve Top-K Relevant Chunks
    ↓
Construct Prompt (Question + Context + History)
    ↓
Language Model (Llama2)
    ↓
Generated Answer + Source Documents
    ↓
Display to User
```

### Conversation Flow

1. **User Input**: User asks a question in natural language
2. **Context Retrieval**: System finds relevant document sections
3. **Prompt Construction**: Question, retrieved context, and conversation history are combined
4. **Generation**: Language model produces an answer
5. **Source Tracking**: System displays which documents were used
6. **Memory Update**: Interaction is stored for context in follow-up questions
7. **Loop**: Process repeats for next question

## Key Functionalities

### 1. Multi-Format Document Support
- **PDF Processing**: Extracts text from academic papers, reports, and books
- **Plain Text**: Handles notes, documentation, and code files
- **CSV Processing**: Queries structured data and spreadsheets
- **Extensibility**: Architecture supports adding new loaders for additional formats

### 2. Privacy-Preserving Processing
- **Local Execution**: All models run on user's hardware
- **No Data Transmission**: Documents never leave the local machine
- **Offline Capability**: Works without internet (after initial model download)
- **Control**: Users maintain complete ownership of their data

### 3. Contextual Question Answering
- **Source-Grounded Responses**: Answers are based on actual document content
- **Citation Tracking**: Every answer shows which documents were referenced
- **Hallucination Mitigation**: Model is instructed to admit uncertainty rather than fabricate
- **Relevance Scoring**: Retrieval system ranks document chunks by relevance

### 4. Conversation Memory
- **Multi-Turn Dialogue**: System remembers previous questions and answers
- **Context Awareness**: Follow-up questions can reference earlier topics
- **Natural Interaction**: Users can ask clarifying questions without repeating context
- **Session Management**: Conversations can span multiple related queries

### 5. Flexible Model Selection
- **Hardware Adaptation**: Choose models based on available GPU/CPU resources
- **Quality vs. Speed**: Trade-off between response quality and generation time
- **Quantization Options**: Compressed models for resource-constrained systems
- **Model Swapping**: Easy configuration changes without code modification

### 6. Configurable Retrieval
- **Adjustable Chunk Size**: Balance between context granularity and coherence
- **Embedding Model Selection**: Different models for accuracy vs. speed
- **Retrieval Parameters**: Control how many document chunks are used
- **Device Configuration**: Run on CPU, NVIDIA CUDA, or Apple Silicon (MPS)

## Project Workflow

### Phase 1: Document Ingestion
Users run the ingestion script once to process their document collection. The system:
- Scans the source directory for supported files
- Extracts and chunks text content
- Generates embeddings for all chunks
- Stores vectors in the local database
- Reports statistics (documents processed, chunks created)

### Phase 2: Interactive Querying
Users launch the query interface for conversational interaction. The system:
- Loads the pre-computed vector database
- Initializes the language model
- Enters an interactive loop waiting for questions
- Processes each query through the RAG pipeline
- Displays answers with source attributions
- Maintains conversation history for context

### Phase 3: Iterative Refinement
As users interact with the system, they can:
- Add new documents by re-running ingestion
- Adjust model parameters for better performance
- Switch between different language models
- Fine-tune retrieval settings
- Export useful question-answer pairs

## Technical Design Decisions

### Why RAG over Fine-Tuning?
- **Dynamic Content**: Documents can be updated without retraining
- **Resource Efficiency**: No need for expensive model training
- **Transparency**: Sources are explicitly shown, not internalized
- **Flexibility**: Same model works across different document collections

### Why Local Execution?
- **Privacy**: Sensitive documents remain on user's device
- **Cost**: No API fees for cloud services
- **Latency**: No network round-trips for queries
- **Availability**: Works offline after initial setup

### Why Chunk-Based Retrieval?
- **Precision**: Relevant sections found without irrelevant context
- **Scalability**: Efficient search across large document sets
- **Context Management**: Fits within model's token limits
- **Granularity**: Users see specific passages, not whole documents

## Use Cases

1. **Research Assistance**: Query academic paper collections
2. **Legal Document Analysis**: Search through contracts and case files
3. **Knowledge Management**: Personal wiki or note-taking systems
4. **Code Documentation**: Query technical documentation locally
5. **Book Analysis**: Interact with personal library collections
6. **Meeting Notes**: Search historical meeting transcripts
7. **Policy Review**: Navigate organizational policy documents

## Limitations and Considerations

1. **Context Window**: Very long documents may require multiple retrievals
2. **Computational Requirements**: Large models need significant hardware
3. **Initial Setup Time**: First-time model downloads can be lengthy
4. **Accuracy Dependency**: Quality depends on retrieval precision
5. **Structured Data**: Works best with text; images and tables have limited support

## Conclusion

MyLocalGPT demonstrates the viability of privacy-preserving document querying using open-source technologies. By combining retrieval mechanisms with language generation, the system provides accurate, source-grounded answers while maintaining complete user privacy. The modular architecture allows adaptation to various hardware configurations and use cases, making advanced AI accessible for personal document management.

---

**Project Type:** Natural Language Processing, Information Retrieval  
**Technologies:** Python, LangChain, Llama2, ChromaDB, HuggingFace  
**Architecture:** Retrieval-Augmented Generation (RAG)  
**Deployment:** Local/On-Premise
