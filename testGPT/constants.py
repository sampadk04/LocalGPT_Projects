SOURCE_DIR = 'DOCS'
CHROMA_DIR = '.chroma_cache'

# chunking parameters
chunk_size = 1024
chunk_overlap = 128


# prompts source: LangChain Docs: https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.

CONTEXT:

{context}

___

TASK:

Answer the question based on the above context: {question}
"""

CONTEXT_QUERY_TEMPLATE = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

CHAT HISTORY:

{chat_history}

___

USER QUESTION:

{question}

"""