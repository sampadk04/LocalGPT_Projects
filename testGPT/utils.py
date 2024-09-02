import os, shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# constants
from constants import SOURCE_DIR, CHROMA_DIR, chunk_size, chunk_overlap
from constants import PROMPT_TEMPLATE, CONTEXT_QUERY_TEMPLATE

# load the embedding function
def get_embedding_function():
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="gemma2:2b")
    return embeddings

# load the documents
def load_documents():
    document_loader = PyPDFDirectoryLoader(SOURCE_DIR)
    return document_loader.load()

# chunk the documents
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents(documents)


# store the chunk_ids to avoid document repeatition
def calculate_chunk_ids(chunks: list[Document]):
    # this creates IDS like "pdf_name.pdf:6:2"
    # Page Source : Page Number: Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if page ID is same as the last, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        #  add it to the page metadata
        chunk.metadata["id"] = chunk_id
    
    return chunks


# clear the database
def clear_database():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)



# update the chromadb
def add_to_chroma(chunks: list[Document]):
    # load the existing database
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_function()
    )

    # calculate the page ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    # add or update the documents
    existing_items = db.get(include=[]) # ids are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # only add documents that do not exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks) > 0:
        print(f"Adding {len(new_chunks)} new documents to the DB")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("No new documents to add to the DB")

def query_rag(query_text: str):
    # prepare the db
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_score(query=query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="gemma2:2b")
    response_text = model.invoke(prompt)

    # formatted_response_text = f"RESPONSE:\n\n {response_text}\n\nSOURCES: {sources}"
    # print(formatted_response_text)

    return response_text, sources


def retrieval_qa_pipeline(use_history=False):
    # prepare the db
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    )

    chat_history = ""
    n_questions = 0

    while True:
        cli_prompt = "Enter a query."
        query_text = input(f"\nModel: {cli_prompt}\nUser: ")

        n_questions += 1

        if query_text.lower() == "exit":
            break

        if use_history and n_questions < 10: # max no. of questions
            # get the contextualized prompt from history
            context_query_template = ChatPromptTemplate.from_template(CONTEXT_QUERY_TEMPLATE)

            context_query_prompt = context_query_template.format(chat_history=chat_history, question=query_text)
            # print(context_query_prompt)

            model = Ollama(model="gemma2:2b")
            context_query = model.invoke(context_query_prompt)

            model_response, sources = query_rag(context_query)

            # update chat history
            chat_history += f"\nModel: {cli_prompt}\n\nUser: {query_text}\nModel: {model_response}"

            # print the response
            # print(f"\n\nQuestion: {query_text}")
            print(f"\nModel: {model_response}")
            print(f"\nSources: {sources}\n\n")
        
        else:
            model_response, sources = query_rag(query_text)

            # print the response
            # print(f"\n\nQuestion: {query_text}")
            print(f"\nModel: {model_response}")
            print(f"\nSources: {sources}\n\n")
    
    return chat_history



# if __name__ == "__main__":
#     embeddings = get_embedding_function()
#     print(embeddings.embed_query("hello world"))