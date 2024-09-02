import argparse

from utils import clear_database, load_documents, split_documents, add_to_chroma

def main():
    # check if the database should be cleared (using the --clear flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("Resetting the database.")
        clear_database()
    
    # create (or update) the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()