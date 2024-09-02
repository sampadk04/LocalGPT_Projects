import argparse
from utils import retrieval_qa_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_history", action="store_true", help="Use chat history.")
    args = parser.parse_args()

    if args.use_history:
        print("Using Chat History")
        retrieval_qa_pipeline(use_history=True)
    else:
        print("Not Using Chat History")
        retrieval_qa_pipeline(use_history=False)

if __name__ == "__main__":
    main()