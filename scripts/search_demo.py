# scripts/search_demo.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
from config import DEFAULT_TOP_K
from lit_agent.agents.retrieval_agent import RetrievalAgent

def main():
    agent = RetrievalAgent.from_config()
    while True:
        query = input("\nEnter your search query. Type q to exit:\n> ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            print("Exiting search demo.")
            break
        if not query:
            print("Empty query. Please enter a valid search query.")
            continue
        results = agent.retrieve(
            query=query,
            top_k=DEFAULT_TOP_K,
            normalize=True,
        )
        print(agent.format_query_plan())
        print(agent.format_results(results))

if __name__ == "__main__":
    main()