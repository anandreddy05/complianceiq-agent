from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)


class QueryOptimizer:
    def __init__(self):
        print("Initializing Query Optimizer...")
        self.client = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=40,
        )

    def expand_query(self, user_query: str) -> str:
        system_prompt = (
            "You are a legal and regulatory data retrieval assistant. "
            "Your job is to take a user's natural language compliance query and generate "
            "a list of exact legal terms, regulatory synonyms, article references, and "
            "official terminology that would appear in compliance documents. "
            "DO NOT answer the question. ONLY output a space-separated list of keywords."
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query),
            ]
            response = self.client.invoke(messages)
            expanded_terms = response.content.strip()
            return f"{user_query} {expanded_terms}"

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return user_query
