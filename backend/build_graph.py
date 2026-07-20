"""Build the knowledge graph from the frozen corpus. Requires OPENAI_API_KEY."""

import os
import sys
import json
import re

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    import spacy
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    # Add parent paths
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.config import CORPUS_PATH, GRAPH_CACHE_PATH, RELATION_TYPES
    from core.corpus import load_corpus
    from core.graph import (
        canonicalise_relation, normalise_entity,
        save_graph, build_graph_from_corpus,
    )

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading corpus...")
    corpus, _ = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus)} chunks")

    print("Setting up KG extraction LLM...")
    kg_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    extraction_prompt = PromptTemplate.from_template("""You are extracting a regulatory knowledge graph from EU financial-services legislation.
Extract regulatory entities and directed relationships from the text.
For every relationship choose EXACTLY ONE label from:
{relation_types}

Return ONLY valid JSON:
{{"entities": ["entity"], "relationships": [["subject", "RELATION", "object"]]}}

Rules:
- Do not invent facts not explicit in the text.
- Prefer concise canonical entity names.
- Relationship subjects and objects must be included as entities.
- Use RELATED_TO only if no more specific label is justified.

Text:
{text}
""")

    def extract_elements(chunk_text):
        try:
            response = kg_llm.invoke(
                extraction_prompt.format(
                    relation_types=", ".join(RELATION_TYPES),
                    text=chunk_text,
                )
            )
            raw = re.sub(r"```json|```", "", response.content.strip()).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(match.group()) if match else {"entities": [], "relationships": []}

            entities = [str(x) for x in parsed.get("entities", [])]
            relationships = []
            for rel in parsed.get("relationships", []):
                if isinstance(rel, list) and len(rel) == 3:
                    relationships.append([rel[0], canonicalise_relation(rel[1]), rel[2]])
            return {"entities": entities, "relationships": relationships}
        except Exception as e:
            print(f"  Extraction error: {e}")
            return {"entities": [], "relationships": []}

    print(f"Building knowledge graph from {len(corpus)} chunks...")
    print("  This will make ~365 API calls to gpt-4o-mini.")
    G = build_graph_from_corpus(corpus, nlp, extract_elements)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    save_graph(G, GRAPH_CACHE_PATH)
    print(f"Graph saved to {GRAPH_CACHE_PATH}")


if __name__ == "__main__":
    main()
