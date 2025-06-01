import rag_core
import os
import sys # For sys.exit()

# Access global instances from rag_core if needed, e.g.
# llm = rag_core.llm
# embeddings = rag_core.embeddings
# SHARED_CONTEXT_BASE_DIR can be accessed via rag_core.SHARED_CONTEXT_BASE_DIR

def main_cli():
    current_vectorstore = None
    # current_docs_for_context should store all documents that form the current_vectorstore
    # This is important for saving the context correctly.
    current_docs_for_context = []

    print("Welcome to the RAG CLI Application!")
    print(f"Contexts will be saved/loaded from: {rag_core.SHARED_CONTEXT_BASE_DIR}")
    print("Available commands: search, query, load, save, reset, exit")

    while True:
        command = input("\n> Enter command: ").strip().lower()

        if command == "search":
            query = input("  Enter search query: ").strip()
            if not query:
                print("  Search query cannot be empty.")
                continue

            print(f"  Searching the web for: '{query}'...")
            scrape_response = rag_core.searxng_discover_and_scrape(query)

            if scrape_response.get("error"):
                print(f"  Error during web discovery/scraping: {scrape_response['error']}")
                continue

            newly_scraped_docs = scrape_response.get("results", [])
            if not newly_scraped_docs:
                print("  No new documents were scraped from the web for this query.")
                # If no new docs, still try to answer based on existing context if user query makes sense for it
                if current_vectorstore:
                    print(f"  Attempting to answer '{query}' using existing knowledge base...")
                    context_docs_for_llm = rag_core.retrieve_context(current_vectorstore, query)
                    if not context_docs_for_llm:
                        print("  Could not retrieve relevant context for your query from the existing documents.")
                    else:
                        llm_response = rag_core.generate_llm_answer(query, context_docs_for_llm, history=None)
                        print("\n--- Answer (from existing context) ---")
                        print(llm_response.get("answer", "No answer generated."))
                        print("\n--- Sources ---")
                        print(llm_response.get("sources_md", "No sources provided."))
                        print("------------------------------------")
                continue # Skip adding to vector store if no new docs

            print(f"  Processing {len(newly_scraped_docs)} newly scraped documents...")

            vs_after_op, status_msg_core = rag_core.get_or_create_vectorstore(
                processed_docs_as_dicts=newly_scraped_docs,
                existing_vectorstore=current_vectorstore
            )

            print(f"  {status_msg_core}")
            if vs_after_op:
                current_vectorstore = vs_after_op
                # Append newly scraped docs to the cumulative list for this session's context
                current_docs_for_context.extend(newly_scraped_docs)

                print(f"  Generating answer for '{query}' using updated knowledge base...")
                context_docs_for_llm = rag_core.retrieve_context(current_vectorstore, query)
                if not context_docs_for_llm:
                    print("  Could not retrieve relevant context for your query from the updated documents.")
                else:
                    llm_response = rag_core.generate_llm_answer(query, context_docs_for_llm, history=None)
                    print("\n--- Answer ---")
                    print(llm_response.get("answer", "No answer generated."))
                    print("\n--- Sources ---")
                    print(llm_response.get("sources_md", "No sources provided."))
                    print("--------------")
            # If vs_after_op is None, current_vectorstore is not updated, current_docs_for_context not updated with these new docs.

        elif command == "query":
            if not current_vectorstore:
                print("  No active knowledge base. Use 'search' or 'load' first.")
                continue
            follow_up_query = input("  Enter your question for the current knowledge base: ").strip()
            if not follow_up_query:
                print("  Query cannot be empty.")
                continue
            print(f"  Querying knowledge base for: '{follow_up_query}'...")
            context_docs_for_llm = rag_core.retrieve_context(current_vectorstore, follow_up_query)
            if not context_docs_for_llm:
                print("  Could not retrieve relevant context for your query.")
            else:
                llm_response = rag_core.generate_llm_answer(follow_up_query, context_docs_for_llm, history=None)
                print("\n--- Answer ---")
                print(llm_response.get("answer", "No answer generated."))
                print("\n--- Sources ---")
                print(llm_response.get("sources_md", "No sources provided."))
                print("--------------")

        elif command == "load":
            context_name = input("  Enter name of context to load: ").strip()
            if not context_name:
                print("  Context name cannot be empty for load.")
                continue
            print(f"  Loading context '{context_name}'...")
            vs, docs, msg = rag_core.core_load_context(context_name)
            print(f"  {msg}")
            if vs: # If FAISS index was loaded, it becomes the current context
                current_vectorstore = vs
                current_docs_for_context = docs # These are all docs associated with the loaded VS
            elif docs: # If only docs were loaded (VS failed or didn't exist)
                print("  Only document metadata was loaded. You might need to 'search' to build a new vector store from these or other documents.")
                # Decide if current_docs_for_context should be just these, or cleared if VS is primary
                current_docs_for_context = docs
                current_vectorstore = None # Ensure VS is None if it wasn't loaded

        elif command == "save":
            if not current_vectorstore:
                print("  No active knowledge base to save. Use 'search' or 'load' first.")
                continue
            context_name = input("  Enter name to save current context as: ").strip()
            if not context_name:
                print("  Context name cannot be empty for save.")
                continue
            print(f"  Saving current context as '{context_name}'...")
            # current_docs_for_context should reflect all docs in the current_vectorstore
            msg = rag_core.core_save_context(current_vectorstore, current_docs_for_context, context_name)
            print(f"  {msg}")

        elif command == "reset":
            current_vectorstore = None
            current_docs_for_context = []
            print("  In-memory knowledge base has been reset.")

        elif command == "exit":
            print("Exiting RAG CLI. Goodbye!")
            break

        else:
            print(f"  Unknown command: '{command}'. Available: search, query, load, save, reset, exit")

if __name__ == "__main__":
    # This ensures that rag_core components (like llm, embeddings) are initialized
    # when rag_core is imported, as their initialization is at the module level.
    main_cli()
