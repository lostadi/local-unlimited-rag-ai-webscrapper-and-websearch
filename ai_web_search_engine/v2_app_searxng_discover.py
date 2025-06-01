#-------------------------------------------------------------------------------------------------#
#-- Standard Library Imports --#
#-------------------------------------------------------------------------------------------------#
import os
import traceback
import time # Still needed for Gradio UI updates if any delays are desired there
import random # Still needed for Gradio UI example values or other UI logic
import json # For parsing potential JSON in UI or returning JSON
from urllib.parse import quote_plus, urljoin, urlparse # May be used by UI logic

#-------------------------------------------------------------------------------------------------#
#-- Third-Party Library Imports --#
#-------------------------------------------------------------------------------------------------#
import gradio as gr
# Removed requests, BeautifulSoup, FAISS, OllamaLLM, OllamaEmbeddings, PromptTemplate,
# Document, RecursiveCharacterTextSplitter as they are now in rag_core

#-------------------------------------------------------------------------------------------------#
#-- Core Logic Imports --#
#-------------------------------------------------------------------------------------------------#
import rag_core # Main import for the refactored logic

#-------------------------------------------------------------------------------------------------#
#-- Application Configuration (UI Specific) --#
#-------------------------------------------------------------------------------------------------#
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
GRADIO_PORT_PY = rag_core.GRADIO_PORT_PY # Use from rag_core, can be overridden by shell

# Constants that were moved to rag_core and are used via rag_core.<CONSTANT_NAME>
# SEARXNG_URL, SEARXNG_MAX_RESULTS, MAX_CONTENT_PER_PAGE, MIN_DELAY_PER_REQUEST_SCRAPE,
# MAX_DELAY_PER_REQUEST_SCRAPE, REQUEST_TIMEOUT, OLLAMA_MODEL_PY, OLLAMA_EMBEDDING_MODEL_PY,
# CHUNK_SIZE, CHUNK_OVERLAP, K_RETRIEVER, K_CONTEXT_FOR_LLM, USER_AGENTS, SHARED_CONTEXT_BASE_DIR

# Global components like llm, embeddings, text_splitter are now in rag_core
# and accessed via rag_core.llm, rag_core.embeddings, rag_core.text_splitter

#-------------------------------------------------------------------------------------------------#
#-- Gradio Application Logic (Wrappers around rag_core functions) --#
#-------------------------------------------------------------------------------------------------#
def perform_search_and_answer(query, existing_vectorstore_state):
    """
    Orchestrates the search, scrape, RAG, and answer generation process.
    This function is called by the chat_interface_handler.
    It now primarily calls functions from rag_core.
    """
    print(f"\nUI_PERFORM_SEARCH: Query='{query}'. Existing vectorstore in state: {'Yes' if existing_vectorstore_state else 'No'}")

    # 1. Discover and Scrape URLs
    scrape_response_dict = rag_core.searxng_discover_and_scrape(query)
    if scrape_response_dict.get("error"):
        status_msg = f"## Web Data Collection Error\n{scrape_response_dict['error']}"
        return existing_vectorstore_state, [], "", "", status_msg

    scraped_results_list_of_dicts = scrape_response_dict["results"]
    if not scraped_results_list_of_dicts:
        status_msg = "## No New Content Found\nNo new content retrieved from the web for your query."
        # Still return existing_vectorstore_state as it might be used for answering without new docs
        return existing_vectorstore_state, [], "", "", status_msg

    # 2. Get or Create/Update Vectorstore
    # rag_core.get_or_create_vectorstore handles chunking internally via _chunk_scraped_documents
    updated_vectorstore, status_detail_core = rag_core.get_or_create_vectorstore(
        processed_docs_as_dicts=scraped_results_list_of_dicts,
        existing_vectorstore=existing_vectorstore_state
    )

    if not updated_vectorstore:
        status_msg = f"## RAG Error\nVector store is unavailable. Core status: {status_detail_core}"
        return None, scraped_results_list_of_dicts, "", "", status_msg

    # 3. Retrieve Context from the (potentially updated) vector store
    # The query for retrieval should ideally be the original user query,
    # or could be expanded/refined based on strategy. For now, using original query.
    retrieved_docs_for_llm = rag_core.retrieve_context(updated_vectorstore, query) # Uses rag_core.K_RETRIEVER

    # 4. Generate LLM Answer
    # History is not passed from this top-level search function; handle_follow_up does that.
    llm_response = rag_core.generate_llm_answer(query, retrieved_docs_for_llm, history=None) # Uses rag_core.llm

    # 5. Compile final status message and results
    final_status_msg = status_detail_core
    answer_text = llm_response["answer"]
    sources_md = llm_response["sources_md"]

    if llm_response.get("error"):
        if llm_response["error"] == "Insufficient context":
             final_status_msg += f"\n\nNote from AI: {answer_text}" # Answer might contain the "insufficient context" message
        else:
            final_status_msg = f"LLM Generation Error: {llm_response['error']}. {status_detail_core}"
    elif not answer_text: # If no LLM error but also no answer
        final_status_msg += " No answer could be generated based on the available information."

    # Return the updated vector store, the newly scraped documents for this turn, answer, sources, and status.
    return updated_vectorstore, scraped_results_list_of_dicts, answer_text, sources_md, final_status_msg.strip()


def chat_interface_handler(user_input, history, state_vectorstore, state_all_scraped_docs_as_dicts):
    history = history or []

    # Call the refactored search and answer function
    vs, newly_scraped_docs, ans, srcs, stat_msg = perform_search_and_answer(user_input, existing_vectorstore=state_vectorstore)

    current_turn_output = ""
    if ans and ans != "Insufficient context.": # Avoid duplicating "Insufficient context" if already in stat_msg
         current_turn_output = f"## Answer\n{ans}\n\n### Sources\n{srcs}"

    if stat_msg:
        if current_turn_output: current_turn_output += f"\n\n---\n*Status: {stat_msg.strip()}*"
        else: current_turn_output = f"*Status: {stat_msg.strip()}*" # Display status if no answer

    if not current_turn_output: # Fallback if no answer and no specific status
        current_turn_output = "No answer generated or no new information processed. Please check logs."

    history.append((user_input, current_turn_output))

    refine_and_save_visible = gr.update(visible=True) if vs else gr.update(visible=False)

    # state_all_scraped_docs_as_dicts now stores only the docs from the *latest* scraping operation.
    return history, vs, newly_scraped_docs or [], refine_and_save_visible, refine_and_save_visible, stat_msg


def handle_follow_up(query, history, vectorstore, state_all_scraped_docs_as_dicts):
    """
    Handles follow-up questions based on the existing vectorstore.
    Does not perform new web searches.
    """
    print(f"\nUI_HANDLE_FOLLOW_UP: Query='{query}'")
    if not vectorstore:
        return "", "", "Error: No context loaded for follow-up. Start a new search or load a context."

    # Use rag_core functions for retrieval and answer generation
    retrieved_docs = rag_core.retrieve_context(vectorstore, query)

    # The 'context_for_llm_as_doc_objects' preparation logic from the original file can be simplified
    # if generate_llm_answer in rag_core handles the K_CONTEXT_FOR_LLM slicing.
    # Assuming generate_llm_answer in rag_core handles this.

    resp = rag_core.generate_llm_answer(query, retrieved_docs, history)

    status_msg = "Follow-up complete."
    if resp.get("error") and resp["error"] != "Insufficient context.": # "Insufficient context" is not an error for status
        status_msg = f"LLM Error on follow-up: {resp['error']}"
    elif resp.get("error") == "Insufficient context.":
        status_msg += f" Note: {resp['answer']}" # Append AI's "insufficient context" message

    return resp["answer"], resp["sources_md"], status_msg


def handle_refine_search():
    return [], None, [], gr.update(visible=False), gr.update(visible=False), "Context cleared. Ready for new search."

#-------------------------------------------------------------------------------------------------#
#-- Persistence Functions (UI Wrappers for rag_core) --#
#-------------------------------------------------------------------------------------------------#
def save_search_context_ui(vectorstore, all_scraped_docs_as_dicts, save_name):
    if not save_name or not save_name.strip():
        return "Error: Save name is empty."
    # Input validation for save_name is good here for immediate UI feedback.
    # core_save_context also has it, which is fine.
    return rag_core.core_save_context(vectorstore, all_scraped_docs_as_dicts, save_name)

def load_search_context_ui(load_name):
    if not load_name or not load_name.strip():
        return None, [], "Error: Load name empty.", gr.update(visible=False), gr.update(visible=False), []

    loaded_vs, loaded_docs, status_msg_core = rag_core.core_load_context(load_name)

    chatbot_history = []
    if loaded_vs or loaded_docs : # If anything was loaded
        chatbot_history = [(None, f"Context '{load_name}' loaded. {status_msg_core} You can now ask questions or perform new searches to add to this context.")]
    else: # Nothing found or error
        chatbot_history = [(None, status_msg_core)]

    return loaded_vs, loaded_docs, status_msg_core, gr.update(visible=True if (loaded_vs or loaded_docs) else False), gr.update(visible=True if loaded_vs else False), chatbot_history

#-------------------------------------------------------------------------------------------------#
#-- Gradio UI Definition --#
#-------------------------------------------------------------------------------------------------#
print("Building Gradio interface (SearXNG-Powered Discovery Version)...")
with gr.Blocks() as interface:
    gr.Markdown("# AI Web Search Assistant (SearXNG-Powered)")
    gr.Markdown(f"Enter query to search/scrape, or load/save context. Results append to existing context. Contexts saved/loaded from: `{rag_core.SHARED_CONTEXT_BASE_DIR}`")

    state_vectorstore = gr.State(None)
    state_all_scraped_docs_as_dicts = gr.State([]) # Stores docs from the *last* scraping operation

    with gr.Row():
        user_input_textbox = gr.Textbox(label="Your Query / Search Term", placeholder="Type query and hit Enter, or use Load/Save...", scale=4, lines=1)
        submit_button = gr.Button("Submit Query / Add to KB", variant="primary", scale=1)

    chatbot = gr.Chatbot(label="Conversation History & Answers", height=400, bubble_full_width=False, value=[(None, "Welcome! Enter a query above, or load a saved context below.")])
    status_textbox = gr.Textbox(label="Status & Notifications", interactive=False, lines=2)

    with gr.Row():
        save_name_textbox = gr.Textbox(label="Enter Name to Save Current Context As:", placeholder="e.g., project_research_v1", scale=3)
        save_button = gr.Button("Save Context", visible=False, scale=1)

    with gr.Row():
        load_name_textbox = gr.Textbox(label="Enter Name of Context to Load:", placeholder="e.g., project_research_v1", scale=3)
        load_button = gr.Button("Load Context", visible=True, scale=1)

    refine_button = gr.Button("Clear Current In-Memory Context & Chat", visible=False)

    gr.Markdown("Use simple names (letters, numbers, underscores) for context files.")

    # Event handlers
    submit_button.click(
        fn=chat_interface_handler,
        inputs=[user_input_textbox, chatbot, state_vectorstore, state_all_scraped_docs_as_dicts],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs_as_dicts, refine_button, save_button, status_textbox]
    ).then(lambda: gr.update(value=""), outputs=[user_input_textbox])

    user_input_textbox.submit(
        fn=chat_interface_handler,
        inputs=[user_input_textbox, chatbot, state_vectorstore, state_all_scraped_docs_as_dicts],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs_as_dicts, refine_button, save_button, status_textbox]
    ).then(lambda: gr.update(value=""), outputs=[user_input_textbox])

    refine_button.click(
        fn=handle_refine_search,
        inputs=[],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs_as_dicts, refine_button, save_button, status_textbox]
    )

    save_button.click(
        fn=save_search_context_ui,
        inputs=[state_vectorstore, state_all_scraped_docs_as_dicts, save_name_textbox],
        outputs=[status_textbox]
    )

    load_button.click(
        fn=load_search_context_ui,
        inputs=[load_name_textbox],
        outputs=[state_vectorstore, state_all_scraped_docs_as_dicts, status_textbox, refine_button, save_button, chatbot]
    )

print("Gradio interface built.")
#-------------------------------------------------------------------------------------------------#
#-- Application Launch --#
#-------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    print(f"Starting Gradio application (SearXNG-Powered Discovery Version)...")
    print(f"--- Configuration ---")
    # Access constants from rag_core module
    print(f"Expecting SearXNG at: {rag_core.SEARXNG_URL}")
    print(f"Gradio Port: {GRADIO_PORT_PY}") # GRADIO_PORT_PY is UI specific
    print(f"LLM Model: {rag_core.OLLAMA_MODEL_PY}")
    print(f"Embedding Model: {rag_core.OLLAMA_EMBEDDING_MODEL_PY}")
    print(f"Shared Context Directory: {rag_core.SHARED_CONTEXT_BASE_DIR}")
    print(f"---------------------")
    print(f"IMPORTANT: Ensure your SearXNG instance is running and accessible at {rag_core.SEARXNG_URL} BEFORE starting this app.")
    print(f"Access Gradio at: http://0.0.0.0:{GRADIO_PORT_PY} or http://localhost:{GRADIO_PORT_PY}")

    interface.launch(server_port=GRADIO_PORT_PY, server_name="0.0.0.0", debug=False)
