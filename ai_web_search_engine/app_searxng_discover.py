#-------------------------------------------------------------------------------------------------#
#-- Standard Library Imports --#
#-------------------------------------------------------------------------------------------------#
import os
import traceback
import time
import random
import json # For parsing SearXNG JSON response
from urllib.parse import quote_plus, urljoin, urlparse
#-------------------------------------------------------------------------------------------------#
#-- Third-Party Library Imports --#
#-------------------------------------------------------------------------------------------------#
from math import inf 
import gradio as gr
import requests # For SearXNG API call and page fetching
from bs4 import BeautifulSoup # For page content parsing

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

#-------------------------------------------------------------------------------------------------#
#-- Application Configuration --#
#-------------------------------------------------------------------------------------------------#
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

SEARXNG_URL = "http://localhost:8080"  # Default SearXNG address - CHANGE IF YOURS IS DIFFERENT
SEARXNG_MAX_RESULTS = 150              # How many results to request from SearXNG
                                       # SearXNG itself will get more from backends and rank them

MAX_CONTENT_PER_PAGE = 250000
MIN_DELAY_PER_REQUEST_SCRAPE = 0.5 # For scraping individual pages
MAX_DELAY_PER_REQUEST_SCRAPE = 1.0 # For scraping individual pages
REQUEST_TIMEOUT = 100000

OLLAMA_MODEL_PY = "huihui_ai/qwen3-abliterated:0.6b"
OLLAMA_EMBEDDING_MODEL_PY = "nomic-embed-text"
GRADIO_PORT_PY = 12123 # New port

K_RETRIEVER = 200
K_CONTEXT_FOR_LLM = 50

USER_AGENTS = [ # For scraping the actual content pages
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36',
]

#-------------------------------------------------------------------------------------------------#
#-- Initialize Core Components --#
#-------------------------------------------------------------------------------------------------#
print("Initializing Langchain components...")
llm = OllamaLLM(model=OLLAMA_MODEL_PY)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_PY)
print("Initialization complete.")

#-------------------------------------------------------------------------------------------------#
#-- Helper for Polite Requests (used for scraping, not SearXNG call directly) --#
#-------------------------------------------------------------------------------------------------#
def make_polite_request(url, method='GET', data=None, headers=None, stream=False):
    req_headers = {'User-Agent': random.choice(USER_AGENTS)}
    if headers: req_headers.update(headers)
    time.sleep(random.uniform(MIN_DELAY_PER_REQUEST_SCRAPE / 2, MAX_DELAY_PER_REQUEST_SCRAPE / 2))
    try:
        response = requests.request(method, url, headers=req_headers, data=data,
                                    timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=stream)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"  REQUEST_HELPER: Error fetching {url}: {e}")
        return None

#-------------------------------------------------------------------------------------------------#
#-- URL Discovery via Local SearXNG Instance --#
#-------------------------------------------------------------------------------------------------#
def discover_urls_via_searxng(query):
    """
    Queries a local SearXNG instance to get search result URLs.
    """
    print(f"SEARXNG_DISCOVERY: Querying local SearXNG for '{query}'")
    # SearXNG JSON API endpoint
    # Note: `pageno=1` might be needed if you want more than default per page from searxng itself
    # `language='all'` or specify, e.g. `en`
    params = {
        'q': query,
        'format': 'json',
        'pageno': 1, # Request first page of results from SearXNG
        # 'language': 'en', # Optional: specify language
        # 'time_range': 'day', # Optional: 'day', 'week', 'month', 'year'
        # 'safesearch': 1, # 0 (None), 1 (Moderate), 2 (Strict)
    }
    
    discovered_urls_info = []
    try:
        # The request to your local SearXNG instance doesn't need the same "politeness"
        # as scraping external sites, but still good to have a timeout.
        response = requests.get(f"{SEARXNG_URL.rstrip('/')}/search", params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if "results" in data and data["results"]:
            print(f"SEARXNG_DISCOVERY: Found {len(data['results'])} results from SearXNG.")
            for res in data["results"][:SEARXNG_MAX_RESULTS]: # Limit how many we process
                title = res.get('title', 'No Title')
                url = res.get('url')
                # content = res.get('content', '') # SearXNG might provide a snippet
                
                if url:
                    # Basic URL cleaning/validation
                    if not url.startswith(('http://', 'https://')):
                        print(f"SEARXNG_DISCOVERY: Skipping invalid/relative URL from SearXNG: {url}")
                        continue
                    discovered_urls_info.append({'title': title, 'href': url}) # 'snippet': content
        else:
            print(f"SEARXNG_DISCOVERY: No 'results' field or empty results from SearXNG.")
            # You could inspect data['infoboxes'] or data['unresponsive_engines'] for debugging SearXNG itself

    except requests.exceptions.ConnectionError as e:
        print(f"SEARXNG_DISCOVERY: CRITICAL ERROR - Could not connect to SearXNG at {SEARXNG_URL}.")
        print(f"                   Ensure SearXNG is running and accessible. Error: {e}")
        return {"error": f"Could not connect to local SearXNG search instance at {SEARXNG_URL}. Please ensure it's running.", "results": []}
    except requests.exceptions.RequestException as e:
        print(f"SEARXNG_DISCOVERY: Error querying SearXNG API: {e}")
        traceback.print_exc()
        return {"error": f"Error querying local SearXNG instance: {str(e)}", "results": []}
    except json.JSONDecodeError as e:
        print(f"SEARXNG_DISCOVERY: Error decoding JSON response from SearXNG: {e}")
        print(f"                   Raw response text: {response.text[:500]}...") # Log part of raw response
        return {"error": f"Invalid JSON response from local SearXNG instance: {str(e)}", "results": []}


    if not discovered_urls_info:
        print("SEARXNG_DISCOVERY: SearXNG yielded no usable URLs.")
        return {"error": "Local SearXNG instance returned no usable URLs for the query.", "results": []}

    print(f"SEARXNG_DISCOVERY: Successfully discovered {len(discovered_urls_info)} URLs via SearXNG.")
    return {"error": None, "results": discovered_urls_info}


#-------------------------------------------------------------------------------------------------#
#-- Content Fetching & Main Orchestrator --#
#-------------------------------------------------------------------------------------------------#
def fetch_page_content(url): # This function remains largely the same
    print(f"  FETCH_PAGE: Attempting to fetch {url}")
    response = make_polite_request(url) # Uses the helper with delays/user-agent
    if not response: return None, None
    try:
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"  FETCH_PAGE: Skipping non-HTML content at {url} (type: {content_type})")
            return None, None
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.string.strip() if title_tag else "No Title Found"
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        if not text_content.strip():
            body = soup.find('body')
            if body:
                for unwanted_tag in body(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
                    unwanted_tag.decompose()
                text_content = body.get_text(separator=' ', strip=True, Sijoiner=' ')
        print(f"  FETCH_PAGE: Parsed '{title}' (len: {len(text_content)}) from {url}")
        return title, text_content[:MAX_CONTENT_PER_PAGE].strip()
    except Exception as e:
        print(f"  FETCH_PAGE: Error processing HTML for {url}: {e}"); return None, None

def searxng_discover_and_scrape(query):
    print(f"SEARXNG_DISCOVER_AND_SCRAPE: Orchestrating for query='{query}'")
    
    # --- Step 1: Discover URLs via SearXNG ---
    url_discovery_response = discover_urls_via_searxng(query)
    if url_discovery_response.get("error"):
        # Pass the error up so it can be shown in the UI
        return url_discovery_response 
    
    discovered_urls_info = url_discovery_response["results"]
    if not discovered_urls_info:
        return {"error": "URL discovery via SearXNG yielded no results.", "results": []}

    print(f"SEARXNG_DISCOVER_AND_SCRAPE: Will attempt to scrape content from {len(discovered_urls_info)} URLs.")
    
    scraped_documents = []
    # --- Step 2: Scrape Content from Discovered URLs ---
    for i, url_info in enumerate(discovered_urls_info):
        url_to_scrape = url_info['href']
        original_title = url_info.get('title', 'Untitled')
        print(f"\nSEARXNG_DISCOVER_AND_SCRAPE: Scraping URL {i+1}/{len(discovered_urls_info)}: {url_to_scrape}")
        page_title, page_content = fetch_page_content(url_to_scrape) # Polite request is inside fetch_page_content
        if page_content and page_content.strip():
            scraped_documents.append({
                "title": page_title if page_title != "No Title Found" else original_title,
                "source": url_to_scrape,
                "page_content": page_content
            })
        # Individual page scraping already has delays within make_polite_request
        # An additional master delay between scraping different URLs *could* be added here if needed.
        # time.sleep(random.uniform(0.5, 1.5)) # Optional extra delay

    if not scraped_documents:
        return {"error": "Failed to scrape any content from the URLs provided by SearXNG.", "results": []}

    print(f"SEARXNG_DISCOVER_AND_SCRAPE: Successfully scraped {len(scraped_documents)} documents.")
    return {"error": None, "results": scraped_documents}


#-------------------------------------------------------------------------------------------------#
#-- RAG/LLM Functions (create_vectorstore, retrieve_context, generate_llm_answer) --#
#-- These remain THE SAME (ensure they are complete from previous versions) --#
#-------------------------------------------------------------------------------------------------#
def create_vectorstore(processed_docs): # (Same as before)
    print(f"CREATE_VECTORSTORE: From {len(processed_docs)} docs.")
    if not processed_docs: return None
    texts = [doc["page_content"] for doc in processed_docs if doc["page_content"]]
    metadatas = [{"source": doc["source"], "title": doc["title"]} for doc in processed_docs if doc["page_content"]]
    if not texts: print("CREATE_VECTORSTORE: No valid texts to create index."); return None
    try:
        vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        print("CREATE_VECTORSTORE: FAISS index created."); return vs
    except Exception as e: print(f"CREATE_VECTORSTORE: Error - {e}"); traceback.print_exc(); return None

def retrieve_context(vectorstore, query, k=K_RETRIEVER): # (Same as before)
    if not vectorstore: print("RETRIEVE_CONTEXT: No vectorstore."); return []
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        print(f"RETRIEVE_CONTEXT: Retrieved {len(docs)} docs."); return docs
    except Exception as e: print(f"RETRIEVE_CONTEXT: Error - {e}"); traceback.print_exc(); return []

def generate_llm_answer(query, context_docs, history=None): # (Same as before, ensure complete)
    actual_context = context_docs[:K_CONTEXT_FOR_LLM]
    if not actual_context:
        return {"answer": "I could not find enough relevant information in the collected web pages to answer your question.", 
                "sources_md": "", "error": "Insufficient context"}
    context_text = "\n\n---\n\n".join([doc.page_content for doc in actual_context])
    sources_md_output = "" # Logic for sources_md_output...
    llm_context_sources_md_list = []
    seen_sources_for_llm = set()
    for doc in actual_context:
        source_url = doc.metadata.get('source')
        source_title = doc.metadata.get('title', 'Source')
        if source_url and source_url not in seen_sources_for_llm:
            safe_title = source_title.replace('[', '(').replace(']', ')')
            llm_context_sources_md_list.append(f"- [{safe_title}]({source_url})")
            seen_sources_for_llm.add(source_url)
    sources_md_output = "\n".join(llm_context_sources_md_list)
    if not sources_md_output: sources_md_output = "*(No specific web sources identified for this answer from the context.)*"

    prompt_template_str = (
        "You are a helpful AI assistant to LouAnn. Based *only* on the following CONTEXT (snippets from web pages), "
        "answer the user's QUESTION. Consider the CHAT HISTORY for conversational context. "
        "Your answer should be concise and directly address the question. "
        "If the CONTEXT is insufficient or irrelevant, state that clearly. Do not make up information.\n\n"
        "CHAT HISTORY:\n{history_formatted}\n\n"
        "CONTEXT (Web Page Snippets):\n------------------------\n{context}\n------------------------\n\n"
        "Question: {question}\n\nAnswer:" )
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    history_formatted = "\n".join([f"Human: {h}\nAI: {a}" for h,a in history]) if history else "(No previous conversation)"
    chain = prompt_template | llm
    try:
        llm_answer_text = chain.invoke({"context": context_text, "question": query, "history_formatted": history_formatted})
        return {"answer": llm_answer_text.strip(), "sources_md": sources_md_output, "error": None}
    except Exception as e:
        print(f"GENERATE_LLM_ANSWER: Error - {e}"); traceback.print_exc()
        return {"answer": f"LLM Error during generation: {e}", "sources_md": sources_md_output, "error": str(e)}

#-------------------------------------------------------------------------------------------------#
#-- Gradio Application Logic --#
#-------------------------------------------------------------------------------------------------#
def perform_initial_search_with_searxng(query):
    print(f"\nPERFORM_INITIAL_SEARCH_WITH_SEARXNG: Query='{query}'")
    scrape_response = searxng_discover_and_scrape(query) # NEW ORCHESTRATOR
    
    if scrape_response.get("error"):
        status = f"## Web Data Collection Error\n{scrape_response['error']}"
        return None, [], "", "", status # Return the error to be displayed
        
    scraped_results_list = scrape_response["results"]
    if not scraped_results_list: # Should be caught by error above, but defensive check
        status = "## No Content Found\nNo content retrieved using SearXNG discovery and scraping."
        return None, [], "", "", status

    vectorstore = create_vectorstore(scraped_results_list)
    if not vectorstore:
        status = "## RAG Error\nCould not create vector store. There might have been no valid content to index."
        return None, scraped_results_list, "", "", status # Return scraped list for potential debugging
    
    retrieved_docs = retrieve_context(vectorstore, query)
    # Use all scraped docs if specific retrieval for the query yields little, to give LLM more to work with.
    context_for_llm = retrieved_docs if len(retrieved_docs) >= K_CONTEXT_FOR_LLM // 2 else scraped_results_list
    
    llm_response = generate_llm_answer(query, context_for_llm, history=None)
    
    status_msg = "Initial search (SearXNG-powered) and answer generation complete."
    answer_text = llm_response["answer"]
    sources_md = llm_response["sources_md"]

    if llm_response.get("error"):
        if llm_response["error"] == "Insufficient context":
             status_msg += f"\n\nNote from AI: {answer_text}" # AI already stated insufficiency
        else: # Other LLM error
            status_msg = f"## LLM Generation Error: {llm_response['error']}"
            # answer_text might contain the error message if not handled above

    # Pass all scraped documents to state so follow-ups can use this full context if needed
    return vectorstore, scraped_results_list, answer_text, sources_md, status_msg


# --- chat_interface_handler, handle_follow_up, etc. (largely same structure) ---
def chat_interface_handler(user_input, history, state_vectorstore, state_all_scraped_docs): # Renamed state var
    print(f"\nCHAT_INTERFACE_HANDLER (SearXNG Discover): Input='{user_input}'")
    history = history or []
    if state_vectorstore is None: # Initial search
        vs, all_docs, ans, srcs, stat = perform_initial_search_with_searxng(user_input)
        
        current_turn_output = ""
        if ans:
             current_turn_output = f"## Answer\n{ans}\n\n### Sources Referenced in this Answer:\n{srcs}"
        if stat:
            if current_turn_output: current_turn_output += f"\n\n---\n*Status: {stat}*"
            else: current_turn_output = f"*Status: {stat}*"
        if not current_turn_output: # If scrape and LLM failed badly
            current_turn_output = "An unexpected error occurred during processing. Please check logs or try again."


        history.append((user_input, current_turn_output))
        buttons_visible = gr.update(visible=True) if vs else gr.update(visible=False)
        # Store *all* initially scraped documents in state_all_scraped_docs
        return history, vs, all_docs if all_docs else [], buttons_visible, buttons_visible, stat
    else: # Follow-up
        ans, srcs, stat = handle_follow_up(user_input, history, state_vectorstore, state_all_scraped_docs) # Pass all_scraped_docs
        resp_fmt = f"{ans}\n\n### Sources Potentially Relevant:\n{srcs}" if ans else stat
        history.append((user_input, resp_fmt))
        return history, state_vectorstore, state_all_scraped_docs, gr.update(visible=True), gr.update(visible=True), stat

def handle_follow_up(query, history, vectorstore, all_scraped_docs): # Takes all_scraped_docs
    print(f"\nHANDLE_FOLLOW_UP: Query='{query}'")
    if not vectorstore: return "", "", "Error: No context for follow-up. Start a new search."
    
    current_retrieved_docs = retrieve_context(vectorstore, query)
    # Use specifically retrieved docs for this follow-up if available, else use all scraped docs
    context_for_llm = current_retrieved_docs if current_retrieved_docs else all_scraped_docs
    
    if not context_for_llm: # Should not happen if all_scraped_docs has content
        return "I don't have any information from the previous search to answer this.", "", "No context."
        
    resp = generate_llm_answer(query, context_for_llm, history)
    status_msg = "Follow-up complete."
    if resp.get("error") and resp["error"] != "Insufficient context":
        status_msg = f"LLM Error on follow-up: {resp['error']}"
    elif resp.get("error") == "Insufficient context": # Append LLM's own message
        status_msg += f" Note: {resp['answer']}"

    return resp["answer"], resp["sources_md"], status_msg

def handle_refine_search(): # Same
    return [], None, [], gr.update(visible=False), gr.update(visible=False), "Ready for new search."

#-------------------------------------------------------------------------------------------------#
#-- Gradio UI Definition --#
#-------------------------------------------------------------------------------------------------#
print("Building Gradio interface (SearXNG-Powered Discovery Version)...")
with gr.Blocks(theme=gr.themes.Glass()) as interface:
    gr.Markdown("# LouAnn's AI Web Search Assistant (SearXNG-Powered)")
    gr.Markdown(
        "Enter a topic. The assistant queries a **local SearXNG instance** for relevant web pages, "
        "then scrapes content from those pages to generate an answer."
        f"\n*   **SearXNG Instance URL (Expected):** `{SEARXNG_URL}` (Must be running separately!)"
        f"\n*   **LLM for Answers:** `{OLLAMA_MODEL_PY}`"
        "\n*   **Note:** This process can be slow. Ensure your local SearXNG instance is running and configured."
    )
    state_vectorstore = gr.State(None)
    state_all_scraped_docs = gr.State([]) # Stores all docs from initial scrape for follow-ups
    
    chatbot = gr.Chatbot(label="Conversation", height=550, bubble_full_width=False)
    with gr.Row():
        user_input_textbox = gr.Textbox(label="Your Query", placeholder="Type your search query here...", scale=4, lines=2)
        submit_button = gr.Button("Submit / Ask", variant="primary", scale=1)
    with gr.Row():
        # continue_button = gr.Button("Continue Discussion", visible=False, interactive=False)
        refine_button = gr.Button("Search Again / New Topic", visible=False) # Only one button needed now
    status_textbox = gr.Textbox(label="Current Action / Status", interactive=False, lines=1, value="Ready. Ensure SearXNG is running.")

    # Event handlers
    submit_button.click(
        fn=chat_interface_handler,
        inputs=[user_input_textbox, chatbot, state_vectorstore, state_all_scraped_docs],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs, refine_button, refine_button, status_textbox] # refine_button used twice for visibility control
    ).then(lambda: gr.update(value=""), outputs=[user_input_textbox])

    user_input_textbox.submit(
        fn=chat_interface_handler,
        inputs=[user_input_textbox, chatbot, state_vectorstore, state_all_scraped_docs],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs, refine_button, refine_button, status_textbox]
    ).then(lambda: gr.update(value=""), outputs=[user_input_textbox])

    refine_button.click(
        fn=handle_refine_search,
        inputs=[],
        outputs=[chatbot, state_vectorstore, state_all_scraped_docs, refine_button, refine_button, status_textbox]
    )

print("Gradio interface built.")
#-------------------------------------------------------------------------------------------------#
#-- Application Launch --#
#-------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    print(f"Starting Gradio application (SearXNG-Powered Discovery Version)...")
    print(f"--- Configuration ---")
    print(f"Expecting SearXNG at: {SEARXNG_URL}")
    print(f"Gradio Port: {GRADIO_PORT_PY}")
    print(f"LLM Model: {OLLAMA_MODEL_PY}")
    print(f"---------------------")
    print(f"IMPORTANT: Ensure your SearXNG instance is running and accessible at {SEARXNG_URL} BEFORE starting this app.")
    print(f"Access Gradio at: http://0.0.0.0:{GRADIO_PORT_PY} or http://localhost:{GRADIO_PORT_PY}")
    
    interface.launch(server_port=GRADIO_PORT_PY, server_name="0.0.0.0", debug=False)
