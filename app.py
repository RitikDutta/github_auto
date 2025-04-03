import os
import json
from typing import TypedDict, Optional
from flask import Flask, request, render_template

# Import the google-generativeai library
import google.generativeai as genai

from langgraph.graph import StateGraph, END

# --- Step 1a: Define your API Key Variable HERE ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual API key string
# For production, use environment variables instead of hardcoding!
# Example using environment variable (recommended):
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyDNOZ4XoQjFLPWMZ0cmIYc1w8nx3eYS8hg"
# ---------------------------------------------------

# --- Step 1b: Configure SDK using the variable ---
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError("Please set the GOOGLE_API_KEY variable with your actual Google API Key.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("--- Google Generative AI SDK Configured ---")
except Exception as e:
    print(f"Error configuring Google Generative AI SDK: {e}")
    raise

# --- Step 2: Define LangGraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    'initial_message' is less relevant for the web app, but kept for structure.
    """
    initial_message: Optional[str] # Less relevant now
    llm_prompt: str
    llm_response_text: Optional[str]

# --- Step 3: Define LangGraph Nodes ---
# This node is optional for the web interaction but kept for logging/example
def log_start_node(state: GraphState) -> dict:
    """Logs the start and the received prompt."""
    print(f"--- Node: Log Start ---")
    print(f"Received prompt: {state.get('llm_prompt', 'N/A')}")
    # Can add a timestamp or other info if desired
    return {"initial_message": "Processing started"} # Example state update

def invoke_llm_node(state: GraphState) -> dict:
    """
    Invokes the Google Gemini LLM using the configured google-generativeai SDK.
    Updates the state with the response text or an error message.
    """
    print("\n--- Node: Invoke LLM (using google.generativeai) ---")
    prompt = state.get("llm_prompt")
    if not prompt:
        print("Error: No LLM prompt found in state.")
        return {"llm_response_text": "Error: Internal - Prompt missing in state."}

    print(f"Sending prompt to Gemini: '{prompt}'")
    try:
        model_name = "models/gemini-1.5-flash-latest" # Or your preferred model
        print(f"Using model: {model_name}")

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(contents=prompt)

        response_text = None
        if response.parts:
            response_text = response.text
        elif hasattr(response, 'text'):
             response_text = response.text

        if response_text:
            print(f"LLM Response Received: '{response_text}'")
            return {"llm_response_text": response_text}
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name
                print(f"LLM call blocked. Reason: {reason}")
                # Return a user-friendly message about blocking
                return {"llm_response_text": f"Request blocked due to: {reason}. Please modify your prompt."}
            else:
                print(f"LLM response issue (no text/parts, not blocked). Response: {response}")
                return {"llm_response_text": "Error: Received an empty response from the AI."}

    except Exception as e:
        print(f"Error invoking LLM via google-generativeai: {e}")
        # Return a generic error to the user, log the specific one
        return {"llm_response_text": f"Error: Could not get response from AI. ({type(e).__name__})"}


# --- Step 4: Define and Compile LangGraph ---
# This is done once when the Flask app starts
workflow = StateGraph(GraphState)
workflow.add_node("logger", log_start_node)       # Start with logging
workflow.add_node("llm_invoker", invoke_llm_node) # Then invoke LLM

workflow.set_entry_point("logger")                # Entry point is logger
workflow.add_edge("logger", "llm_invoker")        # logger -> llm_invoker
workflow.add_edge("llm_invoker", END)             # llm_invoker -> END

langgraph_app = workflow.compile()
print("--- LangGraph Application Compiled ---")


# --- Step 5: Create Flask Application ---
flask_app = Flask(__name__) # Use 'flask_app' to avoid conflict with langgraph 'app'

@flask_app.route('/', methods=['GET', 'POST'])
def home():
    prompt_received = None
    llm_response = None

    if request.method == 'POST':
        prompt_received = request.form.get('prompt')
        if prompt_received:
            print(f"\n--- Flask Request Received ---")
            print(f"User Prompt: {prompt_received}")

            # Prepare initial state for LangGraph
            initial_state = {
                "llm_prompt": prompt_received,
                # Initialize other state fields
                "initial_message": None,
                "llm_response_text": None
            }

            try:
                # Invoke the compiled LangGraph application
                final_state = langgraph_app.invoke(initial_state)
                llm_response = final_state.get("llm_response_text", "Error: No response found in graph state.")

            except Exception as e:
                print(f"Error during LangGraph invocation: {e}")
                llm_response = f"Error: Failed to process the request. ({type(e).__name__})"

            print(f"Response to return: {llm_response}")
            print(f"--- Flask Request Finished ---")

        else:
            # Handle case where form is submitted but prompt is empty
            llm_response = "Please enter a prompt."

    # Render the HTML template
    # Pass the received prompt and the response back to the template
    return render_template('index.html', prompt=prompt_received, response=llm_response)

# --- Step 6: Run the Flask App ---
if __name__ == '__main__':
    # Use debug=True for development (auto-reloads changes)
    # Use host='0.0.0.0' to make it accessible on your network (optional)
    flask_app.run(debug=True, host='0.0.0.0', port=5000)