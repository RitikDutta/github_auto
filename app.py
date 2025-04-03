import os
import base64
import json
from typing import List, Optional, TypedDict, Annotated, Sequence
import operator # For state updates

from flask import Flask, request, render_template, flash
from dotenv import load_dotenv

# --- GitHub Tool Imports ---
from github import Github
from github.GithubException import (
    UnknownObjectException,
    BadCredentialsException,
    GithubException,
)

# --- LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage # Import SystemMessage
from langchain_core.tools import tool # Decorator to create tools
from langgraph.graph import StateGraph, END
# Use MessagesState for convenient agent message history management
from langgraph.prebuilt import ToolExecutor

# Use LangChain's wrapper for Gemini for easier tool integration
from langchain_google_genai import ChatGoogleGenerativeAI


# === 1. GitHub Tool Class (Keep your existing, correct class here) ===
class Github_Auto:
    """
    Manages interactions with a specific GitHub repository.
    (Paste the full class code here - no changes needed inside the class itself)
    """
    def __init__(self, token: str, repo_name: str, branch: str = "main"):
        # ... (rest of your __init__ code) ...
        self.token = token
        self.repo_name = repo_name
        self.branch = branch
        self.github_instance: Optional[Github] = None
        self.repo = None

        if not token: raise ValueError("GitHub token is required.")
        if not repo_name: raise ValueError("Repository name required.")

        try:
            print("Attempting to authenticate with GitHub...")
            self.github_instance = Github(self.token, timeout=30)
            user = self.github_instance.get_user()
            print(f"Auth successful for user: {user.login}")
            print(f"Accessing repository: {self.repo_name}")
            self.repo = self.github_instance.get_repo(self.repo_name)
            print(f"Accessed repo '{self.repo.full_name}', default branch '{self.repo.default_branch}'.")
            print(f"Targeting branch: '{self.branch}'")
            try:
                self.repo.get_branch(self.branch)
                print(f"Target branch '{self.branch}' confirmed.")
            except UnknownObjectException:
                 print(f"WARNING: Target branch '{self.branch}' does not exist. Operations might use default.")
        except BadCredentialsException: print("ERROR: Invalid GitHub token."); raise
        except UnknownObjectException: print(f"ERROR: Repo '{self.repo_name}' not found or permission denied."); raise
        except GithubException as gh_err: print(f"ERROR: GitHub API error: {gh_err}"); raise
        except Exception as e: print(f"ERROR: Unexpected init error: {e}"); raise

    def list_repository_files(self, directory_path: str = "") -> List[str]:
        # ... (rest of your list_repository_files code - ensure it returns List[str]) ...
        if not self.repo: return ["Error: Repository object not initialized."]
        print(f"\nTOOL: Listing files in '{self.repo_name}/{directory_path}' (branch: {self.branch})...")
        all_files = []
        try:
            contents = self.repo.get_contents(directory_path, ref=self.branch)
            content_stack = list(contents)
            while content_stack:
                content_file = content_stack.pop()
                if content_file.type == "dir":
                    # print(f"  Scanning directory: {content_file.path}") # Less verbose log
                    try: content_stack.extend(self.repo.get_contents(content_file.path, ref=self.branch))
                    except Exception as dir_e: print(f"  Warn: Cannot list '{content_file.path}': {dir_e}")
                else:
                    # print(f"  Found file: {content_file.path}") # Less verbose log
                    all_files.append(content_file.path)
            print(f"Found {len(all_files)} files/dirs at top level of '{directory_path or '/'}'. Use recursive listing tool if needed for deeper view.")
            return all_files if all_files else ["No files found in the specified path."]
        except UnknownObjectException: msg = f"Error: Dir '{directory_path}' not found."; print(msg); return [msg]
        except GithubException as e: msg = f"Error listing contents: {e}"; print(msg); return [msg]
        except Exception as e: msg = f"Unexpected error listing files: {e}"; print(msg); return [msg]


    def get_file_content(self, file_path: str) -> str:
        # ... (rest of your get_file_content code - ensure it returns str) ...
        if not self.repo: return "Error: Repository object not initialized."
        print(f"\nTOOL: Attempting fetch content: {file_path} (branch: {self.branch})")
        try:
            file_content_obj = self.repo.get_contents(file_path, ref=self.branch)
            if isinstance(file_content_obj, list): msg = f"Error: Path '{file_path}' is directory."; print(msg); return msg
            if file_content_obj.type != 'file': msg = f"Error: Path '{file_path}' not file (type: {file_content_obj.type})."; print(msg); return msg
            if file_content_obj.content:
                decoded_content = base64.b64decode(file_content_obj.content).decode('utf-8')
                print(f"Success fetch/decode: {file_path}")
                return decoded_content
            else: print(f"File '{file_path}' is empty."); return ""
        except UnknownObjectException: msg = f"Error: File not found at '{file_path}' on branch '{self.branch}'."; print(msg); return msg
        except GithubException as e: msg = f"GitHub error fetching '{file_path}': {e}"; print(msg); return msg
        except Exception as e: msg = f"Unexpected error fetching '{file_path}': {e}"; print(msg); return msg

    def create_or_update_file(self, file_path: str, content: str, commit_message: str) -> str:
        # ... (rest of your create_or_update_file code - ensure it returns str) ...
        if not self.repo: return "Error: Repository object not initialized."
        print(f"\nTOOL: Attempt create/update: {file_path} on branch '{self.branch}'")
        current_sha = None
        try:
            existing_file = self.repo.get_contents(file_path, ref=self.branch)
            if not isinstance(existing_file, list) and existing_file.type == 'file':
                current_sha = existing_file.sha
                print(f"File '{file_path}' exists (SHA: {current_sha}). Updating.")
            else: msg = f"Error: Path '{file_path}' exists but not regular file."; print(msg); return msg
        except UnknownObjectException: print(f"File '{file_path}' not exist. Creating."); current_sha = None
        except GithubException as e: msg = f"Error checking exist '{file_path}': {e}"; print(msg); return msg
        except Exception as e: msg = f"Unexpected error checking '{file_path}': {e}"; print(msg); return msg
        try:
            action = "(update)" if current_sha else "(create)"
            full_commit_message = f"{commit_message} {action}"
            if current_sha:
                commit_info = self.repo.update_file(path=file_path, message=full_commit_message, content=content, sha=current_sha, branch=self.branch)
                success_msg = f"Success update '{file_path}'. Commit: {commit_info['commit'].sha}"
            else:
                commit_info = self.repo.create_file(path=file_path, message=full_commit_message, content=content, branch=self.branch)
                success_msg = f"Success create '{file_path}'. Commit: {commit_info['commit'].sha}"
            print(success_msg); return success_msg
        except GithubException as e: msg = f"Error GitHub op {action} for '{file_path}': {e}"; print(msg); return msg
        except Exception as e: msg = f"Unexpected error {action} for '{file_path}': {e}"; print(msg); return msg


# === 2. Configuration and Initialization ===
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY missing.")
if not GITHUB_TOKEN: raise ValueError("GITHUB_TOKEN missing.")
if not GITHUB_REPO_NAME: raise ValueError("GITHUB_REPO_NAME missing.")

try:
    print("\n--- Initializing Github Bot ---")
    github_bot = Github_Auto(token=GITHUB_TOKEN, repo_name=GITHUB_REPO_NAME, branch=GITHUB_BRANCH)
    print("--- Github Bot Initialized Successfully ---")
except Exception as e:
    print(f"FATAL: Failed to initialize Github_Auto: {e}")
    github_bot = None

# === 3. Define LangGraph Tools ===
tools = []
if github_bot:
    @tool
    def list_github_files(directory_path: str = "") -> List[str]:
        """Lists files/directories within a specific directory path (default is root) of the configured GitHub repository. Does not recurse automatically."""
        return github_bot.list_repository_files(directory_path)

    @tool
    def read_github_file(file_path: str) -> str:
        """Reads the content of a specific file using its full path from the configured GitHub repository."""
        return github_bot.get_file_content(file_path)

    @tool
    def write_github_file(file_path: str, content: str, commit_message: str) -> str:
        """Creates a new file or updates an existing file (requires full path) in the configured GitHub repository."""
        return github_bot.create_or_update_file(file_path, content, commit_message)

    tools = [list_github_files, read_github_file, write_github_file]
    print(f"--- {len(tools)} GitHub Tools Registered ---")
else:
    print("--- WARNING: GitHub Bot failed to initialize. GitHub tools are disabled. ---")

tool_executor = ToolExecutor(tools)

# === 4. Define LLM and Agent Logic ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY,
                             convert_system_message_to_human=True) # Might help Gemini follow instructions
llm_with_tools = llm.bind_tools(tools)

# *** ADD SYSTEM PROMPT ***
system_prompt = f"""You are a helpful assistant that can interact with a GitHub repository ({GITHUB_REPO_NAME} on branch {GITHUB_BRANCH}) using the provided tools.

Available Tools:
- list_github_files: Shows files/directories in a given path (root if no path specified).
- read_github_file: Reads a file's content using its FULL path (e.g., 'folder/file.txt').
- write_github_file: Writes or updates a file using its FULL path.

IMPORTANT INSTRUCTIONS for file operations:
1. Tools like 'read_github_file' and 'write_github_file' REQUIRE the FULL path from the repository root.
2. If a user asks to read or write a file using just the filename (e.g., "read hello.md"), AND you are unsure of the full path OR the 'read_github_file' tool returns a 'File not found' error:
    a. FIRST, use the 'list_github_files' tool (you might need to specify the directory if the user gave a hint, otherwise check the root "").
    b. EXAMINE the output of 'list_github_files' to find the correct full path for the desired file.
    c. THEN, call 'read_github_file' or 'write_github_file' again using the CORRECT FULL PATH you identified.
3. If you list files and still cannot find the requested file, inform the user.
4. When writing a file, ask the user for a suitable commit message if they haven't provided one. Default to a generic message like 'Update [file_path]' if necessary.
5. Present final results clearly to the user. If showing file content, present it directly. If confirming a write, show the success message from the tool.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state: AgentState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    # If the last message is an AIMessage and it has NO tool calls, we can end.
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        print("--- Agent Decision: Last message is AI response without tool calls, finishing ---")
        return "end"
    # Otherwise, continue (agent needs to call tool, or process tool result)
    print("--- Agent Decision: Tool call requested or tool result received, continuing ---")
    return "continue"

def call_model(state: AgentState):
    """Invokes the LLM with the system prompt and current message history."""
    messages = state['messages']
    print(f"\n--- Node: Agent (Calling LLM) ---")
    # *** Inject System Prompt ***
    messages_with_system_prompt = [SystemMessage(content=system_prompt)] + messages
    print(f"Messages sent to LLM: {[m.type for m in messages_with_system_prompt]}")

    # Call LLM with the enhanced message list
    response = llm_with_tools.invoke(messages_with_system_prompt) # Use the list including system prompt

    print(f"LLM Response Type: {type(response)}")
    if hasattr(response, 'content'): print(f"LLM Content: {response.content[:100]}...")
    if hasattr(response, 'tool_calls') and response.tool_calls: print(f"LLM Tool Calls: {response.tool_calls}")

    # Return only the LLM's response message to be added to the state
    return {"messages": [response]}

# Keep the corrected call_tool function from the previous step
def call_tool(state: AgentState):
    """Executes tools based on the LLM's last message."""
    messages = state['messages']
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
         print("--- Node: Tool Executor - Warning: Last message is not an AIMessage with tool calls. Skipping tool execution. ---")
         # It's possible the graph ends here if the previous node was the final AI response.
         # Returning empty dict is okay, the should_continue node will likely catch this state next loop.
         return {} # No tools to execute in this state transition

    print(f"\n--- Node: Tool Executor ---")
    print(f"Executing tool calls: {last_message.tool_calls}")

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')

        if not tool_name or not tool_call_id:
             print(f"Skipping invalid tool call object: {tool_call}")
             continue

        selected_tool = next((t for t in tools if t.name == tool_name), None)
        if not selected_tool:
             print(f"Error: Tool '{tool_name}' not found.")
             tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' is not available.", tool_call_id=tool_call_id))
             continue

        print(f"Invoking tool: {tool_name} with args: {tool_args}")
        try:
            # Use the ToolExecutor to run the tool. ToolExecutor expects name/args.
            # Let's stick to the ToolExecutor pattern for consistency
            action = {"tool": tool_name, "tool_input": tool_args} # Simplified input for ToolExecutor
            # NOTE: ToolExecutor might internally create ToolInvocation or handle directly.
            # Let's invoke the tool directly via its .invoke method, as @tool wraps it correctly.
            response_content = selected_tool.invoke(tool_args)


            # Ensure the content added to ToolMessage is a string
            if not isinstance(response_content, str):
                print(f"Tool output type was {type(response_content)}. Converting to string.")
                try: stringified_content = json.dumps(response_content, indent=2)
                except TypeError: stringified_content = str(response_content)
            else:
                stringified_content = response_content

            print(f"Tool Response (stringified): {stringified_content}")
            tool_messages.append(ToolMessage(content=stringified_content, tool_call_id=tool_call_id))

        except Exception as e:
            print(f"Error executing tool {tool_name} with args {tool_args}: {e}")
            import traceback
            traceback.print_exc()
            tool_messages.append(ToolMessage(content=f"Error executing tool {tool_name}: {e}", tool_call_id=tool_call_id))

    return {"messages": tool_messages}


# === 5. Define the LangGraph Workflow ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent") # Always go back to the agent after action
langgraph_agent_app = workflow.compile()
print("--- LangGraph Agent Compiled ---")

# === 6. Flask Application ===
flask_app = Flask(__name__)
flask_app.secret_key = os.urandom(24)

@flask_app.route('/', methods=['GET', 'POST'])
def home():
    response_message = None
    prompt_received = None
    github_enabled = github_bot is not None and tools

    if request.method == 'POST':
        prompt_received = request.form.get('prompt')

        if not prompt_received:
            flash("Please enter a prompt.", "warning")
        else:
            print(f"\n--- Flask Request Received ---")
            print(f"User Prompt: {prompt_received}")

            if "github" in prompt_received.lower() and not github_enabled:
                 flash("GitHub tools are currently disabled.", "danger")
                 response_message = "GitHub tool interaction is disabled."
            else:
                try:
                    inputs = {"messages": [HumanMessage(content=prompt_received)]}
                    print("Invoking LangGraph agent...")
                    # *** INCREASE RECURSION LIMIT ***
                    final_state = langgraph_agent_app.invoke(inputs, {"recursion_limit": 20}) # Increased limit

                    if final_state and final_state.get('messages'):
                        final_ai_message = None
                        for msg in reversed(final_state['messages']):
                            # Find the last AIMessage that didn't just call a tool
                            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                                final_ai_message = msg
                                break

                        if final_ai_message:
                            response_message = final_ai_message.content
                        else:
                             # Fallback: show last message if no clean AI response found
                             last_message = final_state['messages'][-1]
                             response_message = f"Agent finished. Last step ({last_message.type}): {last_message.content}"
                             print(f"Warning: Agent loop ended without a final AIMessage. Last message was: {last_message}")

                    else:
                        response_message = "Agent finished unexpectedly without messages."

                    print(f"Final Response to return: {response_message}")

                except Exception as e:
                    print(f"Error during LangGraph invocation: {e}")
                    import traceback
                    traceback.print_exc()
                    flash(f"An error occurred processing your request.", "danger")
                    response_message = f"Error processing request. ({type(e).__name__})"

            print(f"--- Flask Request Finished ---")

    return render_template('index.html',
                           prompt=prompt_received,
                           response=response_message,
                           github_enabled=github_enabled)

# === 7. Run Flask App ===
if __name__ == '__main__':
    print("Starting Flask application...")
    flask_app.run(debug=True, host='0.0.0.0', port=5001) # Ensure port is correct