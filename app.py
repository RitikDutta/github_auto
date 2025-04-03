import os
import base64
import json
from typing import List, Optional, TypedDict, Annotated, Sequence
import operator # For state updates
import re # For potential future use

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
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Use LangChain's wrapper for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# === 1. GitHub Tool Class (Keep your existing, correct class here) ===
class Github_Auto:
    # (Paste the full Github_Auto class code here - no changes needed inside the class)
    # ... (ensure all methods like __init__, list_repository_files, get_file_content, create_or_update_file are present) ...
    """
    Manages interactions with a specific GitHub repository.
    Provides methods to authenticate, list files, read file content,
    and create or update files within the designated repository and branch.
    """
    def __init__(self, token: str, repo_name: str, branch: str = "main"):
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
        if not self.repo: return ["Error: Repository object not initialized."]
        print(f"\nTOOL: Listing files in '{self.repo_name}/{directory_path}' (branch: {self.branch})...")
        all_files = []
        try:
            contents = self.repo.get_contents(directory_path, ref=self.branch)
            content_stack = list(contents)
            while content_stack:
                content_file = content_stack.pop()
                if content_file.type == "dir":
                    try: content_stack.extend(self.repo.get_contents(content_file.path, ref=self.branch))
                    except Exception as dir_e: print(f"  Warn: Cannot list '{content_file.path}': {dir_e}")
                else:
                    all_files.append(content_file.path)
            print(f"Found {len(all_files)} files/dirs at top level of '{directory_path or '/'}'.")
            return all_files if all_files else ["No files found in the specified path."]
        except UnknownObjectException: msg = f"Error: Dir '{directory_path}' not found."; print(msg); return [msg]
        except GithubException as e: msg = f"Error listing contents: {e}"; print(msg); return [msg]
        except Exception as e: msg = f"Unexpected error listing files: {e}"; print(msg); return [msg]


    def get_file_content(self, file_path: str) -> str:
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
        if not self.repo: return "Error: Repository object not initialized."
        print(f"\nTOOL: Attempt create/update: {file_path} on branch '{self.branch}'")
        current_sha = None
        try:
            existing_file = self.repo.get_contents(file_path, ref=self.branch)
            if not isinstance(existing_file, list) and existing_file.type == 'file':
                current_sha = existing_file.sha
                print(f"File '{file_path}' exists (SHA: {current_sha}). Updating.")
            else:
                 msg = f"Warning: Path '{file_path}' exists but structure is unexpected. Attempting create/update cautiously."
                 print(msg)
        except UnknownObjectException:
            print(f"File '{file_path}' not exist. Creating."); current_sha = None
        except GithubException as e:
            msg = f"GitHub error checking exist '{file_path}': {e}"; print(msg); return msg
        except Exception as e:
             msg = f"Unexpected error checking '{file_path}': {e}"; print(msg); return msg
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
        except GithubException as e:
             msg = f"Error GitHub op {action} for '{file_path}': {e}"; print(msg)
             if e.status == 409: msg += " (Conflict? Branch changed?)"
             elif e.status == 422 and e.data: msg += f" (Validation: {e.data.get('message', 'Unknown')})"
             return msg
        except Exception as e: msg = f"Unexpected error {action} for '{file_path}': {e}"; print(msg); return msg


# === 2. Configuration and Initialization ===
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# *** Define Expected Directory Structure ***
# This helps the LLM determine the correct path
DIRECTORY_STRUCTURE = {
    "ingredient": "docs/ingredients",
    "formulation": "docs/formulations",
    "test_result": "data/results",
    "default": "docs" # Fallback directory
}
BASE_TEMPLATE_PATH = "base_template.md" # Path to your template file

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
        """Reads the content of a specific file using its full path from the configured GitHub repository. Can be used to read template files."""
        return github_bot.get_file_content(file_path)

    @tool
    def write_github_file(file_path: str, content: str, commit_message: str) -> str:
        """Creates a new file OR completely overwrites an existing file (using its full path) with the provided content in the configured GitHub repository. Requires a commit message."""
        return github_bot.create_or_update_file(file_path, content, commit_message)

    # Note: update_file_section tool is kept from previous step, can be used if needed
    @tool
    def update_file_section(file_path: str, target_section_identifier: str, new_content_for_section: str, commit_message: str) -> str:
        """Updates a specific section within an existing file. Reads the file, finds the *first line* containing the 'target_section_identifier' string, replaces that entire line with 'new_content_for_section', and commits the change. Fails if the file or identifier is not found. Requires a commit message."""
        print(f"\nTOOL: Attempting partial update for: {file_path}")
        current_content = github_bot.get_file_content(file_path)
        if current_content.startswith("Error:"):
            print(f"Partial update failed: Could not read file. Error: {current_content}")
            return f"Error updating section: Could not read file '{file_path}'. Details: {current_content}"
        lines = current_content.splitlines(); found = False; modified_lines = []
        for line in lines:
            if not found and target_section_identifier in line:
                modified_lines.append(new_content_for_section); found = True
                print(f"Found target section identifier '{target_section_identifier}' and replaced line.")
            else: modified_lines.append(line)
        if not found: msg = f"Error updating section: Identifier '{target_section_identifier}' not found in '{file_path}'."; print(msg); return msg
        modified_content = "\n".join(modified_lines)
        print(f"Content modified. Committing partial update for {file_path}...")
        return github_bot.create_or_update_file(file_path, modified_content, commit_message)

    tools = [list_github_files, read_github_file, write_github_file, update_file_section]
    print(f"--- {len(tools)} GitHub Tools Registered ---")
else:
    print("--- WARNING: GitHub Bot failed to initialize. GitHub tools are disabled. ---")

tool_executor = ToolExecutor(tools)

# === 4. Define LLM and Agent Logic ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY,
                             convert_system_message_to_human=True)
llm_with_tools = llm.bind_tools(tools)

# *** SIGNIFICANTLY UPDATED SYSTEM PROMPT ***
system_prompt = f"""You are a helpful assistant managing a GitHub repository ({GITHUB_REPO_NAME} on branch {GITHUB_BRANCH}) using tools.

Available Tools:
- list_github_files(directory_path): Shows files/dirs in a path (root if "").
- read_github_file(file_path): Reads a file's content using FULL path. Crucial for reading the template file.
- write_github_file(file_path, content, commit_message): Creates/Overwrites a file with FULL path, CONTENT, and commit message.
- update_file_section(file_path, target_section_identifier, new_content_for_section, commit_message): Updates a SINGLE line in an existing file. Requires full path, identifier on the line, the new full line content, and commit message.

*** Key Procedures ***

1.  **File Paths:** ALWAYS use full paths from the repository root (e.g., 'docs/ingredients/argan_oil.md').
2.  **Finding Files:** If unsure of a path or `read_github_file` fails, use `list_github_files` to find the correct full path before trying `read_github_file` or `write_github_file` again.
3.  **Commit Messages:** For `write_github_file` or `update_file_section`, if the user doesn't provide a commit message, GENERATE a concise one reflecting the action (e.g., "feat: Add ingredient [name]", "docs: Update [file] structure", "fix: Correct value in [file]"). Include it when calling the tool.
4.  **Creating NEW Structured Files (e.g., ingredients, formulations):**
    a.  **Identify File Type & Name:** Determine the type (e.g., "ingredient") and the specific name (e.g., "test oil"). Create a filename (e.g., `test_oil.md`). All new structured files MUST be Markdown (`.md`).
    b.  **Determine Directory:** Use the following structure to decide the FULL path:
        - Ingredients: `{DIRECTORY_STRUCTURE['ingredient']}/[filename].md`
        - Formulations: `{DIRECTORY_STRUCTURE['formulation']}/[filename].md`
        - Test Results: `{DIRECTORY_STRUCTURE['test_result']}/[filename].md`
        - Other/Unspecified: `{DIRECTORY_STRUCTURE['default']}/[filename].md`
        Example: For an ingredient named "Test Oil", the path is `{DIRECTORY_STRUCTURE['ingredient']}/test_oil.md`.
    c.  **Read Template:** Call `read_github_file` to get the content of the base template file located at `{BASE_TEMPLATE_PATH}`.
    d.  **Generate Content:** Once you have the template content (from the previous step's tool result) AND the user's details for the new item:
        - Understand the structure (headings like # Name, ## Pros, ## Cons) from the template content.
        - **GENERATE THE COMPLETE MARKDOWN CONTENT** for the new file. Replace placeholders (like '[Item Name Placeholder]') and fill in the sections (Pros, Cons, etc.) using the user's provided details, maintaining the template's heading structure.
    e.  **Write File:** Call `write_github_file` with the determined full path, the **complete generated Markdown content**, and a generated commit message.
    f.  **Confirm:** Report success (including commit SHA from the tool output) or failure to the user.

5.  **Updating Sections (`update_file_section`):** Use this ONLY for modifying a *specific existing line*. Provide a unique string from that line as `target_section_identifier` and the complete new line content as `new_content_for_section`. Generate a commit message.
6.  **Clarity:** Respond clearly, confirming actions and presenting results (like file content or commit SHAs).
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# should_continue function remains the same
def should_continue(state: AgentState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        print("--- Agent Decision: Last message is AI response without tool calls, finishing ---")
        return "end"
    print("--- Agent Decision: Tool call requested or tool result received, continuing ---")
    return "continue"

# call_model function remains the same (injects system prompt)
def call_model(state: AgentState):
    messages = state['messages']
    print(f"\n--- Node: Agent (Calling LLM) ---")
    # Always include the system prompt
    messages_with_system_prompt = [SystemMessage(content=system_prompt)] + messages
    print(f"Messages sent to LLM: {[m.type for m in messages_with_system_prompt]}")
    response = llm_with_tools.invoke(messages_with_system_prompt)
    print(f"LLM Response Type: {type(response)}")
    if hasattr(response, 'content'): print(f"LLM Content: {response.content[:100]}...")
    if hasattr(response, 'tool_calls') and response.tool_calls: print(f"LLM Tool Calls: {response.tool_calls}")
    return {"messages": [response]}


# call_tool function remains the same (handles stringifying tool output)
def call_tool(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
         print("--- Node: Tool Executor - Warning: Last message no tool calls. Skip.")
         return {}

    print(f"\n--- Node: Tool Executor ---")
    print(f"Executing tool calls: {last_message.tool_calls}")
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')
        if not tool_name or not tool_call_id: print(f"Skip invalid tool: {tool_call}"); continue

        selected_tool = next((t for t in tools if t.name == tool_name), None)
        if not selected_tool:
             print(f"Error: Tool '{tool_name}' not found.")
             tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' unavailable.", tool_call_id=tool_call_id))
             continue

        print(f"Invoking tool: {tool_name} with args: {tool_args}")
        try:
            # Invoke the tool using the @tool wrapper's logic
            response_content = selected_tool.invoke(tool_args)
            if not isinstance(response_content, str):
                print(f"Tool type {type(response_content)}. Stringify.")
                try: stringified_content = json.dumps(response_content, indent=2)
                except TypeError: stringified_content = str(response_content)
            else: stringified_content = response_content
            print(f"Tool Response (stringified): {stringified_content}")
            tool_messages.append(ToolMessage(content=stringified_content, tool_call_id=tool_call_id))
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            import traceback; traceback.print_exc()
            tool_messages.append(ToolMessage(content=f"Error tool {tool_name}: {e}", tool_call_id=tool_call_id))
    return {"messages": tool_messages}


# === 5. Define the LangGraph Workflow (remains the same) ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
langgraph_agent_app = workflow.compile()
print("--- LangGraph Agent Compiled ---")

# === 6. Flask Application (remains largely the same, check recursion limit) ===
flask_app = Flask(__name__)
flask_app.secret_key = os.urandom(24)

@flask_app.route('/', methods=['GET', 'POST'])
def home():
    response_message = None
    prompt_received = None
    github_enabled = github_bot is not None and tools

    if request.method == 'POST':
        prompt_received = request.form.get('prompt')
        if not prompt_received: flash("Please enter a prompt.", "warning")
        else:
            print(f"\n--- Flask Request Received ---")
            print(f"User Prompt: {prompt_received}")
            if "github" in prompt_received.lower() and not github_enabled:
                 flash("GitHub tools disabled.", "danger")
                 response_message = "GitHub tool interaction disabled."
            else:
                try:
                    inputs = {"messages": [HumanMessage(content=prompt_received)]}
                    print("Invoking LangGraph agent...")
                    # Increased recursion limit for potentially longer template process
                    final_state = langgraph_agent_app.invoke(inputs, {"recursion_limit": 30})

                    if final_state and final_state.get('messages'):
                        final_ai_message = None
                        for msg in reversed(final_state['messages']):
                            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                                final_ai_message = msg; break
                        if final_ai_message: response_message = final_ai_message.content
                        else:
                             last_message = final_state['messages'][-1]
                             response_message = f"Agent finished. Last step ({last_message.type}): {last_message.content}"
                             print(f"Warn: Agent loop end no final AIMessage. Last: {last_message}")
                    else: response_message = "Agent finished unexpectedly."
                    print(f"Final Response to return: {response_message}")
                except Exception as e:
                    print(f"Error during LangGraph invocation: {e}")
                    import traceback; traceback.print_exc()
                    flash(f"An error occurred processing.", "danger")
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