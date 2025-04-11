import os
import base64
import json
from typing import List, Optional, TypedDict, Annotated, Sequence
import operator
import re
import time # For SSE stream keep-alive (optional)
import traceback # For detailed error logging

from flask import Flask, request, render_template, flash, Response, stream_with_context
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
    # (Paste the full Github_Auto class code here)
    """ Manages interactions with a specific GitHub repository. """
    def __init__(self, token: str, repo_name: str, branch: str = "main"):
        self.token = token; self.repo_name = repo_name; self.branch = branch
        self.github_instance: Optional[Github] = None; self.repo = None
        if not token: raise ValueError("GitHub token required.")
        if not repo_name: raise ValueError("Repo name required.")
        try:
            print("Auth GitHub..."); self.github_instance = Github(self.token, timeout=30)
            user = self.github_instance.get_user(); print(f"Auth OK: {user.login}")
            print(f"Accessing repo: {self.repo_name}"); self.repo = self.github_instance.get_repo(self.repo_name)
            print(f"Accessed '{self.repo.full_name}', branch '{self.branch}'")
            try: self.repo.get_branch(self.branch); print("Target branch confirmed.")
            except UnknownObjectException: print(f"WARN: Target branch '{self.branch}' nonexistent.")
        except BadCredentialsException: print("ERR: Invalid GitHub token."); raise
        except UnknownObjectException: print(f"ERR: Repo '{self.repo_name}' not found/access denied."); raise
        except GithubException as e: print(f"ERR: GitHub API: {e}"); raise
        except Exception as e: print(f"ERR: Init: {e}"); raise

    def list_repository_files(self, directory_path: str = "") -> List[str]:
        if not self.repo: return ["Error: Repo object uninitialized."]
        print(f"\nTOOL: List files: '{directory_path or '/'}'..."); all_files = []
        try:
            contents = self.repo.get_contents(directory_path, ref=self.branch); content_stack = list(contents)
            while content_stack:
                item = content_stack.pop()
                if item.type == "dir":
                    try: content_stack.extend(self.repo.get_contents(item.path, ref=self.branch))
                    except Exception as e: print(f"  Warn: Cannot list '{item.path}': {e}")
                else: all_files.append(item.path)
            print(f"Found {len(all_files)} items in '{directory_path or '/'}'.")
            return all_files if all_files else ["No files found."]
        except UnknownObjectException: msg = f"ERR: Dir not found: '{directory_path}'."; print(msg); return [msg]
        except GithubException as e: msg = f"ERR: List files: {e}"; print(msg); return [msg]
        except Exception as e: msg = f"ERR: Unexpected list error: {e}"; print(msg); return [msg]

    def get_file_content(self, file_path: str) -> str:
        if not self.repo: return "Error: Repo object uninitialized."
        print(f"\nTOOL: Read file: {file_path}...");
        try:
            item = self.repo.get_contents(file_path, ref=self.branch)
            if isinstance(item, list): msg = f"ERR: Path is dir: '{file_path}'."; print(msg); return msg
            if item.type != 'file': msg = f"ERR: Path not file: '{file_path}'."; print(msg); return msg
            if item.content: content = base64.b64decode(item.content).decode('utf-8'); print("Success read."); return content
            else: print("File empty."); return ""
        except UnknownObjectException: msg = f"Error: File not found at '{file_path}' on branch '{self.branch}'."; print(msg); return msg # Exact error match
        except GithubException as e: msg = f"ERR: Read file GH: {e}"; print(msg); return msg
        except Exception as e: msg = f"ERR: Unexpected read error: {e}"; print(msg); return msg

    def create_or_update_file(self, file_path: str, content: str, commit_message: str) -> str:
        if not self.repo: return "Error: Repo object uninitialized."
        print(f"\nTOOL: Write file: {file_path}..."); sha = None
        try:
            item = self.repo.get_contents(file_path, ref=self.branch)
            if not isinstance(item, list) and item.type == 'file': sha = item.sha; print("File exists, updating.")
            else: print(f"WARN: Path exists but not file: '{file_path}'.")
        except UnknownObjectException: print("File not exist, creating."); sha = None
        except GithubException as e: msg = f"ERR: Check exists GH: {e}"; print(msg); return msg
        except Exception as e: msg = f"ERR: Unexpected check error: {e}"; print(msg); return msg
        try:
            action = "(update)" if sha else "(create)"; msg = f"{commit_message} {action}"
            if sha: resp = self.repo.update_file(path=file_path, message=msg, content=content, sha=sha, branch=self.branch)
            else: resp = self.repo.create_file(path=file_path, message=msg, content=content, branch=self.branch)
            success_msg = f"Success {action} '{file_path}'. Commit: {resp['commit'].sha}"
            print(success_msg); return success_msg
        except GithubException as e: msg = f"ERR: Write GH op {action}: {e}"; print(msg); return msg
        except Exception as e: msg = f"ERR: Unexpected write error: {e}"; print(msg); return msg

    def update_file_section(self, file_path: str, target_section_identifier: str, new_content_for_section: str, commit_message: str) -> str:
        print(f"\nTOOL: Update section: {file_path}, target: '{target_section_identifier}'")
        content = self.get_file_content(file_path)
        if content.startswith("Error:"): return f"ERR update: Cannot read file. {content}"
        lines = content.splitlines(); found = False; modified = []
        for line in lines:
            if not found and target_section_identifier in line:
                modified.append(new_content_for_section); found = True; print("Target line found & replaced.")
            else: modified.append(line)
        if not found: msg = f"ERR update: Target '{target_section_identifier}' not found in '{file_path}'."; print(msg); return msg
        mod_content = "\n".join(modified); print("Committing section update...")
        return self.create_or_update_file(file_path, mod_content, commit_message)


# === 2. Configuration and Initialization (Same as before) ===
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
DIRECTORY_STRUCTURE = { "ingredient": "docs/ingredients", "formulation": "docs/formulations", "test_result": "data/results", "default": "docs" }
BASE_TEMPLATE_PATH = "base_template.md"
if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY missing.")
if not GITHUB_TOKEN: raise ValueError("GITHUB_TOKEN missing.")
if not GITHUB_REPO_NAME: raise ValueError("GITHUB_REPO_NAME missing.")
try:
    print("\n--- Initializing Github Bot ---")
    github_bot = Github_Auto(token=GITHUB_TOKEN, repo_name=GITHUB_REPO_NAME, branch=GITHUB_BRANCH)
    print("--- Github Bot Initialized Successfully ---")
except Exception as e: print(f"FATAL: Failed to initialize Github_Auto: {e}"); github_bot = None

# === 3. Define LangGraph Tools (Same as before) ===
tools = []
if github_bot:
    @tool
    def list_github_files(directory_path: str = "") -> List[str]:
        """Lists files/directories within a specific directory path (default is root). Does not recurse."""
        return github_bot.list_repository_files(directory_path)
    @tool
    def read_github_file(file_path: str) -> str:
        """Reads the content of a specific file using its full path. Returns 'Error: File not found...' if the path is invalid."""
        return github_bot.get_file_content(file_path)
    @tool
    def write_github_file(file_path: str, content: str, commit_message: str) -> str:
        """Creates/Overwrites a file with the provided full path, content, and commit message."""
        return github_bot.create_or_update_file(file_path, content, commit_message)
    @tool
    def update_file_section(file_path: str, target_section_identifier: str, new_content_for_section: str, commit_message: str) -> str:
        """Updates a SINGLE line in an existing file identified by 'target_section_identifier'. Requires full path, identifier, new line content, commit message."""
        return github_bot.update_file_section(file_path, target_section_identifier, new_content_for_section, commit_message)
    tools = [list_github_files, read_github_file, write_github_file, update_file_section]
    print(f"--- {len(tools)} GitHub Tools Registered ---")
else: print("--- WARNING: GitHub Bot failed to initialize. GitHub tools are disabled. ---")
tool_executor = ToolExecutor(tools)

# === 4. Define LLM and Agent Logic ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
llm_with_tools = llm.bind_tools(tools)

# --- System Prompt (Keep the refined version from previous step) ---
system_prompt = f"""You are a helpful assistant managing a GitHub repository ({GITHUB_REPO_NAME} on branch {GITHUB_BRANCH}) using tools.

Available Tools:
- list_github_files(directory_path): Shows files/dirs in a path (root if "").
- read_github_file(file_path): Reads a file's content using FULL path. Returns 'Error: File not found...' if path is invalid.
- write_github_file(file_path, content, commit_message): Creates/Overwrites a file with FULL path, CONTENT, and commit message.
- update_file_section(file_path, target_section_identifier, new_content_for_section, commit_message): Updates a SINGLE line in an existing file. Requires full path, identifier on the line, the new full line content, and commit message.

*** Key Procedures ***

1.  **File Paths:** ALWAYS use full paths from the repository root (e.g., 'docs/ingredients/argan_oil.md'). Add the `.md` extension for structured content files unless specified otherwise.
2.  **Finding Files (CRITICAL):**
    *   When asked to read/write/update a file and the user provides only a filename (e.g., "read test3.md") or an ambiguous path:
        a.  **First, TRY** the most likely path based on file type (see Directory Structure below) or user context. Call the appropriate tool (e.g., `read_github_file('docs/ingredients/test3.md')`).
        b.  **If the tool returns an error message containing 'Error: File not found'**: This means your path guess was wrong or the file doesn't exist there. **DO NOT immediately ask the user for the path.**
        c.  **Instead, your VERY NEXT action MUST be** to call `list_github_files()` (or `list_github_files('relevant_directory/')` if you have a good guess which directory it *might* be in).
        d.  **Examine the list of files returned by `list_github_files`.** Search for the filename the user requested (e.g., 'test3.md').
        e.  **If you find EXACTLY ONE match** in the list (e.g., you find `docs/ingredients/test3.md`), THEN call the original requested tool (`read_github_file`, `write_github_file`, etc.) again using that **CORRECT FULL PATH**.
        f.  **If you find MULTIPLE matches** (e.g., 'test/test3.md' and 'docs/test3.md') OR **NO match** after listing, THEN you should inform the user that the file wasn't found or is ambiguous, and optionally provide the file list you found to help them clarify.
    *   **Example Recovery Flow:**
        - User: "read test3 file"
        - You (Agent): Try likely path -> call `read_github_file(file_path='docs/ingredients/test3.md')`
        - Tool Result: "Error: File not found at 'docs/ingredients/test3.md'..."
        - You (Agent): File not found, must list files -> call `list_github_files(directory_path='')`
        - Tool Result: `['README.md', 'docs/ingredients/test3.md', 'src/utils.py']` (stringified JSON)
        - You (Agent): Found unique match `docs/ingredients/test3.md` -> call `read_github_file(file_path='docs/ingredients/test3.md')`
        - Tool Result: (File content)
        - You (Agent): Respond to user with the content.

3.  **Commit Messages:** For `write_github_file` or `update_file_section`, if the user doesn't provide one, GENERATE a concise message (e.g., "feat: Add [filename]", "docs: Update [filename]").
4.  **Creating NEW Structured Files:**
    a.  Identify Type & Name -> Filename (always `.md`).
    b.  Determine Directory using this structure: Ingredients -> `{DIRECTORY_STRUCTURE['ingredient']}`, Formulations -> `{DIRECTORY_STRUCTURE['formulation']}`, Test Results -> `{DIRECTORY_STRUCTURE['test_result']}`, Other -> `{DIRECTORY_STRUCTURE['default']}`. Construct the FULL path.
    c.  Read Template: Call `read_github_file(file_path='{BASE_TEMPLATE_PATH}')`.
    d.  Generate Content: Use the template structure from the tool result and user details to generate the full Markdown content for the new file.
    e.  Write File: Call `write_github_file` with the full path, generated content, and generated commit message.
    f.  Confirm success/failure to user.

5.  **Updating Sections (`update_file_section`):** Use ONLY for modifying a specific line. Provide unique text from the target line as `target_section_identifier`, and the complete new line content as `new_content_for_section`. Generate commit message.
6.  **Clarity:** Confirm actions, present results/content clearly. Report errors.
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Agent Nodes (call_model, call_tool, should_continue) - Keep implementations from previous step
# Ensure they handle system prompt injection and stringifying tool output correctly.

def should_continue(state: AgentState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        print("--- Agent Decision: Last message is AI response without tool calls, finishing ---")
        return "end"
    print("--- Agent Decision: Tool call requested or tool result received, continuing ---")
    return "continue"

def call_model(state: AgentState):
    messages = state['messages']
    print(f"\n--- Node: Agent (Calling LLM) ---")
    messages_with_system_prompt = [SystemMessage(content=system_prompt)] + messages
    print(f"Messages sent to LLM: {[m.type for m in messages_with_system_prompt]}")
    try:
        response = llm_with_tools.invoke(messages_with_system_prompt)
        print(f"LLM Response Type: {type(response)}")
        if hasattr(response, 'content'): print(f"LLM Content (trunc): {response.content[:100]}...")
        if hasattr(response, 'tool_calls') and response.tool_calls: print(f"LLM Tool Calls: {response.tool_calls}")
        return {"messages": [response]}
    except Exception as e:
         print(f"LLM Invocation Error: {e}"); traceback.print_exc()
         error_response = AIMessage(content=f"Error invoking LLM: {e}")
         return {"messages": [error_response]} # Return error as AI message

def call_tool(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
         print("--- Node: Tool Executor - Warning: Last message no tool calls. Skip.")
         return {}
    print(f"\n--- Node: Tool Executor ---"); print(f"Executing tool calls: {last_message.tool_calls}")
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name'); tool_args = tool_call.get('args', {}); tool_call_id = tool_call.get('id')
        if not tool_name or not tool_call_id: print(f"Skip invalid tool: {tool_call}"); continue
        selected_tool = next((t for t in tools if t.name == tool_name), None)
        if not selected_tool:
             print(f"Error: Tool '{tool_name}' not found.")
             tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' unavailable.", tool_call_id=tool_call_id))
             continue
        print(f"Invoking tool: {tool_name} with args: {tool_args}")
        try:
            response_content = selected_tool.invoke(tool_args)
            if not isinstance(response_content, str):
                print(f"Tool type {type(response_content)}. Stringify.")
                try: stringified_content = json.dumps(response_content, indent=2)
                except TypeError: stringified_content = str(response_content)
            else: stringified_content = response_content
            print(f"Tool Response (stringified): {stringified_content[:500]}...") # Truncate long responses in log
            tool_messages.append(ToolMessage(content=stringified_content, tool_call_id=tool_call_id))
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}"); traceback.print_exc()
            tool_messages.append(ToolMessage(content=f"Error tool {tool_name}: {e}", tool_call_id=tool_call_id))
    return {"messages": tool_messages}


# === 5. Define the LangGraph Workflow (Same as before) ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
try:
    langgraph_agent_app = workflow.compile()
    print("--- LangGraph Agent Compiled ---")
except Exception as compile_error: print(f"FATAL: LangGraph compile failed: {compile_error}"); langgraph_agent_app = None


# === 6. Flask Application - SSE Implementation ===
flask_app = Flask(__name__)
flask_app.secret_key = os.urandom(24) # For flashing messages (optional)



# SSE route to stream agent execution
@flask_app.route('/', methods=['GET'])
def index():
    github_enabled = github_bot is not None and tools
    agent_ready = langgraph_agent_app is not None
    return render_template('index.html',
                           prompt=None,
                           response=None,
                           github_enabled=github_enabled,
                           agent_ready=agent_ready)

# SSE route to stream agent execution
@flask_app.route('/agent_stream')
def agent_stream():
    # Wrap the entire stream logic setup in a try block to ensure a response is returned
    try:
        prompt = request.args.get('prompt', '') # Default to empty string if not provided
        if not prompt:
            print("SSE Error: No prompt provided in request.")
            # Immediately return an error response
            return Response(f"data: {json.dumps({'type': 'error', 'message': 'No prompt provided.'})}\n\n", mimetype='text/event-stream')

        if not langgraph_agent_app:
             print("SSE Error: Agent not initialized.")
             # Immediately return an error response
             return Response(f"data: {json.dumps({'type': 'error', 'message': 'Agent not initialized.'})}\n\n", mimetype='text/event-stream')

        # Define the generator function *inside* the main function's scope
        # or ensure it captures necessary variables if defined outside
        def generate_sse_events():
            try:
                print(f"\n--- SSE Stream Started for prompt: {prompt[:50]}... ---")
                inputs = {"messages": [HumanMessage(content=prompt)]}
                final_state_messages = [] # Store messages to extract final response
                recursion_depth = 0
                max_recursion = 30

                # Stream the graph execution
                for event in langgraph_agent_app.stream(inputs, {"recursion_limit": max_recursion}):
                    recursion_depth += 1
                    # print(f"DEBUG SSE Event: {event}")

                    status_update = {"type": "status", "message": "Processing..."}
                    log_event_simple = {"type": "log", "data": "Step executed."} # Default log
                    node_name = list(event.keys())[0]
                    log_event_simple["data"] = f"Node '{node_name}' running..." # Update log


                    if node_name == 'agent':
                         status_update["message"] = "Agent: Thinking..."
                         agent_output = event.get('agent', {})
                         messages = agent_output.get('messages', [])
                         if messages:
                             last_msg = messages[-1]
                             if isinstance(last_msg, AIMessage):
                                 if getattr(last_msg, 'tool_calls', None):
                                     tool_names = [tc['name'] for tc in last_msg.tool_calls]
                                     status_update["message"] = f"Agent: Requesting tool(s) - {', '.join(tool_names)}"
                                     log_event_simple["data"] = f"Agent requesting tools: {', '.join(tool_names)}"
                                 else:
                                     status_update["message"] = "Agent: Formulating final response..."
                                     log_event_simple["data"] = "Agent formulating final response."

                    elif node_name == 'action':
                         status_update["message"] = "Action: Processing tool results..."
                         log_event_simple["data"] = "Action node processing results."
                         action_output = event.get('action', {})
                         messages = action_output.get('messages', [])
                         if messages:
                             tool_msgs_summary = []
                             for msg in messages:
                                  if isinstance(msg, ToolMessage):
                                      content_summary = msg.content[:100] + ('...' if len(msg.content)>100 else '')
                                      tool_msgs_summary.append(f"Tool Result ({msg.tool_call_id[:6]}): {content_summary}")
                             if tool_msgs_summary:
                                  status_update["message"] = "; ".join(tool_msgs_summary)
                                  log_event_simple["data"] = f"Tool results processed: {len(tool_msgs_summary)} message(s)."


                    yield f"data: {json.dumps(status_update)}\n\n"
                    yield f"data: {json.dumps(log_event_simple)}\n\n"

                    # Update message history
                    node_output = event[node_name]
                    if isinstance(node_output, dict) and 'messages' in node_output:
                        final_state_messages.extend(node_output['messages'])

                # --- Stream finished ---
                print("--- SSE Stream: Graph execution finished ---")
                final_response_content = "Agent finished, but no final response found."
                if final_state_messages:
                    final_ai_message = None
                    for msg in reversed(final_state_messages):
                        if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                            final_ai_message = msg; break
                    if final_ai_message: final_response_content = final_ai_message.content
                    else:
                        last_msg = final_state_messages[-1]
                        final_response_content = f"Agent finished. Last step result ({last_msg.type}): {last_msg.content}"
                        print(f"Warn: Agent loop end no final AIMessage. Last: {last_message}")

                completion_event = {"type": "complete", "final_response": final_response_content}
                yield f"data: {json.dumps(completion_event)}\n\n"
                print(f"--- SSE Stream: Sent completion event. ---")

            except Exception as e:
                print(f"Error during agent stream processing: {e}")
                traceback.print_exc()
                error_event = {"type": "error", "message": f"An error occurred during processing: {type(e).__name__}"}
                # Yield the error back to the client via SSE
                yield f"data: {json.dumps(error_event)}\n\n"

        # *** Return the Response object wrapping the generator ***
        return Response(generate_sse_events(), mimetype='text/event-stream')

    except Exception as e:
        # Catch errors during the *initial setup* of the stream (before generate_sse_events starts)
        print(f"Error setting up SSE stream: {e}")
        traceback.print_exc()
        # Return an error response directly if setup fails
        # Ensure this is also a valid Response object
        error_msg = f"Failed to start agent stream: {type(e).__name__}"
        return Response(f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n", mimetype='text/event-stream', status=500)


# === 7. Run Flask App (Conditional) ===
if __name__ == '__main__':
    if langgraph_agent_app and github_bot: # Check both agent and bot
        print("Starting Flask application...")
        flask_app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        message = "Flask application will not start because "
        if not langgraph_agent_app: message += "the LangGraph agent failed to compile. "
        if not github_bot: message += "the GitHub connection failed. "
        print(message)
