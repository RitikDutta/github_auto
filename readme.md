## Project Showcase: GitHub Auto

**Motivation and Problem Context:**

This project originated from the practical challenges encountered during chemical research, which involves frequent experimentation, formulation adjustments, and results analysis. While using GitHub proved effective for version controlling and logging research data (protocols, ingredients, formulations, test outcomes), the process of manually creating/updating files, writing commit messages, and pushing changes for each incremental update became repetitive and detracted from the primary research tasks. The need arose for a more streamlined method to interface with the Git repository without interrupting the experimental workflow.

**Project Description: GitHub Auto**

GitHub Auto is an AI agent developed using the LangGraph framework to address this need. Its core function is to serve as a voice-controlled interface for managing a research-focused GitHub repository. The agent interprets natural language voice commands to perform common Git and file management tasks, automating the documentation process associated with ongoing research.

**Core Functionality:**

*   **Voice Command Interface:** The primary interaction method is through voice commands, parsed by the agent to determine user intent.
*   **Repository Interaction:**
    *   **File System Navigation:** Allows users to request listings of files and directories within the repository (e.g., "List files in the 'protocols' folder").
    *   **File Content Access:** Can retrieve and present the content of specific files upon request (e.g., "Show the content of 'experiment_log_101.md'").
*   **Content Management:**
    *   **File Creation:** Creates new files based on user instructions. Notably, it incorporates contextual understanding, enabling it to place files in appropriate directories (e.g., recognizing a "new formulation" should likely go into a `formulations/` directory) and optionally use existing files as templates to maintain structural consistency.
    *   **File Updates:** Modifies specific sections or lines within existing files based on dictated changes (e.g., "In 'results_summary.txt', update the yield for reaction B to 85%").
*   **Automated Git Workflow:** Upon successful file creation or modification, the agent automates the standard Git sequence:
    *   Generates a contextually relevant commit message (e.g., "Create formulation log for Zeta-7" or "Update observations in experiment_gamma_run3.log").
    *   Stages the modified/created file(s).
    *   Commits the changes.
    *   Pushes the commit to the remote GitHub repository.

**Example Interaction Flows:**

1.  **Logging a Result:**
    *   *User:* "GitHub Auto, create a new file in 'test_results' named 'test_alpha_run2.log'. Add the content: 'Result: Pass. Observation: No precipitate formed.'"
    *   *Agent:* Creates the file, adds content, generates commit message (e.g., "Log result for test_alpha_run2"), commits, and pushes.
2.  **Updating a Protocol:**
    *   *User:* "GitHub Auto, in the file 'protocols/standard_reaction_setup.md', add 'Step 4: Verify temperature probe calibration' before the current Step 4."
    *   *Agent:* Modifies the file, generates commit message (e.g., "Update standard_reaction_setup.md with calibration step"), commits, and pushes.
3.  **Checking Repository State:**
    *   *User:* "GitHub Auto, what files are in the 'formulations' directory?"
    *   *Agent:* Lists the files present in that specific directory.

**Technical Implementation:**

*   **Agent Framework:** LangGraph is used to structure the agent's logic, manage state, and coordinate the interaction between different components (LLM, tools).
*   **Natural Language Processing:** Relies on Large Language Models (LLMs) integrated as agents within LangGraph for understanding user commands, extracting intent and parameters, and generating text (like commit messages).
*   **Voice Input:** A standard Speech-to-Text (STT) component converts spoken commands into text for the agent to process.
*   **Tooling:** The agent utilizes tools (likely Python functions or scripts) that interface with the local Git installation (via command-line execution or libraries like GitPython) to perform file system operations and Git commands ( `git add`, `git commit`, `git push`, `ls`, `cat`, etc.).

**Project Outcome:**

GitHub Auto effectively decouples the research documentation process from direct manual interaction with Git. By handling file creation, updates, and the commit/push cycle through voice commands, it allows the user to maintain their research log with significantly reduced interruption and manual effort, integrating the documentation task more seamlessly into the research workflow.