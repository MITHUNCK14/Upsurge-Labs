# main.py (Combined CLI with Agent Execution and Delete Confirmation)

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import traceback
import sys
import shutil # For potential future use like terminal width, though not used here yet

# --- Add project root to Python path ---
# Necessary if agent/tools are in subdirectories relative to main.py
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# --- ---


# --- Define Output Directory ---
# This path MUST be consistent with any paths used inside your agent/tools
try:
    OUTPUT_DIR = Path("outputs").resolve()
    print(f"DEBUG: Output directory set to: {OUTPUT_DIR}")
except Exception as e:
    print(f"CRITICAL ERROR setting OUTPUT_DIR: {e}")
    sys.exit(1)
# --- ---

# --- Load Environment Variables ---
print("DEBUG: Loading environment variables from .env file...")
# Explicitly check path for .env in the project root
dotenv_path = Path(project_root) / '.env'
dotenv_loaded = load_dotenv(dotenv_path=dotenv_path)
if not dotenv_loaded:
    print(f"DEBUG: .env file not found at {dotenv_path} or failed to load.")
else:
    print(f"DEBUG: .env file loaded successfully from {dotenv_path}.")
# --- ---

# --- Import Agent Logic and Tools AFTER loading .env ---
print("DEBUG: Attempting to import agent modules...")
try:
    # Ensure your planner uses the correct LLM initialization based on your .env
    from agent.planner import initialize_agent
    # --- Import the actual delete function (used for confirmation) ---
    # Make sure this tool exists and implements the required logic
    from tools.delete_file_tool import perform_delete
    print("DEBUG: Agent and tool modules imported successfully.")
    # --- Import necessary LangChain exceptions ---
    from langchain_core.exceptions import OutputParserException
except ImportError as e:
    print(f"\n--- Error Importing Modules ---")
    print(f"ERROR: {e}")
    print("Please ensure 'agent/planner.py' and 'tools/delete_file_tool.py' exist and that all dependencies (like langchain, python-dotenv) are installed in your virtual environment.")
    print(f"Project Root (added to sys.path): {project_root}")
    traceback.print_exc() # Print detailed traceback for import errors
    sys.exit(1) # Exit if essential modules can't be loaded
except Exception as e:
    print(f"\n--- Unexpected Error During Import ---")
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
# --- ---


# --- Check for Necessary API Key (Copied from app.py logic) ---
def check_api_key():
    """Checks for common LLM API keys based on agent code introspection."""
    llm_provider = "API"
    api_key_needed = None
    try:
        # Basic check based on function code content (adjust if needed)
        planner_code = initialize_agent.__code__.co_code
        if b'Google' in planner_code or b'gemini' in planner_code:
            api_key_needed = "GOOGLE_API_KEY"
            llm_provider = "Google Gemini"
        elif b'HuggingFace' in planner_code or b'hf_' in planner_code:
            api_key_needed = "HUGGINGFACEHUB_API_TOKEN"
            llm_provider = "Hugging Face"
        elif b'OpenAI' in planner_code or b'sk-' in planner_code:
            api_key_needed = "OPENAI_API_KEY"
            llm_provider = "OpenAI"
        # Add checks for other providers if necessary
    except Exception as inspect_err:
        print(f"DEBUG: Could not inspect agent code for API keys: {inspect_err}")
        pass # Ignore errors during introspection

    if api_key_needed:
        print(f"DEBUG: Checking for environment variable: {api_key_needed} (Detected Provider: {llm_provider})")
        if not os.getenv(api_key_needed):
            print(f"\n🚨 ERROR: {llm_provider} API key not found! 🚨")
            print(f"   Please ensure the environment variable '{api_key_needed}' is set.")
            print(f"   Consider setting it in your '{dotenv_path}' file.")
            return False
        else:
            print(f"DEBUG: Found API key for {llm_provider}.")
    else:
        print("DEBUG: No specific API key check triggered based on agent code inspection.")
    return True
# --- ---


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Agent (Command Line Interface)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Use default formatter
    )
    parser.add_argument(
        "instruction",
        type=str,
        help="Natural language instruction for the agent (e.g., 'Summarize outputs/report.txt')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", # Sets args.verbose to True if flag is present
        default=False,       # Default is False
        help="Enable verbose logging from the agent's execution steps"
    )
    args = parser.parse_args()

    print(f"\n--- Instruction Received ---\n'{args.instruction}'\n")
    print(f"DEBUG: Verbose mode requested: {args.verbose}")

    # --- API Key Check ---
    if not check_api_key():
        sys.exit(1) # Exit if required key is missing

    # Ensure output directory exists
    print(f"DEBUG: Checking/creating output directory: {OUTPUT_DIR}")
    if not OUTPUT_DIR.exists():
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            print(f"ERROR: Could not create output directory {OUTPUT_DIR}: {e}")
            # Depending on the task, you might want to exit here
            # sys.exit(1)

    agent_executor = None # Initialize agent_executor to None
    try:
        print("\n--- Initializing Agent ---")
        # Initialize the LangChain agent executor
        # Pass verbose flag from CLI args to the initializer
        agent_executor = initialize_agent(verbose=args.verbose)
        if agent_executor:
             print("--- Agent Initialized Successfully ---")
        else:
             # This case should ideally be handled by an exception in initialize_agent
             print("ERROR: Agent initialization returned None unexpectedly!")
             sys.exit(1) # Exit if initialization failed critically

        print("\n--- Starting Agent Execution... ---")
        # Run the agent with the user's instruction
        # The agent's internal verbose logging (if enabled) should print here
        result = agent_executor.invoke({"input": args.instruction})
        print("--- Agent Execution Attempt Finished ---") # Mark when invoke returns

        if not result:
             print("WARNING: Agent invocation returned None or empty result.")
             # Decide how to proceed, maybe exit or just report
             raw_agent_output = "Agent returned no result." # Provide a default message
        else:
             print(f"DEBUG: Raw agent result type: {type(result)}")
             print(f"DEBUG: Raw agent result content: {result}")

             # Get the raw output from the agent - handle if result is not a dict
             # (Adapting logic from the provided main.py)
             if isinstance(result, dict):
                  # Common LangChain agent structure
                  raw_agent_output = result.get('output', 'Agent finished, but no specific "output" key found in result dictionary.')
             elif isinstance(result, str):
                  # Some simpler chains/agents might return a string directly
                  raw_agent_output = result
             else:
                  # Handle unexpected result types
                  raw_agent_output = f"Agent finished, but returned an unexpected result type: {type(result)}. Content: {result}"

             print(f"DEBUG: Extracted raw agent output: '{raw_agent_output}'")


        # --- Check for Delete Confirmation Request (Logic from provided main.py) ---
        delete_prefix = "CONFIRM_DELETE|" # Define the prefix the agent tool should output
        print(f"DEBUG: Checking if output starts with '{delete_prefix}'")

        if isinstance(raw_agent_output, str) and raw_agent_output.startswith(delete_prefix):
            print("DEBUG: Detected delete confirmation request.")
            try:
                # Extract the relative path AFTER the prefix
                relative_path_to_delete = raw_agent_output.split('|', 1)[1].strip()
                if not relative_path_to_delete:
                    raise ValueError("Extracted relative path for deletion is empty.")

                # Construct full path carefully relative to OUTPUT_DIR
                # Use resolve() to get the absolute path and normalize '..' etc.
                full_path_to_delete = (OUTPUT_DIR / relative_path_to_delete).resolve()
                print(f"DEBUG: Resolved full path for potential deletion: {full_path_to_delete}")

                # --- CRUCIAL SECURITY CHECK ---
                # Ensure the resolved path is actually WITHIN the designated OUTPUT_DIR
                # This prevents deleting files outside the intended scope (e.g., "../important_file.txt")
                if not full_path_to_delete.is_relative_to(OUTPUT_DIR):
                    print(f"\n--- SECURITY ERROR ---")
                    print(f"ERROR: Agent requested deletion of a path outside the designated output directory!")
                    print(f"   Requested relative path: '{relative_path_to_delete}'")
                    print(f"   Resolved absolute path:  '{full_path_to_delete}'")
                    print(f"   Allowed base directory: '{OUTPUT_DIR}'")
                    print("--- Deletion Denied ---")
                    # Do NOT proceed with deletion

                else:
                    # Path is safe, proceed with user confirmation
                    print(f"\n!! AGENT REQUESTS DELETE CONFIRMATION !!")
                    print(f"   File: '{relative_path_to_delete}' (inside '{OUTPUT_DIR.name}')")
                    print(f"   Full path: '{full_path_to_delete}'")

                    # Check if the file actually exists before asking
                    if not full_path_to_delete.is_file():
                         print(f"   WARNING: File does not exist at the specified path. Cannot delete.")
                    else:
                         # Prompt user for confirmation via command line
                         user_confirmation = input("   Are you sure you want to delete this file? (yes/no): ").strip().lower()

                         if user_confirmation in ['yes', 'y']:
                             print(f"--- User confirmed deletion. Attempting to delete using perform_delete tool... ---")
                             # Call the actual delete function from the imported tool
                             # Ensure perform_delete expects the full path string
                             success, message = perform_delete(str(full_path_to_delete))
                             if success:
                                 print(f"   SUCCESS: {message}")
                             else:
                                 print(f"   ERROR: {message}")
                         else:
                             print("--- Deletion cancelled by user. ---")

            except IndexError:
                print("\n--- Error Processing Agent Output ---")
                print("ERROR: Agent requested deletion but the output format was incorrect (missing '|' or path?).")
                print(f"Raw Output: {raw_agent_output}")
            except ValueError as ve:
                print("\n--- Error Processing Agent Output ---")
                print(f"ERROR: Agent requested deletion but the path seems invalid or is empty: {ve}")
                print(f"Raw Output: {raw_agent_output}")
            except Exception as e:
                print("\n--- Error During Confirmation/Deletion ---")
                print(f"An unexpected error occurred: {e}")
                traceback.print_exc()

        else:
            # If not a delete confirmation, print the agent's final answer normally
            print(f"\n--- Final Answer ---")
            print(raw_agent_output)


    # --- Specific LangChain Exception Handling ---
    except OutputParserException as e:
        print("\n--- Agent Execution Error ---")
        print(f"ERROR: The LLM failed to format its response correctly for the agent.")
        print(f"This often means it couldn't determine the next action or didn't follow the expected format.")
        print(f"Details: {e}")
        # Consider printing the problematic LLM output if available in 'e'
        # print(f"Problematic LLM Output: {e.llm_output}") # Check attribute name
        traceback.print_exc()

    # --- General Exception Handling (copied from provided main.py) ---
    except ValueError as e:
         # Can be raised by API key checks, path issues etc.
         print("\n--- Configuration or Value Error ---")
         print(f"ERROR: {e}")
         print("Check API keys in .env, file paths, or input values.")
         traceback.print_exc() # Show where the ValueError originated
    except ImportError as e:
         # Should be caught earlier, but added as a safeguard
         print("\n--- Dependency Error ---")
         print(f"ERROR: Missing required library: {e}")
         print("Ensure all packages are installed (e.g., pip install -r requirements.txt).")
         traceback.print_exc()
    except Exception as e:
        # Catch-all for any other unexpected errors during initialization or execution
        print("\n--- Unexpected Agent Execution Error ---")
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        traceback.print_exc()

    finally:
        # This block always executes, regardless of errors
        print("\n--- Task Execution Attempt Complete ---")
        print(f"Check the '{OUTPUT_DIR.name}' directory ({OUTPUT_DIR}) for any generated/modified files.")


if __name__ == "__main__":
    # Ensure the script entry point is clear
    print("DEBUG: Starting main execution block...")
    main()
    print("DEBUG: Finished main execution block.")
