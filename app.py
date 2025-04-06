# app.py (Revised UI Version)

import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout
import traceback
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import datetime # To display file modification times

# --- Add project root to Python path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# --- ---

# --- Load Environment Variables ---
load_dotenv()
# --- ---

# --- Define Output Directory ---
OUTPUT_DIR = Path("outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# --- ---

# --- Import Agent Logic ---
try:
    from agent.planner import initialize_agent
    # --- Import the actual delete function if needed directly (unlikely needed here) ---
    # from tools.delete_file_tool import perform_delete
except ImportError as e:
    st.error(f"🚨 Error importing agent logic: {e}. Make sure 'agent/planner.py' exists and dependencies are installed.", icon="🚨")
    traceback.print_exc()
    st.stop()
except Exception as e:
    st.error(f"🚨 An unexpected error occurred during agent import: {e}", icon="🚨")
    traceback.print_exc()
    st.stop()
# --- ---

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Agent Interface",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="collapsed",
    page_icon="🤖"
)

# --- Custom CSS Styling (Minor adjustments optional) ---
st.markdown("""
<style>
    /* General dark theme styling remains similar */
    body {
        background-color: #121212;
        color: #E0E0E0; /* Slightly lighter default text */
    }
    .stApp {
         /* background: url("your_background_image.png"); /* Optional background */
         background-size: cover;
    }
    h1 {
        text-align: center;
        color: #4CAF50; /* Green title */
        font-size: 2.8em;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    .stCaption {
        text-align: center;
        color: #A0A0A0; /* Lighter gray caption */
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .stTextArea [data-baseweb="textarea"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333333;
        border-radius: 8px; /* Slightly more rounded */
        font-size: 1.1em;
    }
    /* Main Run button slightly larger */
    div[data-testid="stButton"] > button[kind="primary"] {
        font-size: 1.1em;
        padding: 12px 24px;
        background-color: #4CAF50;
        border-radius: 8px;
    }
     div[data-testid="stButton"] > button[kind="primary"]:hover {
         background-color: #45a049;
     }
    /* Secondary buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
         border-radius: 8px;
         border: 1px solid #4CAF50; /* Green border */
         color: #4CAF50;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
         border-color: #45a049;
         color: #45a049;
         background-color: rgba(76, 175, 80, 0.1); /* Slight green background on hover */
    }

    /* Style containers for visual separation */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stExpander"] {
         background-color: #1E1E1E;
         border-radius: 8px;
         border: 1px solid #333;
    }
     [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stTickBar"] > div {
        background-color: #1E1E1E;
    }

     /* File details container */
     .file-details-container {
         background-color: rgba(40, 40, 40, 0.8); /* Slightly transparent dark */
         padding: 15px;
         border-radius: 8px;
         border: 1px solid #444;
         margin-top: 10px;
     }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤖 Autonomous AI Agent")
st.caption("Input your instruction and let the AI handle the task.")

# --- Check for Necessary API Key ---
# Adapt this check based on the LLM you are using in planner.py
llm_provider = "API"
api_key_needed = None
try:
    # Basic check based on function code content (adjust if needed)
    planner_code = initialize_agent.__code__.co_code
    if b'Google' in planner_code or b'gemini' in planner_code:
        api_key_needed = "GOOGLE_API_KEY"; llm_provider = "Google Gemini"
    elif b'HuggingFace' in planner_code or b'hf_' in planner_code:
        api_key_needed = "HUGGINGFACEHUB_API_TOKEN"; llm_provider = "Hugging Face"
    elif b'OpenAI' in planner_code or b'sk-' in planner_code:
        api_key_needed = "OPENAI_API_KEY"; llm_provider = "OpenAI"
except Exception as inspect_err:
     print(f"DEBUG: Could not inspect agent code for API keys: {inspect_err}")

if api_key_needed and not os.getenv(api_key_needed):
    st.error(f"🚨 **{llm_provider} API key not found!** Please ensure `{api_key_needed}` is set in your `.env` file.", icon="🔑")
    st.stop()
else:
     st.info(f"✅ API Key check passed (or not required based on inspection). Provider detected: {llm_provider}", icon="🔑")
     time.sleep(1) # Show message briefly
     st.empty() # Clear the info message
# --- ---

# --- Agent Initialization (Cached) ---
@st.cache_resource(show_spinner="Initializing AI Agent...")
def load_agent_executor():
    print("--- Attempting to initialize agent executor (cached) ---")
    try:
        # Initialize non-verbosely for caching; verbose output capture happens during invoke
        agent_exec = initialize_agent(verbose=False) # Set verbose=False for caching
        print("--- Agent executor initialized successfully ---")
        return agent_exec
    except Exception as e:
        st.error(f"💥 **Fatal Error:** Failed to initialize the AI agent. Check console logs. Error: {e}", icon="🔥")
        print("--- Agent Initialization Error ---")
        traceback.print_exc()
        print("--- End Agent Initialization Error ---")
        return None

agent_executor = load_agent_executor()

if not agent_executor:
    st.warning("Agent could not be loaded. The application cannot proceed.")
    st.stop()
# --- ---


# ==============================================================================
#                            AGENT INTERACTION SECTION
# ==============================================================================
st.header("💬 Interact with the Agent")

# Use columns for better layout
col_instruct, col_run = st.columns([4, 1]) # Instruction area takes more space

with col_instruct:
    instruction = st.text_area(
        "**Enter your instruction:**",
        height=100, # Slightly shorter default height
        placeholder="e.g., Analyze the sentiment of 'outputs/customer_feedback.txt' and save the result to 'outputs/sentiment_analysis.txt'.",
        label_visibility="collapsed" # Hide label as we have the header
    )

with col_run:
    # Place button lower, vertically centered if possible (hard in streamlit columns)
    st.write("") # Spacer
    st.write("") # Spacer
    submit_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)


# --- Agent Execution Logic & Result Display ---
# Use session state to store the last result and logs
if 'last_final_output' not in st.session_state:
    st.session_state.last_final_output = None
if 'last_verbose_output' not in st.session_state: # Still useful for debugging
    st.session_state.last_verbose_output = None

if submit_button and instruction:
    st.info("⏳ Processing your request... Please wait.", icon="🧠")
    stdout_capture = io.StringIO()
    st.session_state.last_final_output = None # Clear previous results
    st.session_state.last_verbose_output = None

    try:
        with st.spinner("Agent thinking and acting... 🤔"):
            # --- IMPORTANT: Verbose Logging ---
            # Set capture_output to True if you want to capture Langchain's verbose prints
            # This can be VERY verbose. Keep False for cleaner UI unless debugging.
            capture_output = False
            if capture_output:
                 with redirect_stdout(stdout_capture): # Capture prints
                    result = agent_executor.invoke({"input": instruction})
                 st.session_state.last_verbose_output = stdout_capture.getvalue() # Store captured logs
            else:
                 # Run without capturing stdout (faster, cleaner UI)
                 result = agent_executor.invoke({"input": instruction})

        # --- Process Result ---
        if isinstance(result, dict):
            final_output = result.get('output', '*Agent finished, but no specific "output" key found in result.*')
        elif isinstance(result, str):
             final_output = result # Handle agents returning strings
        else:
             final_output = f"*Agent finished, but returned unexpected result type: {type(result)}*"

        st.session_state.last_final_output = final_output # Store result
        st.success("✅ Agent execution completed successfully!")

        # --- Handle potential delete confirmation (if agent output follows convention) ---
        # Check if the agent's output indicates a need for confirmation
        # This is an alternative to the CLI approach - requires agent tool to output this specific format.
        delete_prefix = "CONFIRM_DELETE|"
        if isinstance(final_output, str) and final_output.startswith(delete_prefix):
            relative_path_to_delete = final_output.split('|', 1)[1].strip()
            st.session_state.pending_delete_path = relative_path_to_delete
            st.session_state.last_final_output = f"Agent requested confirmation to delete: {relative_path_to_delete}" # Update display message
            # We'll handle the actual confirmation UI in the file browser section for consistency
        else:
             if 'pending_delete_path' in st.session_state:
                  del st.session_state['pending_delete_path'] # Clear any old pending delete


    except Exception as e:
        st.error(f"❌ An error occurred during agent execution: {e}", icon="🔥")
        print("\n--- Agent Execution Error in Streamlit App ---")
        traceback.print_exc()
        print("--- End Agent Execution Error ---\n")
        # Store any logs captured before the error if capture was enabled
        if 'capture_output' in locals() and capture_output:
             st.session_state.last_verbose_output = stdout_capture.getvalue()

elif submit_button and not instruction:
    st.warning("⚠️ Please enter an instruction before clicking 'Run Agent'.", icon="✍️")


# --- Display Agent Results / PDF Generation ---
if st.session_state.last_final_output:
    st.markdown("---")
    with st.container(border=True): # Use container for visual grouping
        st.subheader("📄 Agent's Final Output:")
        # Display as a code block for potentially long/formatted output
        st.code(st.session_state.last_final_output, language=None)

        # --- PDF Generation Option ---
        st.markdown("---") # Divider within the container
        st.markdown("**Report Options:**")
        pdf_col, _ = st.columns([1,3]) # Button in smaller column
        with pdf_col:
            if st.button("📄 Generate PDF Report", key="pdf_generate"):
                 try:
                     # Example chart data (replace with actual dynamic data if possible)
                     chart_data = [
                         {"x": [2019, 2020, 2021, 2022], "y": [100, 150, 120, 180], "label": "Metric A", "title": "Example Trend A"},
                         {"x": [2019, 2020, 2021, 2022], "y": [80, 90, 110, 100], "label": "Metric B", "title": "Example Trend B"},
                     ]
                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                     pdf_filename = f"generated_report_{timestamp}.pdf"
                     output_pdf_path = OUTPUT_DIR / pdf_filename

                     # Generate PDF in a temporary directory to avoid clutter? Or directly in outputs.
                     # Using FPDF and Matplotlib directly here.
                     pdf = FPDF()
                     pdf.set_auto_page_break(auto=True, margin=15)
                     pdf.add_page()
                     pdf.set_font("Arial", size=12)
                     pdf.set_font("Arial", style="B", size=16)
                     pdf.cell(0, 10, txt="Generated AI Agent Report", ln=True, align="C")
                     pdf.ln(10)
                     pdf.set_font("Arial", size=12)
                     pdf.multi_cell(0, 10, txt=f"Instruction: {st.session_state.get('last_instruction', 'N/A')}\n--- Output ---\n{st.session_state.last_final_output}")
                     pdf.ln(5)

                     # Add Charts to PDF
                     pdf.add_page()
                     pdf.set_font("Arial", style="B", size=14)
                     pdf.cell(0, 10, txt="Example Charts", ln=True, align="L")
                     pdf.ln(5)
                     chart_image_files = []
                     for idx, data in enumerate(chart_data):
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_chart:
                            plt.figure(figsize=(8, 5)) # Slightly larger figure
                            plt.plot(data["x"], data["y"], label=data.get("label", "Data"), marker='o')
                            plt.title(data.get("title", f"Chart {idx+1}"))
                            plt.xlabel(data.get("xlabel", "Year"))
                            plt.ylabel(data.get("ylabel", "Value"))
                            plt.legend()
                            plt.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout()
                            plt.savefig(temp_chart.name)
                            plt.close() # Close the plot figure
                            chart_image_files.append(temp_chart.name)

                     # Embed images (handle potential errors)
                     img_width = 180 # mm
                     img_height = 110 # mm approx for 8x5 figure
                     y_pos = pdf.get_y()
                     for i, img_path in enumerate(chart_image_files):
                         try:
                             pdf.image(img_path, x=15, y=y_pos, w=img_width) # Adjust x, y, w, h as needed
                             y_pos += img_height + 10 # Move down for next image + padding
                             if y_pos > 250: # Simple check to avoid going off page
                                 pdf.add_page()
                                 y_pos = 30
                         except Exception as img_err:
                             print(f"Error embedding image {img_path}: {img_err}")
                             st.warning(f"Could not embed chart {i+1} in PDF.")
                         finally:
                             os.unlink(img_path) # Clean up temp image file

                     # Save PDF
                     pdf.output(str(output_pdf_path))

                     st.success(f"📄 PDF report generated: '{output_pdf_path.name}'")
                     with open(output_pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_file,
                            file_name=pdf_filename, # Use dynamic filename
                            mime="application/pdf",
                            key="pdf_download"
                        )
                 except Exception as e:
                     st.error(f"❌ Failed to generate PDF: {e}")
                     traceback.print_exc()

# --- Verbose Log Expander (Optional - for debugging) ---
if st.session_state.last_verbose_output:
     with st.expander("🕵️ Last Agent Thought Process (Verbose Logs)", expanded=False):
        st.text_area("Logs:", value=st.session_state.last_verbose_output, height=400, disabled=True, key="verbose_log_area_display")


# ==============================================================================
#                            OUTPUT FILES BROWSER SECTION
# ==============================================================================
st.divider() # Visual separator
st.header("📂 Output Files Management")
st.caption(f"Browse, preview, download, or delete files in the `{OUTPUT_DIR.name}` directory.")

# --- Function to get files (Enable caching) ---
@st.cache_data # Re-enable caching, refresh button invalidates it
def get_files_in_outputs(base_dir):
    print(f"--- Scanning directory: {base_dir} ---") # Keep for debug
    files_info = []
    try:
        if not base_dir.exists() or not base_dir.is_dir():
            return [], f"Output directory '{base_dir.name}' not found or is not a directory."

        # Include modification time
        all_items = sorted(
            [item for item in base_dir.rglob('*') if item.is_file() and item.name != '.gitkeep'],
            key=lambda x: x.stat().st_mtime, # Sort by modification time
            reverse=True # Newest first
        )

        for item in all_items:
            try:
                stat_result = item.stat()
                relative_path = item.relative_to(base_dir)
                files_info.append({
                    "display_path": str(relative_path),
                    "full_path": item,
                    "size": stat_result.st_size,
                    "modified": datetime.datetime.fromtimestamp(stat_result.st_mtime)
                })
            except Exception as stat_err:
                 print(f"Could not process file {item}: {stat_err}") # Log error

        if not files_info:
            return [], "Output directory is currently empty." # More user-friendly message
        return files_info, None # Return data and no error message
    except PermissionError:
        return [], f"Error: Permission denied when scanning {base_dir}."
    except Exception as e:
        traceback.print_exc()
        return [], f"An unexpected error occurred while scanning the output directory: {e}"

# --- Refresh Button and File Selection ---
col_refresh, col_select = st.columns([1, 4])

with col_refresh:
    if st.button("🔄 Refresh List", key="refresh_files", help="Rescan the outputs directory"):
        st.cache_data.clear() # Clear the file list cache
        st.session_state.selected_file_display = "-- Select a file --" # Reset selection on refresh
        st.rerun()

output_files_list, error_msg = get_files_in_outputs(OUTPUT_DIR)

with col_select:
    if error_msg:
        st.warning(error_msg, icon="⚠️")
    elif not output_files_list:
        st.info("Output directory is empty. Run the agent to generate some files!", icon="📄")
    else:
        # Format options for selectbox including size and modified time
        file_options = {
            f"{info['display_path']} ({info['size']/1024:.1f} KB | {info['modified']:%Y-%m-%d %H:%M})": info['full_path']
            for info in output_files_list
        }
        # Add placeholder
        display_list = ["-- Select a file --"] + list(file_options.keys())

        # --- Session state for remembering selection ---
        if 'selected_file_display' not in st.session_state:
             st.session_state.selected_file_display = display_list[0]

        # Ensure the selected file still exists in the current list before rendering selectbox
        if st.session_state.selected_file_display != display_list[0] and st.session_state.selected_file_display not in display_list:
             st.info(f"Previously selected file '{st.session_state.selected_file_display}' seems to be gone. Resetting selection.", icon="❓")
             st.session_state.selected_file_display = display_list[0] # Reset to placeholder

        # --- Selectbox ---
        selected_display_option = st.selectbox(
            "**Select Output File:**",
            options=display_list,
            key='file_selector',
            index=display_list.index(st.session_state.selected_file_display), # Use state for index
            label_visibility="collapsed"
        )
        # Update session state on change
        if selected_display_option != st.session_state.selected_file_display:
             st.session_state.selected_file_display = selected_display_option
             # Clear delete confirmation state when selection changes
             confirm_key = f"confirm_del_{selected_display_option}" # Construct potential key
             if confirm_key in st.session_state:
                 st.session_state[confirm_key] = False
             st.rerun() # Rerun immediately on selection change for smoother UI update


# --- Display Selected File Content and Action Buttons ---
if 'selected_display_option' in locals() and selected_display_option != "-- Select a file --":
    selected_full_path = file_options.get(selected_display_option)

    # Check existence *after* getting from dictionary
    if selected_full_path and selected_full_path.exists():
        file_info = next((f for f in output_files_list if f['full_path'] == selected_full_path), None) # Get full info dict

        # Use a container for the selected file's details area
        with st.container(border=True): # Adds a nice border
            display_name = file_info['display_path']
            st.subheader(f"📄 Details for: `{display_name}`")
            st.caption(f"Size: {file_info['size']/1024:.2f} KB | Modified: {file_info['modified']:%Y-%m-%d %H:%M:%S}")

            st.markdown("---") # Divider

            # --- Action Buttons Row ---
            col_dl, col_del, col_spacer = st.columns([1, 1, 3]) # Adjust spacing

            # Download Button
            with col_dl:
                 try:
                     with open(selected_full_path, "rb") as fp:
                         st.download_button(
                             label="⬇️ Download",
                             data=fp,
                             file_name=selected_full_path.name, # Use actual file name
                             mime="application/octet-stream", # General binary type
                             key=f"dl_{selected_display_option}",
                             use_container_width=True
                         )
                 except Exception as e:
                     st.error(f"Download Error: {e}", icon="❌")

            # --- Delete Button and Confirmation ---
            with col_del:
                delete_key = f"del_{selected_display_option}" # Unique key per file option
                confirm_key = f"confirm_{delete_key}"

                # Initialize confirmation state for this specific file if not present
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False

                # Show initial delete button OR confirmation dialog
                if not st.session_state[confirm_key]:
                    # Show the initial "Delete" button
                    if st.button("🗑️ Delete File", key=delete_key, type="secondary", help="Delete this file from the outputs directory", use_container_width=True):
                        st.session_state[confirm_key] = True # Set confirmation flag for THIS file
                        st.rerun() # Rerun immediately to show confirmation
                else:
                    # Show confirmation message and buttons
                    st.error(f"**Confirm Deletion:** Are you sure you want to permanently delete `{display_name}`?", icon="⚠️") # Use error for prominence
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("✔️ YES, DELETE", key=f"confirm_yes_{delete_key}", type="primary", help="Confirm Deletion", use_container_width=True):
                            try:
                                print(f"--- Deleting file via UI: {selected_full_path} ---")
                                selected_full_path.unlink() # Perform deletion
                                st.success(f"Deleted `{display_name}`!", icon="✅")
                                # Reset state AFTER successful deletion
                                st.session_state[confirm_key] = False
                                st.session_state.selected_file_display = display_list[0] # Reset dropdown
                                st.cache_data.clear() # Clear file cache
                                time.sleep(0.8) # Brief pause for user to see success msg
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting `{display_name}`: {e}", icon="❌")
                                print(f"--- Deletion Error: {e} ---")
                                traceback.print_exc()
                                st.session_state[confirm_key] = False # Reset confirmation on error
                                st.rerun() # Rerun to hide buttons after error msg
                    with col_cancel:
                         if st.button("❌ Cancel", key=f"confirm_no_{delete_key}", type="secondary", use_container_width=True):
                              st.session_state[confirm_key] = False # Clear confirmation flag
                              st.rerun() # Rerun to hide confirmation

            st.markdown("---") # Divider before preview

            # --- Content Preview (inside an expander) ---
            with st.expander("**👀 Content Preview**", expanded=True): # Default to expanded
                text_extensions = ['.txt', '.md', '.log', '.csv', '.json', '.py', '.yaml', '.yml', '.html', '.css', '.js', '.sh', '.toml'] # Added toml
                file_ext = selected_full_path.suffix.lower()
                max_preview_size = 1 * 1024 * 1024 # 1 MB limit

                if file_info['size'] == 0:
                     st.info("📄 File is empty.")
                elif file_ext in text_extensions:
                    if file_info['size'] > max_preview_size:
                        st.warning(f"💾 File is large ({file_info['size']/1024/1024:.1f} MB). Preview below is truncated to 1MB. Use Download for full content.", icon="✂️")
                    try:
                        # Read potentially large file safely
                         with open(selected_full_path, "r", encoding="utf-8", errors="ignore") as f:
                             # Read only up to the limit + 1 to check for truncation
                             content = f.read(max_preview_size + 1)
                         content_to_display = content[:max_preview_size] # Slice to the limit

                         # --- Display Logic ---
                         if file_ext == '.md':
                             st.markdown(content_to_display, unsafe_allow_html=False) # Render Markdown
                         elif file_ext == '.csv':
                              # Basic CSV handling - display as code or try DataFrame
                              try:
                                  import pandas as pd
                                  df_preview = pd.read_csv(io.StringIO(content_to_display))
                                  st.dataframe(df_preview)
                              except ImportError:
                                  st.code(content_to_display, language='csv')
                              except Exception as csv_err:
                                   st.warning(f"Could not parse as CSV for preview: {csv_err}. Displaying raw text.")
                                   st.code(content_to_display, language=None)
                         else:
                             # Map extensions to languages for st.code
                             lang_map = {'.py':'python', '.js':'javascript', '.html':'html', '.css':'css', '.sh':'bash', '.json':'json', '.yaml':'yaml', '.yml':'yaml', '.toml': 'toml', '.log': None, '.txt': None}
                             st.code(content_to_display, language=lang_map.get(file_ext, None))

                    except Exception as e:
                        st.error(f"Error reading/displaying text file content: {e}", icon="❌")
                else: # Non-text file or unknown extension
                    st.info(f"📄 Preview not available for this file type ({file_ext}). Use the 'Download' button.", icon="ℹ️")

    # Handle case where file disappeared between listing and selection processing
    elif 'selected_display_option' in locals() and selected_display_option != "-- Select a file --":
         st.error("Selected file not found. It might have been deleted externally. Refreshing list...", icon="❓")
         st.cache_data.clear()
         st.session_state.selected_file_display = "-- Select a file --" # Reset selection
         time.sleep(1)
         st.rerun()


# --- Footer Info ---
st.divider()
st.markdown("<p style='text-align: center; color: grey; font-size: 0.9em;'>AI Agent Interface | Outputs are saved in the <code>outputs/</code> directory.</p>", unsafe_allow_html=True)
# --- ---
