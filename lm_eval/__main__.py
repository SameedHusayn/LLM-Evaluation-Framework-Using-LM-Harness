from validation import validate_params

import streamlit as st
import threading
import uvicorn
from fastapi import FastAPI, Request
import streamlit as st
from validation import validate_params

# FastAPI app
app = FastAPI()

# Streamlit interface setup
def run_streamlit():
    # Define all tasks
    all_tasks = [
        "anli", "arc_challenge", "arithmetic", "asdiv", "bigbench_multiple_choice", 
        "blimp", "commonsense_qa", "coqa", "drop", "eq_bench", "fda", "glue", 
        "gpqa", "gsm8k", "hellaswag", "inverse_scaling_mc", "lambada", "leaderboard", 
        "mathqa", "med_concepts_qa", "mmlu", "mmlusr", "mutual", "qasper", 
        "squadv2", "super-glue-lm-eval-v1", "truthfulqa", "unscramble", "wikitext"
    ]

    # Define recommended tasks
    recommended_tasks = ["glue", "mmlu", "hellaswag", "truthfulqa","anli"]

    # Combine options with "all" and "recommended"
    tasks_options = ["all", "recommended"] + all_tasks

    # Initialize session state for selected_tasks if not already set
    if 'selected_tasks' not in st.session_state:
        st.session_state.selected_tasks = []

    # Callback function to handle selection logic
    def handle_selection():
        selected = st.session_state.selected_tasks.copy()
        
        # If "all" is selected
        if "all" in selected:
            # Select all individual tasks
            st.session_state.selected_tasks = all_tasks.copy()
        
        # If "recommended" is selected
        elif "recommended" in selected:
            # Select recommended tasks
            st.session_state.selected_tasks = recommended_tasks.copy()
        
        else:
            # Ensure "all" and "recommended" are not in the selection
            st.session_state.selected_tasks = [task for task in selected if task not in ["all", "recommended"]]

    # Streamlit UI
    st.title("LLM Evaluation")

    # Multiselect with callback
    selected_tasks = st.multiselect(
        "Select one or multiple tasks",
        options=tasks_options,
        default=st.session_state.selected_tasks,
        key='selected_tasks',
        on_change=handle_selection
    )

    # Determine the display based on selection
    def get_display_tasks(selected):
        if set(selected) == set(all_tasks):
            return "All tasks selected."
        elif set(selected) == set(recommended_tasks):
            return "Recommended tasks selected: " + ", ".join(recommended_tasks)
        else:
            return ", ".join(selected) if selected else ""

    display_tasks_str = get_display_tasks(selected_tasks)

    # Only display selected tasks if there are any
    if display_tasks_str:
        st.write("**Selected Tasks:**", display_tasks_str)

    # Input field for model name
    model_name = st.text_input("Enter model name")

    # Number of shots with conditional input
    num_shots = None
    set_num_shots = st.checkbox("Set Number of Shots?")
    if set_num_shots:
        num_shots = st.number_input("Enter number of shots", min_value=1, step=1)

    # Limit with conditional input
    limit = None
    set_limit = st.checkbox("Set Limit?")
    if set_limit:
        limit = st.number_input("Enter limit (float)", min_value=0.0, step=0.1, format="%.3f")

    # Option to push results to hub
    push_to_hub = st.checkbox("Do you want to push results to Hugging Face Hub?")
    hf_org_name, hf_token, make_public = None, None, None
    if push_to_hub:
        hf_org_name = st.text_input("Enter Hugging Face Org Name")
        hf_token = st.text_input("Enter Hugging Face Token", type="password")
        make_public = st.checkbox("Make the repository public?")
        hf_repo_name = st.checkbox("Do you want to set a repo name? (Default: lm-eval-results)")
        if hf_repo_name:
            hf_repo_name = st.text_input("Enter Repo Name")
        else:
            hf_repo_name = "lm-eval-results"


    # Function to process and display the results
    def process_tasks(selected_tasks, model_name, num_shots, limit, push_to_hub, hf_org_name, hf_token, make_public):
        st.write("**Processing the following tasks:**")
        st.write(", ".join(selected_tasks))
        st.write("**Model Name:**", model_name)
        st.write("**Number of Shots:**", num_shots if num_shots is not None else "Not set")
        st.write("**Limit:**", limit if limit is not None else "Not set")
        
        if push_to_hub:
            st.write("**Push to Hub Settings:**")
            st.write("Hugging Face Username:", hf_org_name)
            st.write("Hugging Face Token:", "******" if hf_token else "Not set")
            st.write("Public:", "True" if make_public else "False")
        else:
            st.write("Not pushing results to hub.")
        
        # Simulate result processing and return a message
        st.success("Results processed successfully!")

    # Process the information when the user clicks the button
    if st.button("Submit"):
        if not selected_tasks:
            st.error("Please select at least one task.")
        elif not model_name:
            st.error("Please enter the model name.")
        elif push_to_hub and (not hf_org_name or not hf_token):
            st.error("Please provide all Hugging Face Hub details.")
        else:
            output_path = "D:\\Code\\Python\\office\\LLM-Evaluation-Framework-Using-LM-Harness\\results\\"+model_name+"\\"
            selected_tasks = ','.join(selected_tasks)
            print(selected_tasks)
            
            if push_to_hub:
                hf_hub_log_args = "hub_results_org="+hf_org_name+",push_results_to_hub=True,push_samples_to_hub=True,token="+hf_token+",hub_repo_name="+hf_repo_name+",details_repo_name="+hf_repo_name+",public_repo="+str(make_public)
            else:
                hf_hub_log_args = ""            
            
            params = {
                "model": "openai-completions",
                "model_args": "model="+model_name,
                "tasks": selected_tasks,
                "num_fewshot": num_shots,
                "batch_size": 1,
                "max_batch_size": None,
                "device": None,
                "output_path": output_path,
                "limit": limit,
                "use_cache": None,
                "cache_requests": True,
                "check_integrity": False,
                "write_out": False,
                "log_samples": True,
                "system_instruction": None,
                "apply_chat_template": False,
                "fewshot_as_multiturn": False,
                "show_config": False,
                "include_path": None,
                "gen_kwargs": None,
                "verbosity": "INFO",
                "wandb_args": "",
                "hf_hub_log_args": hf_hub_log_args,
                "predict_only": False,
                "seed": [0, 1234, 1234, 1234],
                "trust_remote_code": True,

            }
            processed_results, group_results = validate_params(params)
            if group_results == None:
                st.write(processed_results)
                print(processed_results)
            else:
                print(processed_results)
                print(group_results)
                st.write(processed_results)
                st.write(group_results)

def convert_params(params):
    """
    Converts string representations of booleans and None to actual Python types.
    """
    for key, value in params.items():
        if isinstance(value, str):
            if value.lower() == "true":
                params[key] = True
            elif value.lower() == "false":
                params[key] = False
            elif value.lower() == "none":
                params[key] = None
    return params

# FastAPI route to accept POST requests
@app.post("/validate")
async def validate(request: Request):
    data = await request.json()
    params = data.get("params", {})
    # Convert string values to their appropriate types
    params = convert_params(params)

    processed_results, group_results = validate_params(params)
    return {"processed_results": processed_results, "group_results": group_results}

# Run FastAPI server on a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run Streamlit on a separate thread
def run_streamlit():
    # Use os.system to run Streamlit with subprocess
    import os
    os.system("streamlit run __main__.py --server.port 8501")

# Main function to run both FastAPI and Streamlit
if __name__ == "__main__":
    # Create two threads: one for FastAPI, one for Streamlit
    fastapi_thread = threading.Thread(target=run_fastapi)
    streamlit_thread = threading.Thread(target=run_streamlit)

    # Start the threads
    fastapi_thread.start()
    streamlit_thread.start()

    # Keep the threads running
    fastapi_thread.join()
    streamlit_thread.join()
