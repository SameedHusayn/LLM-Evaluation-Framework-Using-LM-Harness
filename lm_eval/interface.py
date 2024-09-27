# interface.py

import os
import streamlit as st
import requests  # To communicate with FastAPI
from validation import validate_params

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

    # Create a selectbox for choosing between Hugging Face and OpenAI
    model_option = st.selectbox("Select a model type:", ("Hugging Face", "OpenAI"))

    # Input field for model name
    model_name = st.text_input("Enter model name")

    # Set the model_type variable based on the user's selection
    if model_option == "Hugging Face":
        model_type = "hf"
        model_args = f"pretrained={model_name},trust_remote_code=True,add_bos_token=True,tokenizer={model_name}"
    elif model_option == "OpenAI":
        model_type = "openai-completions"
        model_args = f"model={model_name}"


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
    hf_repo_name = "lm-eval-results"  # Default repo name
    if push_to_hub:
        hf_org_name = st.text_input("Enter Hugging Face Org Name")
        hf_token = st.text_input("Enter Hugging Face Token", type="password")
        make_public = st.checkbox("Make the repository public?")
        set_repo_name = st.checkbox("Do you want to set a repo name? (Default: lm-eval-results)")
        if set_repo_name:
            hf_repo_name = st.text_input("Enter Repo Name")

    # Function to process and display the results
    def process_tasks(selected_tasks, model_name, num_shots, limit, push_to_hub, hf_org_name, hf_token, make_public, hf_repo_name):
        st.write("**Processing the following tasks:**")
        st.write(", ".join(selected_tasks))
        st.write("**Model Name:**", model_name)
        st.write("**Number of Shots:**", num_shots if num_shots is not None else "Not set")
        st.write("**Limit:**", limit if limit is not None else "Not set")
        
        if push_to_hub:
            st.write("**Push to Hub Settings:**")
            st.write("Hugging Face Org Name:", hf_org_name)
            st.write("Hugging Face Token:", "******" if hf_token else "Not set")
            st.write("Public:", "True" if make_public else "False")
            st.write("Repo Name:", hf_repo_name)
        else:
            st.write("Not pushing results to hub.")
        
        # Simulate result processing and return a message
        st.success("Results processed successfully!")

    # Function to send data to FastAPI
    def send_to_api(params):
        try:
            response = requests.post("http://localhost:5000/validate", json={"params": params})
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the FastAPI server. Ensure it's running.")

    # Process the information when the user clicks the button
    if st.button("Submit"):
        if not selected_tasks:
            st.error("Please select at least one task.")
        elif not model_name:
            st.error("Please enter the model name.")
        elif push_to_hub and (not hf_org_name or not hf_token):
            st.error("Please provide all Hugging Face Hub details.")
        else:
            # # Prepare parameters
            # base_dir = os.getcwd()  # Current working directory
            # output_path = os.path.join(base_dir, "results", model_name)
            # os.makedirs(output_path, exist_ok=True)
            selected_tasks_str = ','.join(selected_tasks)
            
            if push_to_hub:
                hf_hub_log_args = f"hub_results_org={hf_org_name},push_results_to_hub=True,push_samples_to_hub=True,token={hf_token},hub_repo_name={hf_repo_name},details_repo_name={hf_repo_name},public_repo={str(make_public)}"
            else:
                hf_hub_log_args = ""
            
            params = {
                "model": model_type,
                "model_args": model_args,
                "tasks": selected_tasks_str,
                "num_fewshot": num_shots,
                "batch_size": 1,
                "max_batch_size": None,
                "device": None,
                "output_path": "",
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

            # Option 1: Validate parameters locally
            # processed_results, group_results = validate_params(params)
            # if group_results is None:
            #     st.write(processed_results)
            # else:
            #     st.write(processed_results)
            #     st.write(group_results)
            
            # Option 2: Send parameters to FastAPI for validation
            api_response = send_to_api(params)
            if api_response:
                processed_results = api_response.get("processed_results")
                group_results = api_response.get("group_results")
                output_path = api_response.get("output_path")
                
                if group_results is None:
                    st.write(processed_results)
                else:
                    st.write(processed_results)
                    st.write(group_results)
                
                st.write("**Results saved to:** ",output_path)
                # Additionally, you can call process_tasks to display success
                process_tasks(selected_tasks, model_name, num_shots, limit, push_to_hub, hf_org_name, hf_token, make_public, hf_repo_name)

if __name__ == "__main__":
    run_streamlit()
