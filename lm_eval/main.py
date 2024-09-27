# main.py

import subprocess
import sys
import os
import signal
import time
import threading

def stream_output(pipe, name):
    """
    Reads lines from a subprocess pipe and prints them with a prefix.

    Args:
        pipe (io.TextIOWrapper): The pipe to read from (stdout or stderr).
        name (str): A prefix name to identify the source of the output.
    """
    with pipe:
        for line in iter(pipe.readline, ''):
            print(f"[{name}] {line.rstrip()}")

def start_subprocess(command, name):
    """
    Starts a subprocess and begins streaming its output.

    Args:
        command (list): The command to execute as a subprocess.
        name (str): A prefix name to identify the source of the output.

    Returns:
        subprocess.Popen: The subprocess object.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # Line-buffered
        universal_newlines=True  # Text mode
    )
    
    # Start threads to handle stdout and stderr
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"{name} STDOUT"), daemon=True)
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name} STDERR"), daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    return process

def main():
    # Determine the directory where main.py resides
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_path = os.path.join(current_dir, "api.py")
    interface_path = os.path.join(current_dir, "interface.py")

    # Check if api.py exists
    if not os.path.isfile(api_path):
        print(f"Error: '{api_path}' does not exist.")
        sys.exit(1)
    
    # Check if interface.py exists
    if not os.path.isfile(interface_path):
        print(f"Error: '{interface_path}' does not exist.")
        sys.exit(1)

    # Start FastAPI subprocess
    print("Starting FastAPI server (http://localhost:5000)...")
    fastapi_command = [sys.executable, api_path]
    fastapi_process = start_subprocess(fastapi_command, "FastAPI")

    # Allow some time for FastAPI to start
    time.sleep(2)  # Adjust if necessary

    # Start Streamlit subprocess
    print("Starting Streamlit app (http://localhost:8000)...")
    streamlit_command = ["streamlit", "run", interface_path, "--server.port", "8000"]
    streamlit_process = start_subprocess(streamlit_command, "Streamlit")

    try:
        while True:
            # Check if FastAPI has terminated
            fastapi_retcode = fastapi_process.poll()
            if fastapi_retcode is not None:
                print(f"FastAPI server exited with code {fastapi_retcode}.")
                break

            # Check if Streamlit has terminated
            streamlit_retcode = streamlit_process.poll()
            if streamlit_retcode is not None:
                print(f"Streamlit app exited with code {streamlit_retcode}.")
                break

            time.sleep(1)  # Avoid busy waiting
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt. Shutting down...")

        # Terminate FastAPI subprocess
        if fastapi_process.poll() is None:
            print("Terminating FastAPI server...")
            fastapi_process.terminate()
            try:
                fastapi_process.wait(timeout=5)
                print("FastAPI server terminated.")
            except subprocess.TimeoutExpired:
                print("FastAPI server did not terminate in time. Killing...")
                fastapi_process.kill()

        # Terminate Streamlit subprocess
        if streamlit_process.poll() is None:
            print("Terminating Streamlit app...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
                print("Streamlit app terminated.")
            except subprocess.TimeoutExpired:
                print("Streamlit app did not terminate in time. Killing...")
                streamlit_process.kill()

        print("Shutdown complete.")

    finally:
        # Optionally, handle any cleanup here
        pass

if __name__ == "__main__":
    main()
