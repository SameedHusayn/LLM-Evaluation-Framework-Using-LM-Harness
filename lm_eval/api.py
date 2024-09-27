# api.py

from fastapi import FastAPI, Request
import uvicorn
from validation import validate_params
import os

app = FastAPI()

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
        if key == "output_path":
            base_dir = os.getcwd()  # Current working directory
            output_path = os.path.join(base_dir, "results")
            os.makedirs(output_path, exist_ok=True)
            params[key] = output_path
    return params

@app.post("/validate")
async def validate(request: Request):
    data = await request.json()
    params = data.get("params", {})
    params = convert_params(params)
    processed_results, group_results = validate_params(params)
    return {"processed_results": processed_results, "group_results": group_results, "output_path": params['output_path']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
