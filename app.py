import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from inference import load_model_for_inference, predict_next_chars

app = FastAPI()
templates = Jinja2Templates(directory="templates")

try:
    loaded_model, char_to_int, int_to_char = load_model_for_inference()
    print("Model and mappings loaded successfully on startup.")
except FileNotFoundError:
    print("WARNING: Model files not found. Please train the model first by running 'python train.py'")
    loaded_model, char_to_int, int_to_char = None, None, None


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def handle_predict(request: Request, prompt: str = Form(...)):
    error = None
    prediction = ""
    if not loaded_model:
        error = "Model is not loaded. Please run train.py first."
    else:
        prediction = predict_next_chars(
            loaded_model, char_to_int, int_to_char, prompt=prompt, n_chars=120
        )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prompt": prompt,
            "prediction": prediction,
            "error": error,
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
