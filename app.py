from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for dev or prod
    allow_methods=["*"],
    allow_headers=["*"],
) 

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/prompt")
async def chat_with_ai(data: PromptRequest):
    prompt = data.prompt

    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    conversation_history.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="ttss-copilot-gpt4-chat",
        messages=conversation_history
    )

    reply = completion.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": reply})

    return {"response": reply}