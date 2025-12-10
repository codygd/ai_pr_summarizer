import os

from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import json

app = FastAPI()
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

class PRPayload(BaseModel):
    pr_title: str
    pr_body: str
    diff: str

@app.post("/summarize-pr")
def summarize_pr(payload: PRPayload):
    prompt = f"""
    You are a senior software engineer. Summarize this pull request clearly and concisely.

    Title:
    {payload.pr_title}

    Body:
    {payload.pr_body}

    Diff:
    {payload.diff}

    Return:
    - A 2–3 sentence summary
    - Main changes included in the diff
    - Any risks, breaking changes, or test implications
    """


    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId="google.gemma-3-12b-it",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())

    # Nova response structure
    summary = result["choices"][0]["message"]["content"]

    return {"summary": summary}





from mangum import Mangum
handler = Mangum(app)



"""
next steps
setup AWS creds
maybe set limits on how much stats are used.
install serverless (first need node) 
configure serverless with AWS creds.
try deploying this thing?
"""