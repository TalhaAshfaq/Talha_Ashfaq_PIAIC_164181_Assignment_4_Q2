import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, set_default_openai_api
from dotenv import load_dotenv
load_dotenv()
import os
import json

gemini_api_key = os.getenv("GEMINI_API_KEY")

#provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

set_tracing_disabled(True)
set_default_openai_api("chat_completions")

agent = Agent(
    instructions="You are a helpful Assistant that can aswer questions",
    name="Panavaersity Support Agent",
    model = model
)

#result = Runner.run_sync(
 #   agent,
  #  input = "What is capital of Pakistan"  
#)

#print(result.final_output)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello i am a LLM, how can i help you?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        agent,
        input = history,  
        )
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history",history)
    await cl.Message(
        content=result.final_output
    ).send()

    with open("chat_history.json", "w") as f:
        json.dump(history, f, indent=4)
