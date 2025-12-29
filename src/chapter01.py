import uuid
from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event

# --- Load .env if present ---
import os
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass  # dotenv is optional


# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.

def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels. 
    Args:
        request (str): The user's request for booking.

    Returns:
        str: A confirmation message for the booking that the booking was handled.
    """
    print("------- Booking Handler Called -------")
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """
    Handles general information requests.
    
    Args:
        request (str): The user's request.
    Returns:
        A message indicating that the information request was handled.
    """
    print("------- Info Handler Called -------")
    return f"Information request for '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """
    Handles requests that couldn't be clearly delegated.
    """
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# --- Create Tools ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# --- Define specialized sub-agents equipped with their respective tools ---
booking_agent = Agent(
    name="Booker",
    model="gemini-2.5-flash",
    description="""A specialist agent that handles all flights 
                and hotel booking requests by calling the booking tool.""",
    tools=[booking_tool]
    )

info_agent = Agent(
    name="Info",
    model="gemini-2.5-flash",
    description="""A specialist agent that handles all general information 
                and answers user questions by calling the info tool.""",
    tools=[info_tool]
    )

# --- Define the parent agent with explicit delegation instructions
root_agent = Agent(
    name="Coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyze incoming user requests "
        "and delegate them to the appropriate specialist agents. Do not try to answer the user directly. \n"
        "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent. \n "
        "- For all other general information questions, delegate to the 'Info' agent. \n"
    ),
    description="A coordinator that routes user requests to the correct specialist agent.",
    # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
    sub_agents=[booking_agent, info_agent]
)

# Expose root_agent at module level
__all__ = ["root_agent"]

# --- Execute Logic --- 
async def run_coordinator(runner: InMemoryRunner, request: str):
    """
    Runs the coordinator agent with a given request and delegates.
    """
    print(f"\n --- Running Coordinator for request: '{request}' --- ")
    final_result = ""
    
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user", 
                parts=[types.Part(text=request)]
                )
        ):
            if event.is_final_response() and event.content:
                # Try to get text directly from event content
                # to avoid iterating parts
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    # Fallback to iterating parts and extract text (might trigger warning)
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = " ".join(text_parts)
                # Assume the loop should break after the final response
                break
            print(f"Coordinator Final Response: {final_result}")
            return final_result
    except Exception as e:
        print(f"An error occurred while processing the your request: {e}")
        return f"An error occurred while processing your request: {e}"

async def main():
    """Main function to run the ADK example.
    """
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requests Google ADK installed and authenticated.")
    
    runner = InMemoryRunner(root_agent)
    
    # Example Usage
    result_a = await run_coordinator(runner, "Book me a hotel in Paris.")
    print(f"Final Output A: {result_a}\n")
    
    result_b = await run_coordinator(runner, "What is the highest mountain in the world?")
    print(f"Final Output B: {result_b}\n")
    
    result_c = await run_coordinator(runner, "Tell me a random fact.")
    print(f"Final Output C: {result_c}\n")
    
    result_d = await run_coordinator(runner, "Find flights to Tokyo next month.")
    print(f"Final Output D: {result_d}\n")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())