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
    # Try loading from current directory first (chapter2/.env)
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        # Fallback to parent directory
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
except ImportError:
    pass  # dotenv is optional

# --- Configure Service Account and Vertex AI credentials ---
# Service account authentication is configured via GOOGLE_APPLICATION_CREDENTIALS
# This environment variable should point to your service account JSON key file.
#
# Configuration options (in order of precedence):
# 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (already set)
# 2. SERVICE_ACCOUNT_KEY_PATH from .env file
# 3. Default path: cred/genai-vertex-data-engineering.json (relative to project root)
#
# To use a service account:
# - Set SERVICE_ACCOUNT_KEY_PATH in your .env file, OR
# - Set GOOGLE_APPLICATION_CREDENTIALS environment variable directly

# Get service account key path from environment or use default
service_account_path = (
    os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or
    os.environ.get('SERVICE_ACCOUNT_KEY_PATH') or
    os.path.join(os.path.dirname(__file__), '..', '..', 'cred', 'genai-vertex-data-engineering.json')
)

# Set GOOGLE_APPLICATION_CREDENTIALS if service account file exists
if os.path.exists(service_account_path) and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(service_account_path)
    print(f"Using service account: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
elif not os.path.exists(service_account_path):
    print(f"Warning: Service account file not found at {service_account_path}")
    print("Set GOOGLE_APPLICATION_CREDENTIALS or SERVICE_ACCOUNT_KEY_PATH to use service account authentication.")

# Get Vertex AI configuration from environment variables
VERTEX_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT') or os.environ.get('PROJECT')
VERTEX_LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION') or os.environ.get('GCP_LOCATION') or os.environ.get('LOCATION', 'us-central1')

# Set environment variables as fallback for underlying Client initialization
if VERTEX_PROJECT and not os.environ.get('GOOGLE_CLOUD_PROJECT'):
    os.environ['GOOGLE_CLOUD_PROJECT'] = VERTEX_PROJECT
if VERTEX_LOCATION and not os.environ.get('GOOGLE_CLOUD_LOCATION'):
    os.environ['GOOGLE_CLOUD_LOCATION'] = VERTEX_LOCATION


# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents for customer support.

def technical_support_handler(issue: str) -> str:
    """
    Handles technical support requests and troubleshooting.
    
    Args:
        issue (str): The technical issue description.
    
    Returns:
        str: A response with troubleshooting steps or solution.
    """
    print("------- Technical Support Handler Called -------")
    return f"Technical support for '{issue}': Troubleshooting steps have been provided. Issue logged for tracking."

def billing_handler(inquiry: str) -> str:
    """
    Handles billing and payment-related inquiries.
    
    Args:
        inquiry (str): The billing inquiry or question.
    
    Returns:
        str: A response with billing information or resolution.
    """
    print("------- Billing Handler Called -------")
    return f"Billing inquiry for '{inquiry}': Account information retrieved. Payment status confirmed."

def product_info_handler(question: str) -> str:
    """
    Handles product information and feature questions.
    
    Args:
        question (str): The product-related question.
    
    Returns:
        str: A response with product information.
    """
    print("------- Product Info Handler Called -------")
    return f"Product information for '{question}': Detailed product specifications and features provided."

# --- Create Tools ---
technical_support_tool = FunctionTool(technical_support_handler)
billing_tool = FunctionTool(billing_handler)
product_info_tool = FunctionTool(product_info_handler)

# --- Define specialized sub-agents equipped with their respective tools ---
# Configure model - use string model name
# Vertex AI configuration is handled via environment variables set above:
# - GOOGLE_APPLICATION_CREDENTIALS (service account)
# - GOOGLE_CLOUD_PROJECT (project ID)
# - GOOGLE_CLOUD_LOCATION (location)
# The Agent's underlying Client will detect these and use Vertex AI automatically
model_name = "gemini-2.5-flash"

technical_agent = Agent(
    name="TechnicalSupport",
    model=model_name,
    description="""A specialist agent that handles technical support requests, 
                troubleshooting, and technical issues by calling the technical support tool.""",
    tools=[technical_support_tool]
)

billing_agent = Agent(
    name="Billing",
    model=model_name,
    description="""A specialist agent that handles billing inquiries, payment questions, 
                and account-related financial matters by calling the billing tool.""",
    tools=[billing_tool]
)

product_agent = Agent(
    name="ProductInfo",
    model=model_name,
    description="""A specialist agent that handles product information requests, 
                feature questions, and product specifications by calling the product info tool.""",
    tools=[product_info_tool]
)

# --- Define the parent agent with explicit delegation instructions
root_agent = Agent(
    name="SupportCoordinator",
    model=model_name,
    instruction=(
        "You are a customer support coordinator. Your task is to analyze incoming customer requests "
        "and delegate them to the appropriate specialist agents. Do not try to answer the customer directly. \n"
        "- For technical issues, troubleshooting, or technical support requests, delegate to the 'TechnicalSupport' agent. \n"
        "- For billing inquiries, payment questions, or account financial matters, delegate to the 'Billing' agent. \n"
        "- For product information, feature questions, or product specifications, delegate to the 'ProductInfo' agent. \n"
    ),
    description="A customer support coordinator that routes customer requests to the correct specialist agent.",
    # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
    sub_agents=[technical_agent, billing_agent, product_agent]
)

# Expose root_agent at module level
__all__ = ["root_agent"]

# --- Execute Logic --- 
async def run_support_coordinator(runner: InMemoryRunner, request: str):
    """
    Runs the support coordinator agent with a given request and delegates.
    """
    print(f"\n --- Running Support Coordinator for request: '{request}' --- ")
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
            print(f"Support Coordinator Final Response: {final_result}")
            return final_result
    except Exception as e:
        print(f"An error occurred while processing the request: {e}")
        return f"An error occurred while processing your request: {e}"

async def main():
    """Main function to run the ADK example.
    """
    print("--- Google ADK Customer Support Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")
    
    runner = InMemoryRunner(root_agent)
    
    # Example Usage
    result_a = await run_support_coordinator(runner, "I can't log into my account. Can you help?")
    print(f"Final Output A: {result_a}\n")
    
    result_b = await run_support_coordinator(runner, "What are the payment options available?")
    print(f"Final Output B: {result_b}\n")
    
    result_c = await run_support_coordinator(runner, "What features does the premium plan include?")
    print(f"Final Output C: {result_c}\n")
    
    result_d = await run_support_coordinator(runner, "My payment failed. What should I do?")
    print(f"Final Output D: {result_d}\n")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

