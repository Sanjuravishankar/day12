import os
from dotenv import load_dotenv # type: ignore
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain_community.tools import DuckDuckGoSearchRun  # Updated import
from langchain_community.utilities import WikipediaAPIWrapper  # Updated import

# Load environment variables from a .env file
load_dotenv()

# Retrieve the correct OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI chat model
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY)  # Ensure OPENAI_API_KEY is correct

# Initialize DuckDuckGoSearchRun and WikipediaAPIWrapper
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()

# Define tools
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on various topics.",
)

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A tool to retrieve information from Wikipedia about specific topics.",
)

# Define prompt templates and other necessary components as before...

# Initialize memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the conversation chain for planning
plan_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

# Initialize the agent with the tools, model, and memory
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[search_tool, wikipedia_tool],
    llm=llm,
    verbose=True,  # verbose option is for printing logs (only for development)
    max_iterations=3,
    prompt=prompt,
    memory=memory,
)

# Example usage
input_query = "List the seven wonders of the world."
response = agent({"input": input_query})

# Print the response
print(response)
