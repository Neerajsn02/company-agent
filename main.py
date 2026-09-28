from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing import Optional, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver # can use postgres here
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
import subprocess
import os
import uuid


llm = init_chat_model(
    model="llama3.1:8b",
    model_provider="ollama",
    temperature=0
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


@tool
def read_file(path: str):
    """Read a file from the workspace"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@tool
def write_file(path: str, content: str):
    """Create or overwrite a file in the workspace"""
    with open(path, "w", encoding="utf-8") as f:
        return f.write(content)

    return f"successfuly wrote to {path}"

@tool
def list_directory(path: str = ".") -> str:
    """List the files and directories"""
    items = os.listdir(path)
    return "\n".join(items)

@tool
def search_code(query: str, directory: str = "."):
    """Search for a string in source files"""
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open (path, "r", encoding="utf-8") as f:
                    content = f.read()
                if query in content:
                    results.append(path)
            except (UnicodeDecodeError, PermissionError):
                continue

    return "\n".join(results)

@tool
def run_command(command: str) -> str:
    """Run a shell command in the worksapce"""
    result = subprocess.run(
        command, 
        shell=True,
        capture_output=True,
        text=True,
        timeout=60
    )

    return (
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}\n\n"
        f"EXIT CODE: {result.returncode}"
    )

tools = [
    read_file,
    write_file,
    list_directory,
    search_code,
    run_command
]

# code_llm = ChatOllama(
#     model="qwen2.5-coder:7b",
#     temperature=0
# )

code_agent = create_agent(llm, tools)


# pydantic schema that defines the output of an llm call - used with with_structured_output
class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user wants to just chat, ' \
    'ask for knowledge, or change some code in the project')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: Optional[str]

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier) 
    system_prompt = (
    'Classify the user message into exactly one category. Respond with only one label.\n\n'
    '"chat" = greetings, small talk, opinions, casual conversation\n'
    '"knowledge" = questions asking for facts, information, or referencing documents/notes\n'
    '"code" = explicit requests to write, edit, debug, or explain code, or modify the project\n\n'
    'Examples:\n'
    'User: "hi, how are you" -> chat\n'
    'User: "what is in the report" -> knowledge\n'
    'User: "fix this error" -> code\n'
    'User: "tell me about the weather" -> chat\n'
    'User: "summarize the uploaded pdf" -> knowledge\n'
    )

    # result will be a pydantic object of type IntentClassifer
    result = structured_llm.invoke([
        {'role': 'system', 'content': f'{system_prompt}'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])

    return {'message_intent': result.message_intent}

def prompt_llm_chat(state: State):
    messages = [
        {'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice'} 
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

def prompt_llm_rag(state: State):

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="my_knowledge_base"
    )

    query = state['messages'][-1].content
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    for d in docs:
        print(d.page_content[:200], '\n---')
    context = '\n'.join([doc.page_content for doc in docs])
    system_prompt = (
        "You are given reference material below. Do not repeat, echo, or imitate "
        "the tone or phrasing of the reference material. Answer the user's question "
        "directly and specifically, using only facts from the reference material.\n\n"
        f"Reference material:\n{context}\n\n"
        f"User query: \n{query}\n\n"
        "If the reference material does not contain the answer, say: "
        "\"I don't know based on the provided context.\""
    )

    messages = [
        {'role': 'system', 
        'content': f'{system_prompt}'}
    ]

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_code(state: State):
        user_prompt = state['messages'][-1].content
        response = code_agent.invoke(
            {'messages':[{'role': 'user', 'content': user_prompt}]}
        )

        return {'messages': [{'role': 'assistant', 'content': response['messages'][-1].content}]}


graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('code_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], 
                                    {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'code_agent'})
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('code_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input("Enter Message: ")
    if user_message == "exit":
        break
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config = config)
    print(result['messages'][-1].content)
    


