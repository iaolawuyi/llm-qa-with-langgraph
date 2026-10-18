import os
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
import streamlit as st

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Q&A Assistant with History",
    page_icon="🤖",
    layout="centered"
)

# 1. State Schemas with Reducer
class QAItem(BaseModel):
    question: str
    answer: str | None = None

class QAWorkflowState(TypedDict):
    # Using operator.add ensures new items are appended to the list instead of overwritten
    history: Annotated[list[QAItem], operator.add]

# 2. LLM Initialization
model_name = os.getenv('OPENROUTER_MODEL')

if not model_name:
    st.error("Set OPENROUTER_MODEL in your .env file.")
    st.stop()

llm = ChatOpenRouter(
    model=model_name,
    temperature=0,
    max_retries=1,
)

# 3. Define the Q&A Node with a System Prompt
def llm_qa_node(state: QAWorkflowState) -> dict:
    # Get the latest question from the end of the history list
    current_question = state['history'][-1].question
    
    system_prompt = SystemMessage(
        content="You are a helpful, concise assistant designed to provide direct, accurate answers to simple questions."
    )
    human_message = HumanMessage(content=current_question)
    
    response = llm.invoke([system_prompt, human_message]).content
    
    return {
        'history': [QAItem(question=current_question, answer=response)]
    }

# 4. Build and Compile the Graph
@st.cache_resource
def get_workflow():
    graph = StateGraph(QAWorkflowState)
    graph.add_node("llm_qa", llm_qa_node)
    graph.add_edge(START, "llm_qa")
    graph.add_edge("llm_qa", END)
    return graph.compile()

workflow = get_workflow()

# 5. Streamlit Chat Interface
st.title("Q&A Assistant")
st.markdown("Ask multiple questions—your conversation history is appended automatically.")

if "workflow_history" not in st.session_state:
    st.session_state.workflow_history = []

for item in st.session_state.workflow_history:
    with st.chat_message("user"):
        st.markdown(item.question)
    if item.answer:
        with st.chat_message("assistant"):
            st.markdown(item.answer)

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the workflow passing the new question as a list item
                final_state = workflow.invoke({
                    'history': [QAItem(question=prompt)]
                })
                
                # Extract the latest generated answer
                latest_answer = final_state['history'][-1].answer
                st.markdown(latest_answer)
                
                # Update Streamlit session state with the full accumulated history
                st.session_state.workflow_history = final_state['history']
                
            except Exception as e:
                st.error(f"An error occurred: {e}")