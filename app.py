import streamlit as st
from AskQuery import ChatKnowledgeBase  # Replace with your actual module name

st.title("EmpowerMen")

# Initialize ChatKnowledgeBase
@st.cache_resource  
def load_chat_kb():
    chat_kb = ChatKnowledgeBase()
    chat_kb.ingest('./knowledge_base')  # Adjust the path to your knowledge base directory
    return chat_kb

chat_kb = load_chat_kb()

# Function to handle user queries and display responses
def ask_bot(query):
    response = chat_kb.ask(query)
    return response

# Streamlit UI
user_input = st.text_input("Ask me anything:")
if st.button("Ask"):
    if user_input:
        response = ask_bot(user_input)
        st.markdown(f"**Bot:** {response}")

# Clear button (optional)
if st.button("Clear Knowledge Base"):
    chat_kb.clear()
    st.success("Knowledge base cleared.")

# Instructions or additional UI elements can be added here

