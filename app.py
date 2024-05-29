import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from htmlTemplates import css, bot_template, user_template

st.set_page_config(page_title="Query Assistant", page_icon=":robot_face:")
st.write(css, unsafe_allow_html=True)


# Initialize session state
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("Query Assistant")
    st.subheader("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Get your API key from [OpenAI Website](https://platform.openai.com/api-keys)")
    os.environ["OPENAI_API_KEY"] = str(openai_api_key)

    custom_urls = st.text_area("Enter URLs (optional)", placeholder="Enter URLs separated by (,)")

    # Custom prompt text area
    custom_prompt_template = st.text_area("User Prompts", placeholder="Enter your custom prompt here...(Optional)")

    if st.button("Load Tools"):
        with st.spinner("Loading tools and creating agent..."):
            # Load Wikipedia tool
            api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=600)
            wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

            # Load arXiv tool
            arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=600)
            arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

            if custom_urls:
                urls = [url.strip() for url in custom_urls.split(",")]
                all_documents = []
                for url in urls:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                    all_documents.extend(documents)

                vectordb = FAISS.from_documents(all_documents, OpenAIEmbeddings())
                retriever = vectordb.as_retriever()
                retriever_tool = create_retriever_tool(retriever, "custom_search", "Search for information if you find any matching keywords from the provided URLs then use this tool and provide the best fit answer from that")
                tools = [wiki_tool, arxiv_tool, retriever_tool]
            else:
                tools = [wiki_tool, arxiv_tool]

            # Load language model
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.4)

            # Set the prompt template
            if custom_prompt_template:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", custom_prompt_template),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ])
            else:
                prompt = hub.pull("hwchase17/openai-functions-agent")

            # Create the agent with memory
            agent = create_openai_tools_agent(llm, tools, prompt=prompt.partial(chat_history=st.session_state.memory.buffer))
            st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            st.success("Tools loaded successfully!")

# Main app
user_query = st.chat_input("Enter your query:")

if user_query and st.session_state.agent_executor:
    with st.spinner("Processing your query..."):
        response = st.session_state.agent_executor.invoke({"input": user_query})
        st.session_state.memory.save_context({"input": user_query}, {"chat_history": response["output"]})
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        print(st.session_state.chat_history)


    for message in st.session_state.chat_history:
      if message["role"] == "user":
            st.write(user_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)
      else:
            st.write(bot_template.replace(
                "{{MSG}}", message["content"]), unsafe_allow_html=True)