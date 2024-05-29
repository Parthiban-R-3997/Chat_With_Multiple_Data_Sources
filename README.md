# Chat_With_Multiple_Data_Sources

This Streamlit web application leverages LangChain to provide an advanced query assistant that integrates multiple tools and data sources. Users can query information from Wikipedia, arXiv, and custom URLs. The application also supports custom prompts and maintains conversation history.


## Deployed Link
Document Q&A app is Deployed And Available [Here](https://huggingface.co/spaces/Parthiban97/Chat_With_Multiple_Data_Sources)


## Screenshots
![data3_2](https://github.com/Parthiban-R-3997/Chat_Groq_Document_Q_A/assets/26496805/4e7b3d1b-c660-4340-88e0-3cf91c1c8d57)
![data3_3](https://github.com/Parthiban-R-3997/Chat_Groq_Document_Q_A/assets/26496805/d5ee0b91-8121-48ea-9fa4-6aff0bc68ae9)
![data3_1](https://github.com/Parthiban-R-3997/Chat_Groq_Document_Q_A/assets/26496805/0aba916f-d173-4f9a-900d-a2328992d44c)


## Features
- **Integration with Wikipedia and arXiv**: Retrieve information directly from Wikipedia and arXiv.
- **Custom URL Support**: Load documents from custom URLs seperated by commas (,) and search within them.
- **Conversation History**: Maintains a conversation history using LangChain's ConversationBufferMemory.
- **Custom Prompts**: Users can define their custom prompts to tailor the assistant's responses.

### LangChain Toolkits
LangChain toolkits provide a set of utilities and pre-built components that simplify the interaction with language models:

- **Document Loaders**: Tools to load documents from various sources. For example, `WebBaseLoader` allows loading web pages and splitting them into manageable chunks.
- **Text Splitters**: Components like `RecursiveCharacterTextSplitter` help divide documents into chunks of a specified size, optimizing them for processing.
- **Embeddings**: `OpenAIEmbeddings` generate vector representations of text, useful for similarity searches and information retrieval.
- **Vector Stores**: `FAISS` is used to store and query vector embeddings efficiently.

### LangChain Agents
LangChain agents are orchestrators that manage the interaction between the language model and various tools:

- **create_openai_tools_agent**: This function creates an agent that can interact with multiple tools using an OpenAI language model. The agent is configured with a prompt template and can utilize tools like WikipediaQueryRun, ArxivQueryRun, and custom URL retrievers.
- **AgentExecutor**: Manages the execution of the agent, handling user inputs and generating responses.

## Components

### Main Components
1. **ChatOpenAI**: The main language model used for generating responses.
2. **ConversationBufferMemory**: Stores the conversation history, allowing the agent to maintain context across multiple interactions.
3. **WikipediaQueryRun**: A tool for retrieving information from Wikipedia.
4. **ArxivQueryRun**: A tool for retrieving information from arXiv.
5. **WebBaseLoader**: Loads documents from specified URLs.
6. **FAISS**: Vector store for managing document embeddings.
7. **ChatPromptTemplate**: Defines the structure of prompts used by the language model.

### Configuration
The application allows users to configure the following settings via the sidebar:

- **OpenAI API Key**: For accessing OpenAI's language models.
- **Custom URLs**: Users can provide URLs to load and search documents.
- **Custom Prompts**: Define custom prompt templates for tailored responses.

### Usage
Users can enter their queries in the main input area. The agent processes the query using the configured tools and returns the response. The conversation history is displayed, showing interactions between the user and the assistant.

### Memory Management
The application uses `ConversationBufferMemory` to retain the conversation history, ensuring context is preserved across multiple user queries. This memory is updated with each interaction, providing a coherent and continuous conversation experience.

## Example Usage
1. Enter your OpenAI API Key in the sidebar.
2. Optionally, provide URLs to load custom documents.
3. Define custom prompts if needed.
4. Click "Load Tools" to initialize the agent.
5. Enter your query in the main input area and receive a response based on the integrated tools and data sources.
