import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from random import shuffle

# --- Streamlit Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood. Powered by LangChain + Gemini LLM.")

# --- API Key Input / Secret ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- Load LLM & Memory ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Load Dataset (limit for speed) ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-song-finder/main/spotify_songs.csv"
with st.spinner("📥 Loading dataset..."):
    df = pd.read_csv(csv_url).head(300)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    documents = [Document(page_content=text, metadata={"index": i}) for i, text in enumerate(df["combined_text"])]

# --- Embedding & Vectorstore ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
with st.spinner("🔍 Creating vector index..."):
    vectorstore = FAISS.from_documents(documents, embedding_model)

# --- Tool Definitions ---
def classify_mood(query: str) -> str:
    prompt = f"Classify the emotional mood of this text (examples: happy, sad, nostalgic, energetic, romantic):\n\n{query}"
    try:
        return llm.invoke(prompt).content.strip().lower()
    except Exception as e:
        return "[error]"

def infer_genre(query: str) -> str:
    prompt = f"Suggest a suitable music genre for this mood or query:\n\n{query}"
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "pop"

def retrieve_similar_songs(query: str, k=3) -> str:
    try:
        results = vectorstore.similarity_search(query, k=k)
        songs = [f"🎵 {doc.page_content}" for doc in results]
        return "\n".join(songs) if songs else "No songs found for that mood."
    except Exception as e:
        return f"[Error retrieving songs: {e}]"

def randomize_list(text_block: str) -> str:
    lines = text_block.strip().splitlines()
    shuffle(lines)
    return "\n".join(lines)

# --- Tools List ---
tools = [
    Tool(name="MoodClassifier", func=classify_mood, description="Detects emotional mood from user input."),
    Tool(name="InferGenre", func=infer_genre, description="Suggests a suitable music genre."),
    Tool(name="RetrieveSimilarSongs", func=retrieve_similar_songs, description="Finds matching songs from the dataset."),
    Tool(name="Randomizer", func=randomize_list, description="Randomizes the order of song recommendations."),
]

# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

# --- Streamlit Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

k_value = st.sidebar.slider("Number of Songs to Recommend", 1, 10, 3)

user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🤖 Thinking..."):
        try:
            response = agent.run(user_input)
        except Exception as e:
            # fallback kalau agent gagal
            st.warning("LLM quota may be exceeded. Falling back to similarity search.")
            response = retrieve_similar_songs(user_input, k=k_value)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", response))

# --- Display Chat History ---
for speaker, text in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(text)
