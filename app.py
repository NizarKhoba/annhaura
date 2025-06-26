import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from random import shuffle

# --- Streamlit Setup ---
st.set_page_config(page_title="🎵 Mood Song Finder", page_icon="🎶")
st.title("🎶 Mood Song Finder")
st.markdown("Find songs that match your mood. Powered by Gemini + LangChain.")

# --- API Key Input ---
api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Enter your **Google API Key**", type="password")
if not api_key:
    st.warning("Please enter your API Key to continue.", icon="🔑")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Load Models ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Load and Process Dataset ---
csv_url = "https://raw.githubusercontent.com/annhaura/mood-song-finder/main/spotify_songs.csv"
with st.spinner("📥 Loading dataset..."):
    df = pd.read_csv(csv_url).head(300)
    df["combined_text"] = df.apply(lambda row: f"{row['track_name']} by {row['track_artist']}", axis=1)
    documents = [Document(page_content=text) for text in df["combined_text"]]

with st.spinner("🔍 Indexing songs..."):
    vectorstore = FAISS.from_documents(documents, embedding_model)

# --- Mood Detection (Hybrid) ---
def classify_mood(query: str) -> str:
    query_lower = query.lower()
    # Rule-based shortcut
    if "happy" in query_lower:
        return "happy"
    elif "sad" in query_lower:
        return "sad"
    elif "nostalgic" in query_lower or "miss" in query_lower:
        return "nostalgic"
    elif "love" in query_lower or "romantic" in query_lower:
        return "romantic"
    elif "energetic" in query_lower or "workout" in query_lower:
        return "energetic"
    else:
        # Fallback to LLM
        try:
            prompt = f"Classify the emotional mood of this sentence into one of the following moods: happy, sad, nostalgic, romantic, energetic.\n\nText: {query}"
            result = llm.invoke(prompt)
            return result.content.strip().lower()
        except Exception as e:
            return "unknown"

# --- Song Retrieval ---
def get_recommendations(query: str, k: int = 5) -> list:
    try:
        results = vectorstore.similarity_search(query, k=k)
        return [f"🎵 {doc.page_content}" for doc in results]
    except Exception as e:
        return [f"[Error retrieving songs: {e}]"]

# --- Randomizer ---
def randomize_list(song_list: list) -> list:
    shuffle(song_list)
    return song_list

# --- Streamlit Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What kind of music do you want to hear today?")
if user_input:
    with st.spinner("🔎 Detecting mood..."):
        mood = classify_mood(user_input)

    st.markdown(f"**Detected mood:** `{mood}`" if mood != "unknown" else "Could not detect mood. Searching by your input.")

    search_query = mood if mood != "unknown" else user_input

    with st.spinner("🎧 Finding songs..."):
        recommendations = get_recommendations(search_query)
        recommendations = randomize_list(recommendations)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", "\n".join(recommendations)))

# --- Display Chat History ---
for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)
