import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Music AI", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
pop_model = pickle.load(open("popularity_model.pkl", "rb"))
mood_model = pickle.load(open("mood_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
df = pickle.load(open("songs.pkl", "rb"))
recommender = pickle.load(open("recommender.pkl", "rb"))

features = ["energy", "danceability", "tempo", "valence", "loudness"]

# -----------------------------
# TITLE
# -----------------------------
st.title("🎵 Smart Music Intelligence System")
st.subheader("AI-Based Mood Prediction & Recommendation System")

# -----------------------------
# SEARCH
# -----------------------------
search = st.text_input("🔍 Search Song")

filtered = df[df["song_name"].str.contains(search, case=False, na=False)]
song_list = filtered["song_name"].values if len(filtered) > 0 else df["song_name"].values

selected_song = st.selectbox("Select Song", song_list)
song_index = df[df["song_name"] == selected_song].index[0]

# -----------------------------
# SLIDERS
# -----------------------------
st.subheader("🎛️ Adjust Features")

col1, col2, col3 = st.columns(3)

with col1:
    energy = st.slider("Energy", 0.0, 1.0, float(df.loc[song_index, "energy"]))
    dance = st.slider("Danceability", 0.0, 1.0, float(df.loc[song_index, "danceability"]))

with col2:
    tempo = st.slider("Tempo", 0.0, 1.0, float(df.loc[song_index, "tempo"]))
    valence = st.slider("Valence", 0.0, 1.0, float(df.loc[song_index, "valence"]))

with col3:
    loud = st.slider("Loudness", 0.0, 1.0, float(df.loc[song_index, "loudness"]))

# -----------------------------
# INPUT PROCESSING
# -----------------------------
input_df = pd.DataFrame([[energy, dance, tempo, valence, loud]], columns=features)

scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(scaled, columns=features)

# -----------------------------
# SHOW FEATURE VALUES
# -----------------------------
st.write("### 🔍 Selected Feature Values")
st.write(input_df)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🎯 Predict"):

    X_pred = input_scaled

    pop = pop_model.predict(X_pred)[0]
    mood = mood_model.predict(X_pred)[0]
    probs = mood_model.predict_proba(X_pred)[0]

    st.success(f"🎧 Predicted Mood: {mood}")
    st.success(f"📈 Predicted Popularity: {pop:.2f}")

    # Confidence
    st.info(f"Confidence: {np.max(probs)*100:.2f}%")

    # Graph
    fig, ax = plt.subplots()
    ax.bar(mood_model.classes_, probs)
    ax.set_title("Mood Probability Distribution")
    ax.set_xlabel("Mood")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# RECOMMENDATION
# -----------------------------
st.subheader("🎶 Similar Songs")

def recommend(song_index):
    song_vec = df.loc[[song_index], features]
    distances, indices = recommender.kneighbors(song_vec)

    return [(df.iloc[i]["song_name"], 1 - d)
            for i, d in zip(indices[0][1:], distances[0][1:])]

for song, score in recommend(song_index):
    st.write(f"🎵 {song}  | Similarity: {score:.2f}")

# -----------------------------
# LIKES
# -----------------------------
st.subheader("❤️ Your Likes")

if "liked" not in st.session_state:
    st.session_state.liked = []

if st.button("❤️ Like this song"):
    if song_index not in st.session_state.liked:
        st.session_state.liked.append(song_index)

st.write([df.iloc[i]["song_name"] for i in st.session_state.liked])

# -----------------------------
# SMART RECOMMENDATION
# -----------------------------
st.subheader("🔥 Recommended For You")

def recommend_from_likes(liked):
    if not liked:
        return []

    vectors = df.loc[liked, features]
    avg = vectors.mean().to_frame().T

    distances, indices = recommender.kneighbors(avg)

    return df.iloc[indices[0][1:6]]["song_name"].tolist()

recs = recommend_from_likes(st.session_state.liked)

if not recs:
    st.info("👉 Like songs to get recommendations")
else:
    for r in recs:
        st.write("👉", r)
