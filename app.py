import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Wyrocznia Słodyczy AI", page_icon="🍬", layout="centered")

st.title("🔮 Wyrocznia AI: Klasyfikator Słodyczy")
st.markdown("Wpisz parametry zmyślonego cukierka, a mój model Sztucznej Inteligencji zgadnie, do jakiej kategorii należy!")

@st.cache_resource
def trenuj_model():
    df = pd.read_csv('cukrasy_clean.csv')
    X = df[['szerokosc', 'dlugosc', 'twardosc', 'kwas', 'smaczek']]
    y = df['kategoria'].str.strip()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = trenuj_model()

st.subheader("⚙️ Ustaw parametry:")
szerokosc = st.slider("📐 Szerokość [mm]", 1.0, 50.0, 20.0, step=0.5)
dlugosc = st.slider("📏 Długość [mm]", 5.0, 200.0, 50.0, step=1.0)
twardosc = st.slider("🧱 Twardość (-1 miękki do 1 twardy)", -1.0, 1.0, 0.0, step=0.1)
kwas = st.slider("🍋 Kwasowość (-1 słodki do 1 kwaśny)", -1.0, 1.0, 0.0, step=0.1)
smaczek = st.slider("⭐ Ocena smaku (1 do 10)", 1.0, 10.0, 6.0, step=0.5)

if st.button("🤖 Zapytaj Wyrocznię AI", use_container_width=True):
    dane_testowe = np.array([[szerokosc, dlugosc, twardosc, kwas, smaczek]])
    predykcja = model.predict(dane_testowe)[0]
    prawdopodobienstwa = model.predict_proba(dane_testowe)[0]
    pewnosc = np.max(prawdopodobienstwa) * 100
    
    st.divider()
    st.success(f"### 👉 Według AI to kategoria: **{predykcja.upper()}**")
    st.info(f"🎯 Algorytm jest pewien swojej decyzji na: **{pewnosc:.1f}%**")
    st.balloons()
  
