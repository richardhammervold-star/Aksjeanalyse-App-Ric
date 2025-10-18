# Aksjeanalyse – Streamlit-app

Denne appen viser sannsynlighet for at en aksje stiger (basert på en enkel logistisk regresjon),
modellens treffsikkerhet, anbefaling (KJØP/HOLD/SELG), og grafer.

## Kjør lokalt
```bash
pip install -r requirements.txt
streamlit run aksje_prediksjon_app.py
```

## Publiser gratis på Streamlit Cloud
1. Last opp denne mappen til et GitHub-repo
2. Gå til https://streamlit.io/cloud → New app
3. Velg repoet → `main` branch → main file: `aksje_prediksjon_app.py`
4. Deploy

Ferdig URL kan åpnes på mobil og PC.