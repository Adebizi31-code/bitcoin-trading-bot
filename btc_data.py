from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Configuration de matplotlib pour afficher les caractères accentués
plt.rcParams['font.family'] = 'sans-serif'

# Initialisation de l'API CoinGecko
cg = CoinGeckoAPI()

# Calcul des dates (dernière année seulement)
end_date = int(time.time())
start_date = end_date - (365 * 24 * 60 * 60)  # 1 an en secondes

# Récupération des données
print("Téléchargement des données du Bitcoin pour la dernière année...")
bitcoin_data = cg.get_coin_market_chart_range_by_id(
    id='bitcoin',
    vs_currency='usd',
    from_timestamp=start_date,
    to_timestamp=end_date
)

# Conversion en DataFrame
prices_df = pd.DataFrame(bitcoin_data['prices'], columns=['timestamp', 'price'])
prices_df['date'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
prices_df.set_index('date', inplace=True)

# Sauvegarde des données en CSV
prices_df.to_csv('btc_historical_data.csv')

# Création du graphique
plt.figure(figsize=(15, 8))
plt.plot(prices_df.index, prices_df['price'], label='Prix du Bitcoin')
plt.title('Prix du Bitcoin sur la dernière année')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.grid(True)
plt.legend()

# Sauvegarde du graphique
plt.savefig('btc_price_chart.png')
print("Données et graphique sauvegardés avec succès !")

# Affichage des statistiques principales
print("\nStatistiques principales :")
print(f"Prix actuel : {prices_df['price'].iloc[-1]:.2f} USD")
print(f"Prix le plus haut : {prices_df['price'].max():.2f} USD")
print(f"Prix le plus bas : {prices_df['price'].min():.2f} USD")
print(f"Prix moyen : {prices_df['price'].mean():.2f} USD") 