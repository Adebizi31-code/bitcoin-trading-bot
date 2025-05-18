from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Configuration de matplotlib pour afficher les caractères accentués
plt.rcParams['font.family'] = 'sans-serif'

def get_bitcoin_historical_data(days=365):
    """Récupère les données historiques de Bitcoin depuis CoinGecko"""
    print("Récupération des données historiques de Bitcoin...")
    
    cg = CoinGeckoAPI()
    
    try:
        # Récupération des données
        data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
        
        # Création du DataFrame
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Conversion des timestamps
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)
        
        # Ajout des volumes
        prices_df['volume'] = volumes_df['volume']
        
        # Sauvegarde des données
        prices_df.to_csv('btc_historical_data.csv')
        print(f"Données sauvegardées dans btc_historical_data.csv")
        
        # Création du graphique
        plt.figure(figsize=(15, 8))
        plt.plot(prices_df.index, prices_df['price'], label='Prix du Bitcoin')
        plt.title('Prix du Bitcoin sur la dernière année')
        plt.xlabel('Date')
        plt.ylabel('Prix (USD)')
        plt.grid(True)
        plt.legend()
        plt.savefig('btc_price_chart.png')
        plt.close()
        
        # Affichage des statistiques
        print("\nStatistiques principales:")
        print(f"Prix actuel : {prices_df['price'].iloc[-1]:,.2f} USD")
        print(f"Prix le plus haut : {prices_df['price'].max():,.2f} USD")
        print(f"Prix le plus bas : {prices_df['price'].min():,.2f} USD")
        print(f"Prix moyen : {prices_df['price'].mean():,.2f} USD")
        
        return prices_df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        return None

if __name__ == "__main__":
    df = get_bitcoin_historical_data()
    if df is not None:
        print(f"\nDonnées récupérées avec succès. Shape: {df.shape}")
        print("\nAperçu des données:")
        print(df.head()) 