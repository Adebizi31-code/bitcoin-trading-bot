# Stratégie de Trading Bitcoin

Ce projet implémente une stratégie de trading automatisée pour le Bitcoin utilisant différents indicateurs techniques.

## Fonctionnalités

- Téléchargement automatique des données Bitcoin via CoinGecko API
- Calcul d'indicateurs techniques (SMA, EMA, MACD, RSI, Bollinger Bands)
- Génération de signaux de trading
- Analyse détaillée des performances
- Visualisation des trades et de la performance

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer la stratégie :
```bash
python btc_trading_strategy.py
```

## Résultats

La stratégie analyse les données Bitcoin sur un an et génère :
- Des signaux d'achat/vente basés sur les indicateurs techniques
- Une analyse détaillée des trades (gains, pertes, statistiques)
- Des graphiques de performance
- Un suivi du capital

## Structure du projet

- `btc_trading_strategy.py` : Script principal
- `requirements.txt` : Dépendances Python
- `btc_trading_results.png` : Graphique des résultats (généré)
- `trading_results_detailed.json` : Statistiques détaillées (généré) 