# Bot de Trading Bitcoin avec Levier

Ce projet implémente une stratégie de trading automatisée sophistiquée pour le Bitcoin, utilisant le levier et divers indicateurs techniques.

## Caractéristiques Principales

- Capital initial : 2000 USDT
- Levier maximum : 15x
- Indicateurs techniques : EMA, RSI, MACD, Stochastique
- Stop-loss et take-profit dynamiques
- Protection du capital avec limite de drawdown à -10%
- Limite de 3 trades par jour
- Objectif de profit quotidien : 5%

## Performance

- Rendement total : +66.62%
- Ratio de Sharpe : 2.45
- Drawdown maximum : 60.61%

## Prérequis

- Python 3.8+
- Pip (gestionnaire de paquets Python)

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd bitcoin-trading-bot
```

2. Créer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du Projet

- `btc_trading_strategy.py` : Script principal de la stratégie
- `btc_data.py` : Gestion des données historiques
- `best_btc_trading_model.joblib` : Modèle optimisé sauvegardé
- `best_scaler.joblib` : Scaler pour la normalisation des données
- `optimal_strategy_parameters.json` : Paramètres optimaux de la stratégie
- `btc_historical_data.csv` : Données historiques du Bitcoin
- `btc_trading_results.png` : Visualisation des résultats
- `requirements.txt` : Liste des dépendances

## Utilisation

1. Télécharger les données historiques :
```bash
python btc_data.py
```

2. Lancer le backtest :
```bash
python btc_trading_strategy.py
```

## Gestion des Risques

- Stop-loss serré à -1%
- Protection du capital à -10% de drawdown
- Ajustement dynamique du levier selon la volatilité
- Filtres de qualité pour les signaux de trading
- Zones RSI dynamiques

## Avertissement

Le trading avec levier comporte des risques significatifs de perte. Ce bot est fourni à titre éducatif uniquement.
