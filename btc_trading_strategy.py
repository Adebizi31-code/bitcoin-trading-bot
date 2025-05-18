import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
import time
from datetime import datetime, timedelta

def add_technical_indicators(df):
    print("Ajout des indicateurs techniques...")
    
    # Tendance
    df['SMA_20'] = SMAIndicator(close=df['price'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['price'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['price'], window=20).ema_indicator()
    
    # MACD
    macd = MACD(close=df['price'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['price']).rsi()
    
    # Bollinger
    bollinger = BollingerBands(close=df['price'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    
    # Rendements
    df['returns'] = df['price'].pct_change()
    
    return df

def analyze_trades(df, signals):
    """Analyse des trades"""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    
    for i in range(len(signals)):
        if not in_position and signals[i] == 1:
            in_position = True
            entry_price = df['price'].iloc[i]
            entry_date = df.index[i]
        elif in_position and signals[i] == 0:
            exit_price = df['price'].iloc[i]
            exit_date = df.index[i]
            profit_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct
            })
            in_position = False
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    stats = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df['profit_pct'] > 0]),
        'losing_trades': len(trades_df[trades_df['profit_pct'] <= 0]),
        'avg_win': trades_df[trades_df['profit_pct'] > 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] <= 0]) > 0 else 0,
        'max_win': trades_df['profit_pct'].max(),
        'max_loss': trades_df['profit_pct'].min(),
        'trades_df': trades_df
    }
    
    stats['win_rate'] = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
    
    return stats

print("Téléchargement des données Bitcoin...")
cg = CoinGeckoAPI()
end_date = int(time.time())
start_date = end_date - (365 * 24 * 60 * 60)

bitcoin_data = cg.get_coin_market_chart_range_by_id(
    id='bitcoin',
    vs_currency='usd',
    from_timestamp=start_date,
    to_timestamp=end_date
)

# Préparation des données
df = pd.DataFrame(bitcoin_data['prices'], columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
df = df.drop('timestamp', axis=1)

# Ajout des indicateurs techniques
df = add_technical_indicators(df)
df = df.dropna()

# Création des signaux
print("Génération des signaux de trading...")
df['signal'] = 0
df.loc[(df['price'] > df['SMA_20']) & (df['RSI'] < 70) & (df['MACD'] > df['MACD_signal']), 'signal'] = 1
df.loc[(df['price'] < df['SMA_20']) & (df['RSI'] > 30) & (df['MACD'] < df['MACD_signal']), 'signal'] = -1

# Analyse des trades
signals = (df['signal'] == 1).astype(int)
trade_analysis = analyze_trades(df, signals)

print("\n=== RÉSULTATS DE LA STRATÉGIE SUR 1 AN ===")
print(f"\nNombre total de trades: {trade_analysis['total_trades']}")
print(f"Trades gagnants: {trade_analysis['winning_trades']}")
print(f"Trades perdants: {trade_analysis['losing_trades']}")
print(f"Taux de réussite: {trade_analysis['win_rate']:.2f}%")

print(f"\nGain moyen sur trades gagnants: +{trade_analysis['avg_win']:.2f}%")
print(f"Perte moyenne sur trades perdants: {trade_analysis['avg_loss']:.2f}%")
print(f"Plus grand gain: +{trade_analysis['max_win']:.2f}%")
print(f"Plus grande perte: {trade_analysis['max_loss']:.2f}%")

# Calcul de la performance globale
initial_capital = 100000  # 100,000 USD
portfolio_value = initial_capital
trade_sizes = 0.1  # 10% du capital par trade

for _, trade in trade_analysis['trades_df'].iterrows():
    trade_amount = portfolio_value * trade_sizes
    profit_loss = trade_amount * (trade['profit_pct'] / 100)
    portfolio_value += profit_loss

total_return = ((portfolio_value - initial_capital) / initial_capital) * 100

print(f"\nPerformance avec capital initial de {initial_capital:,.2f} USD:")
print(f"Capital final: {portfolio_value:,.2f} USD")
print(f"Rendement total: {total_return:.2f}%")
print(f"Rendement mensuel moyen: {total_return/12:.2f}%")

# Visualisation
plt.figure(figsize=(15, 10))

# Prix et signaux
plt.subplot(2, 1, 1)
plt.plot(df.index, df['price'], label='Prix BTC', alpha=0.7)

# Points d'entrée/sortie
for _, trade in trade_analysis['trades_df'].iterrows():
    plt.scatter(trade['entry_date'], trade['entry_price'], color='g', marker='^', s=100)
    plt.scatter(trade['exit_date'], trade['exit_price'], color='r', marker='v', s=100)

plt.title('Bitcoin - Prix et Signaux de Trading')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.legend(['Prix', 'Achat', 'Vente'])
plt.grid(True)

# Performance du portefeuille
plt.subplot(2, 1, 2)
portfolio_values = []
current_value = initial_capital

for _, trade in trade_analysis['trades_df'].iterrows():
    trade_amount = current_value * trade_sizes
    profit_loss = trade_amount * (trade['profit_pct'] / 100)
    current_value += profit_loss
    portfolio_values.append({
        'date': trade['exit_date'],
        'value': current_value
    })

portfolio_df = pd.DataFrame(portfolio_values)
if not portfolio_df.empty:
    plt.plot(portfolio_df['date'], portfolio_df['value'])
    plt.title('Évolution du Capital')
    plt.xlabel('Date')
    plt.ylabel('Capital (USD)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('btc_trading_results.png')
print("\nLe graphique a été sauvegardé dans 'btc_trading_results.png'") 