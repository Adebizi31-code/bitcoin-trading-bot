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

def analyze_trades_with_leverage(df, signals, leverage=4):
    """Analyse détaillée des trades avec levier"""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    
    # Stop loss et take profit en pourcentage (adaptés pour le levier)
    stop_loss_pct = -5.0  # Stop loss à -5% (soit -20% avec levier x4)
    take_profit_pct = 7.5  # Take profit à +7.5% (soit +30% avec levier x4)
    
    for i in range(len(signals)):
        if not in_position and signals[i] == 1:
            in_position = True
            entry_price = df['price'].iloc[i]
            entry_date = df.index[i]
        elif in_position:
            current_price = df['price'].iloc[i]
            price_change_pct = (current_price - entry_price) / entry_price * 100
            
            # Vérification du stop loss et take profit
            if price_change_pct <= stop_loss_pct or price_change_pct >= take_profit_pct or signals[i] == 0:
                exit_price = current_price
                exit_date = df.index[i]
                profit_pct = price_change_pct * leverage  # Application du levier
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'holding_period': (exit_date - entry_date).days,
                    'exit_reason': 'Stop Loss' if price_change_pct <= stop_loss_pct else 'Take Profit' if price_change_pct >= take_profit_pct else 'Signal'
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
        'avg_holding_period': trades_df['holding_period'].mean(),
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

# Création des signaux avec paramètres plus stricts
print("Génération des signaux de trading (version levier x4)...")
df['signal'] = 0

# Conditions d'entrée plus strictes pour le trading avec levier
df.loc[(df['price'] > df['SMA_20']) & 
       (df['price'] > df['SMA_50']) &  # Tendance haussière confirmée
       (df['RSI'] > 40) & (df['RSI'] < 65) &  # Zone neutre du RSI
       (df['MACD'] > df['MACD_signal']) & 
       (df['price'] > df['BB_low']), 'signal'] = 1

df.loc[(df['price'] < df['SMA_20']) |
       (df['RSI'] > 70) |
       (df['MACD'] < df['MACD_signal']), 'signal'] = -1

# Analyse des trades avec levier x4
signals = (df['signal'] == 1).astype(int)
trade_analysis = analyze_trades_with_leverage(df, signals, leverage=4)

print("\n=== RÉSULTATS DÉTAILLÉS DE LA STRATÉGIE AVEC LEVIER x4 SUR 1 AN ===")
print("\nPARAMÈTRES DE TRADING:")
print("Levier: x4")
print("Capital utilisé par trade: 100%")
print("Stop Loss: -5% (équivalent à -20% avec levier)")
print("Take Profit: +7.5% (équivalent à +30% avec levier)")

print("\nSTATISTIQUES DES TRADES:")
print(f"Nombre total de trades: {trade_analysis['total_trades']}")
print(f"Trades gagnants: {trade_analysis['winning_trades']}")
print(f"Trades perdants: {trade_analysis['losing_trades']}")
print(f"Taux de réussite: {trade_analysis['win_rate']:.2f}%")
print(f"Durée moyenne de détention: {trade_analysis['avg_holding_period']:.1f} jours")

print("\nPERFORMANCE DES TRADES (AVEC LEVIER x4):")
print(f"Gain moyen sur trades gagnants: +{trade_analysis['avg_win']:.2f}%")
print(f"Perte moyenne sur trades perdants: {trade_analysis['avg_loss']:.2f}%")
print(f"Plus grand gain: +{trade_analysis['max_win']:.2f}%")
print(f"Plus grande perte: {trade_analysis['max_loss']:.2f}%")

# Calcul de la performance avec 1000 USD et levier x4
initial_capital = 1000  # 1,000 USD
portfolio_value = initial_capital
trade_sizes = 1.0  # 100% du capital par trade

cumulative_returns = []
dates = []
trade_profits = []

for _, trade in trade_analysis['trades_df'].iterrows():
    trade_amount = portfolio_value * trade_sizes
    profit_loss = trade_amount * (trade['profit_pct'] / 100)
    portfolio_value += profit_loss
    
    if portfolio_value <= 0:  # Vérification de la faillite
        print("\n⚠️ ATTENTION: Faillite détectée! Le capital a atteint ou dépassé 0.")
        break
    
    trade_profits.append({
        'date': trade['exit_date'],
        'profit_usd': profit_loss,
        'profit_pct': trade['profit_pct'],
        'portfolio_value': portfolio_value,
        'exit_reason': trade['exit_reason']
    })
    
    cumulative_returns.append(portfolio_value)
    dates.append(trade['exit_date'])

print("\nPERFORMANCE DÉTAILLÉE DU PORTEFEUILLE:")
print(f"Capital initial: {initial_capital:,.2f} USD")
print(f"Capital final: {portfolio_value:,.2f} USD")
total_return = ((portfolio_value - initial_capital) / initial_capital) * 100
print(f"Rendement total: {total_return:.2f}%")
print(f"Rendement mensuel moyen: {total_return/12:.2f}%")

# Calcul des métriques de risque
if trade_profits:
    profits_df = pd.DataFrame(trade_profits)
    monthly_returns = profits_df.groupby(profits_df['date'].dt.to_period('M'))['profit_pct'].sum()
    
    print("\nMÉTRIQUES DE RISQUE:")
    print(f"Volatilité mensuelle: {monthly_returns.std():.2f}%")
    if monthly_returns.std() != 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std() * np.sqrt(12))
        print(f"Ratio de Sharpe: {sharpe:.2f}")
    
    drawdowns = []
    peak = initial_capital
    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        drawdowns.append(drawdown)
    
    print(f"Drawdown maximum: {max(drawdowns):.2f}%")

# Visualisation améliorée
plt.figure(figsize=(15, 12))

# Prix et signaux
plt.subplot(3, 1, 1)
plt.plot(df.index, df['price'], label='Prix BTC', alpha=0.7)
plt.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.6)
plt.plot(df.index, df['BB_high'], label='BB Haut', alpha=0.4)
plt.plot(df.index, df['BB_low'], label='BB Bas', alpha=0.4)

for _, trade in trade_analysis['trades_df'].iterrows():
    color = 'g' if trade['profit_pct'] > 0 else 'r'
    plt.scatter(trade['entry_date'], trade['entry_price'], color=color, marker='^', s=100)
    plt.scatter(trade['exit_date'], trade['exit_price'], color=color, marker='v', s=100)

plt.title('Bitcoin - Prix et Signaux de Trading (Levier x4)')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.legend()
plt.grid(True)

# Performance du portefeuille
plt.subplot(3, 1, 2)
plt.plot(dates, cumulative_returns, label='Valeur du Portefeuille')
plt.title('Évolution du Capital (1000 USD initial, Levier x4)')
plt.xlabel('Date')
plt.ylabel('Capital (USD)')
plt.legend()
plt.grid(True)

# Drawdown
plt.subplot(3, 1, 3)
plt.plot(dates, drawdowns, label='Drawdown', color='red')
plt.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
plt.title('Drawdown (%)')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('btc_trading_results_leveraged.png')
print("\nLe graphique détaillé a été sauvegardé dans 'btc_trading_results_leveraged.png'") 