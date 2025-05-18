import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt
from datetime import datetime
import json

def add_technical_indicators(df):
    """Ajout des indicateurs techniques optimisés"""
    print("Ajout des indicateurs techniques...")
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['price'], window=14).rsi()
    
    # Bandes de Bollinger
    bb = BollingerBands(close=df['price'], window=20, window_dev=2.0)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
    
    # MACD
    macd = MACD(close=df['price'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Stochastique
    stoch = StochasticOscillator(high=df['price'], low=df['price'], close=df['price'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    # ATR pour la volatilité
    df['ATR'] = AverageTrueRange(high=df['price'], low=df['price'], close=df['price']).average_true_range()
    
    return df

def analyze_trades_aggressive(df, signals, leverage=10, initial_capital=2000):
    """Analyse des trades avec gestion du risque optimisée"""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    portfolio_value = initial_capital
    max_portfolio_value = initial_capital
    
    # Paramètres de gestion des risques optimisés
    base_stop_loss_pct = -1.5  # Stop loss plus large
    base_take_profit_pct = 3.0  # Take profit plus ambitieux
    
    daily_profits = {}
    current_day_profit = 0
    max_daily_trades = 3
    daily_trades_count = {}
    
    signals_array = signals.values
    prices = df['price'].values
    dates = df.index
    
    for i in range(len(signals_array)):
        current_date = dates[i].date()
        
        # Réinitialisation des compteurs quotidiens
        if current_date not in daily_profits:
            daily_profits[current_date] = 0
            daily_trades_count[current_date] = 0
            current_day_profit = 0
        
        # Protection contre les pertes quotidiennes
        if current_day_profit <= -3:  # Arrêt si perte quotidienne > 3%
            continue
            
        # Limite de trades quotidiens
        if daily_trades_count[current_date] >= max_daily_trades:
            continue
        
        # Ajustement dynamique du levier selon la volatilité
        volatility = df['volatility'].iloc[i] if 'volatility' in df else 0.02
        dynamic_leverage = min(10, int(8 / (volatility * 100))) if volatility > 0 else 8
        
        if not in_position and signals_array[i] == 1:
            in_position = True
            entry_price = prices[i]
            entry_date = dates[i]
            
        elif in_position:
            current_price = prices[i]
            price_change_pct = (current_price - entry_price) / entry_price * 100
            leveraged_change = price_change_pct * dynamic_leverage
            
            # Conditions de sortie dynamiques
            stop_loss_hit = leveraged_change <= base_stop_loss_pct * (1 + volatility * 50)
            take_profit_hit = leveraged_change >= base_take_profit_pct * (1 + volatility * 25)
            
            if stop_loss_hit or take_profit_hit or signals_array[i] == -1:
                exit_price = current_price
                exit_date = dates[i]
                profit_pct = leveraged_change
                
                # Calcul du P&L
                trade_amount = portfolio_value
                profit_loss = trade_amount * (profit_pct / 100)
                portfolio_value += profit_loss
                
                if portfolio_value > max_portfolio_value:
                    max_portfolio_value = portfolio_value
                
                # Mise à jour des profits journaliers
                daily_profits[current_date] += profit_pct
                current_day_profit = daily_profits[current_date]
                daily_trades_count[current_date] += 1
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'portfolio_value': portfolio_value,
                    'leverage_used': dynamic_leverage,
                    'daily_profit': current_day_profit
                })
                
                in_position = False
                
                # Protection du capital
                if portfolio_value <= initial_capital * 0.93:  # Stop à -7%
                    print(f"⚠️ Protection du capital activée! Arrêt à {portfolio_value:.2f} USDT")
                    break
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Calcul des statistiques
    stats = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df['profit_pct'] > 0]),
        'losing_trades': len(trades_df[trades_df['profit_pct'] <= 0]),
        'win_rate': (len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) * 100),
        'avg_win': trades_df[trades_df['profit_pct'] > 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] <= 0]) > 0 else 0,
        'max_win': trades_df['profit_pct'].max(),
        'max_loss': trades_df['profit_pct'].min(),
        'final_portfolio': portfolio_value,
        'total_return': ((portfolio_value - initial_capital) / initial_capital) * 100,
        'max_drawdown': ((max_portfolio_value - portfolio_value) / max_portfolio_value) * 100,
        'daily_profits': daily_profits,
        'trades_df': trades_df
    }
    
    # Calcul du ratio de Sharpe
    if len(trades_df) > 0:
        returns_series = trades_df['profit_pct']
        stats['sharpe_ratio'] = (returns_series.mean() / returns_series.std() * np.sqrt(252)) if returns_series.std() != 0 else 0
    
    return stats

def generate_signals_aggressive(df, params):
    """Génère les signaux de trading avec filtres de qualité"""
    print("Génération des signaux de trading...")
    df['signal'] = 0
    
    # Paramètres RSI optimisés
    rsi_oversold = params.get('rsi_oversold', 45)
    rsi_overbought = params.get('rsi_overbought', 55)
    print(f"Paramètres RSI - Survente: {rsi_oversold}, Surachat: {rsi_overbought}")
    
    # Moyennes mobiles pour la tendance
    df['EMA_short'] = EMAIndicator(close=df['price'], window=8).ema_indicator()
    df['EMA_long'] = EMAIndicator(close=df['price'], window=21).ema_indicator()
    trend_up = df['EMA_short'] > df['EMA_long']
    
    # Calcul de la volatilité
    df['volatility'] = df['price'].pct_change().rolling(window=14).std()
    df['volatility_ma'] = df['volatility'].rolling(window=30).mean()
    
    # Vérification des données
    print("\nVérification des données:")
    print(f"Nombre de lignes: {len(df)}")
    print(f"Colonnes disponibles: {df.columns.tolist()}")
    print("\nAperçu des indicateurs:")
    print(f"RSI min: {df['RSI'].min():.2f}, max: {df['RSI'].max():.2f}")
    print(f"Stoch_k min: {df['Stoch_k'].min():.2f}, max: {df['Stoch_k'].max():.2f}")
    print(f"Volatilité moyenne: {df['volatility'].mean():.4f}")
    
    # Conditions d'entrée simplifiées
    price_near_bb_low = df['price'] < df['BB_low'] * 1.02  # Prix proche de la bande basse
    rsi_condition = (df['RSI'] < rsi_oversold) | ((df['RSI'] < 50) & (df['RSI'] > df['RSI'].shift(1)))  # RSI en survente ou croissant
    macd_condition = (df['MACD'] > df['MACD_signal']) | (df['MACD'] > df['MACD'].shift(1))  # MACD haussier ou croissant
    stoch_condition = (df['Stoch_k'] < 40) | (df['Stoch_k'] > df['Stoch_k'].shift(1))  # Stochastique bas ou croissant
    
    # Combinaison des conditions d'entrée (plus souples)
    long_condition = (
        price_near_bb_low &  # Prix proche de la bande basse
        (rsi_condition | macd_condition) &  # RSI favorable OU MACD favorable
        (stoch_condition | trend_up)  # Stochastique favorable OU tendance haussière
    )
    
    # Conditions de sortie simplifiées
    price_near_bb_high = df['price'] > df['BB_high'] * 0.98  # Prix proche de la bande haute
    rsi_high = df['RSI'] > rsi_overbought
    macd_bearish = df['MACD'] < df['MACD_signal']
    stoch_high = df['Stoch_k'] > 60
    
    # Combinaison des conditions de sortie (plus souples)
    exit_condition = (
        (price_near_bb_high & rsi_high) |  # Prix haut + RSI haut
        (macd_bearish & stoch_high) |  # MACD baissier + Stochastique haut
        (df['price'] < df['EMA_long'])  # Prix sous la moyenne longue
    )
    
    # Application des signaux
    df.loc[long_condition, 'signal'] = 1
    df.loc[exit_condition & (df['signal'].shift(1) == 1), 'signal'] = -1
    
    # Statistiques des signaux
    total_signals = len(df[df['signal'] != 0])
    long_signals = len(df[df['signal'] == 1])
    exit_signals = len(df[df['signal'] == -1])
    
    print("\nStatistiques des signaux générés:")
    print(f"Total des signaux: {total_signals}")
    print(f"Signaux d'entrée: {long_signals}")
    print(f"Signaux de sortie: {exit_signals}")
    
    if total_signals == 0:
        print("\nDétails des conditions:")
        print(f"Prix proche BB basse: {len(df[price_near_bb_low])}")
        print(f"Condition RSI: {len(df[rsi_condition])}")
        print(f"Condition MACD: {len(df[macd_condition])}")
        print(f"Condition Stochastique: {len(df[stoch_condition])}")
        print(f"Tendance haussière: {len(df[trend_up])}")
    
    return df['signal']

def optimize_strategy_aggressive(df):
    """Optimisation des paramètres de la stratégie"""
    best_params = None
    best_return = float('-inf')
    best_stats = None
    
    # Grille de paramètres à tester
    param_grid = {
        'rsi_oversold': [40, 45, 50],
        'rsi_overbought': [50, 55, 60]
    }
    
    print("\nOptimisation des paramètres...")
    total_combinations = len(param_grid['rsi_oversold']) * len(param_grid['rsi_overbought'])
    current_combination = 0
    
    for rsi_oversold in param_grid['rsi_oversold']:
        for rsi_overbought in param_grid['rsi_overbought']:
            current_combination += 1
            if rsi_oversold >= rsi_overbought:
                continue
                
            print(f"\nTest de la combinaison {current_combination}/{total_combinations}")
            print(f"RSI survente: {rsi_oversold}, RSI surachat: {rsi_overbought}")
            
            params = {
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought
            }
            
            signals = generate_signals_aggressive(df.copy(), params)
            stats = analyze_trades_aggressive(df, signals)
            
            if stats and stats['total_return'] > best_return:
                best_return = stats['total_return']
                best_params = params
                best_stats = stats
                
                print("\nNouveaux meilleurs paramètres trouvés!")
                print(f"RSI survente: {rsi_oversold}")
                print(f"RSI surachat: {rsi_overbought}")
                print(f"Rendement total: {best_return:.2f}%")
                print(f"Nombre de trades: {stats['total_trades']}")
                print(f"Ratio de Sharpe: {stats['sharpe_ratio']:.2f}")
                print("---")
    
    if best_params:
        # Sauvegarde des meilleurs paramètres
        with open('optimal_strategy_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        print("\nMeilleurs paramètres sauvegardés dans optimal_strategy_parameters.json")
    
    return best_params, best_stats

def plot_trading_results(stats, title="Résultats du Trading"):
    """Visualisation des résultats de trading"""
    if not stats or 'trades_df' not in stats:
        print("Pas de données de trading à afficher")
        return
    
    trades_df = stats['trades_df']
    
    # Création du graphique
    plt.figure(figsize=(15, 10))
    
    # Évolution du portefeuille
    plt.subplot(2, 1, 1)
    plt.plot(trades_df['exit_date'], trades_df['portfolio_value'], label='Valeur du portefeuille')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Valeur du portefeuille (USDT)')
    plt.grid(True)
    plt.legend()
    
    # Distribution des profits
    plt.subplot(2, 1, 2)
    plt.hist(trades_df['profit_pct'], bins=50, alpha=0.75)
    plt.title('Distribution des profits')
    plt.xlabel('Profit (%)')
    plt.ylabel('Nombre de trades')
    plt.grid(True)
    
    # Ajout des statistiques
    stats_text = f"""
    Nombre total de trades: {stats['total_trades']}
    Taux de réussite: {stats['win_rate']:.2f}%
    Rendement total: {stats['total_return']:.2f}%
    Ratio de Sharpe: {stats['sharpe_ratio']:.2f}
    Drawdown maximum: {stats['max_drawdown']:.2f}%
    """
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig('btc_trading_results_optimized.png')
    plt.close()

def main():
    """Fonction principale"""
    print("Chargement des données...")
    try:
        df = pd.read_csv('btc_historical_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        print("Préparation des données...")
        df = add_technical_indicators(df)
        
        print("Optimisation de la stratégie...")
        best_params, best_stats = optimize_strategy_aggressive(df)
        
        if best_stats:
            print("\nMeilleurs résultats trouvés:")
            print(f"Nombre total de trades: {best_stats['total_trades']}")
            print(f"Taux de réussite: {best_stats['win_rate']:.2f}%")
            print(f"Rendement total: {best_stats['total_return']:.2f}%")
            print(f"Ratio de Sharpe: {best_stats['sharpe_ratio']:.2f}")
            print(f"Drawdown maximum: {best_stats['max_drawdown']:.2f}%")
            
            plot_trading_results(best_stats, "Résultats du Trading Optimisé")
            print("\nGraphique des résultats sauvegardé dans 'btc_trading_results_optimized.png'")
        else:
            print("Aucun trade valide trouvé avec les paramètres testés")
    except Exception as e:
        print(f"Erreur lors du traitement des données: {e}")

if __name__ == "__main__":
    main() 