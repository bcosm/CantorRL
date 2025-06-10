# QuantConnect RL Hedging Strategy

This directory contains the QuantConnect implementation of your trained reinforcement learning hedging strategy.

## Files Overview

- **`main.py`** - Main QuantConnect algorithm implementation
- **`model_wrapper.py`** - Wrapper for loading and using your trained RL model
- **`option_calculator.py`** - Black-Scholes pricing and Greeks calculations
- **`prepare_model.py`** - Utility to extract and prepare your trained model for deployment
- **`test_local.py`** - Local testing script to validate components before deployment
- **`config.json`** - Project configuration parameters

## Quick Start

### 1. Test Locally First
```bash
cd quantconnect
python test_local.py
```

### 2. Prepare Your Trained Model
```bash
python prepare_model.py
```

This will:
- Extract weights from your trained PPO model
- Extract normalization statistics from VecNormalize
- Create deployment-ready files in `model_files/` directory

### 3. Upload to QuantConnect

1. Log into your QuantConnect account
2. Create a new project
3. Copy the contents of `main.py`, `model_wrapper.py`, and `option_calculator.py` into your project
4. Go to the Object Store tab and upload these files:
   - `model_files/policy_weights.pth`
   - `model_files/normalization_stats.pkl`
   - `model_files/architecture_info.pkl`

### 4. Update Model Loading

In `model_wrapper.py`, replace the simulated loading code with actual ObjectStore loading (see instructions in `model_files/DEPLOYMENT_INSTRUCTIONS.md`).

### 5. Configure and Test

1. Adjust parameters in `main.py` (dates, cash, position sizes, etc.)
2. Start with a small date range for testing
3. Monitor logs for any issues
4. Scale up once everything is working

## Strategy Overview

The algorithm implements a reinforcement learning based dynamic hedging strategy:

1. **Underlying Asset**: SPY (S&P 500 ETF)
2. **Hedging Instruments**: SPY options (calls and puts)
3. **Rebalancing**: Every hour during market hours
4. **Model**: Recurrent PPO trained on simulated rough Bergomi paths

### Key Features

- **RL-based decisions**: Uses your trained PPO model to make hedging decisions
- **Greek-aware**: Calculates and uses option Greeks in decision making
- **Transaction costs**: Includes realistic transaction costs
- **Position limits**: Enforces maximum position sizes for risk management
- **Real-time adaptation**: Adapts to changing market conditions

### Observation Space (13 features)

1. Normalized stock price
2. 1-period price return
3. 5-period price return  
4. Volatility change
5. Normalized call position
6. Normalized put position
7. Call option moneyness
8. Put option moneyness
9. Current implied volatility
10. Call delta
11. Put delta
12. Normalized portfolio delta
13. Normalized portfolio gamma

### Action Space (2 actions)

1. Call option trade signal [-1, 1]
2. Put option trade signal [-1, 1]

Actions are scaled to actual trade sizes based on `max_trade_per_step`.

## Configuration Parameters

Key parameters you can adjust in `main.py`:

```python
# Backtest period
self.SetStartDate(2023, 1, 1)
self.SetEndDate(2024, 1, 1)

# Portfolio size
self.SetCash(1000000)
self.shares_to_hedge = 10000

# Risk limits
self.max_contracts_per_type = 200
self.max_trade_per_step = 15

# Costs
self.transaction_cost_per_contract = 0.05

# Rebalancing frequency
self.TimeRules.Every(TimeSpan.FromMinutes(60))
```

## Monitoring and Debugging

### Key Logs to Watch

- Model loading success/failure
- Option trade executions
- Position tracking updates
- Greek calculations
- Prediction errors

### Performance Metrics

The algorithm tracks:
- Total return
- Sharpe ratio
- Maximum drawdown
- Option trading costs
- Position P&L
- Greek exposure

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check that files are uploaded to ObjectStore
   - Verify file names match exactly
   - Check error logs for specific issues

2. **No option trades**
   - Verify option universe is being populated
   - Check that ATM options are available
   - Monitor option filter settings

3. **Prediction errors**
   - Check observation vector construction
   - Verify normalization statistics
   - Check for NaN/infinite values

4. **Position tracking issues**
   - Monitor order fill events
   - Check position update logic
   - Verify contract multipliers

### Support

If you encounter issues:
1. Check the QuantConnect logs first
2. Test components individually using `test_local.py`
3. Verify your model files are correctly prepared
4. Check that all required libraries are available

## Advanced Configuration

### Custom Option Filter
Modify `OptionFilter()` to change which options are traded:
```python
def OptionFilter(self, universe):
    return (universe
            .Strikes(-5, 5)  # Wider strike range
            .Expiration(timedelta(15), timedelta(60))  # Different expiry range
            .OnlyApplyFilterAtMarketOpen())
```

### Alternative Rebalancing
Change rebalancing frequency or triggers:
```python
# Rebalance on specific market events
self.Schedule.On(
    self.DateRules.EveryDay("SPY"),
    self.TimeRules.AfterMarketOpen("SPY", 30),  # 30 min after open
    self.Rebalance
)
```

### Risk Management
Add additional risk controls:
```python
def ExecuteOptionTrades(self, actions):
    # Add portfolio heat check
    if self.Portfolio.TotalUnrealizedProfit < -50000:  # Stop loss
        return
    
    # Add volatility regime filter
    if self.last_vol > 0.4:  # High vol regime
        actions *= 0.5  # Reduce position sizes
    
    # Existing trade logic...
```

Good luck with your QuantConnect deployment! 🚀
