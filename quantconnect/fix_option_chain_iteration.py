# Quick fix for QuantConnect option chain iteration issue
# Copy this _find_atm_options method to replace the problematic one

def _find_atm_options(self, option_chain, spy_price):
    """Find at-the-money call and put options"""
    calls = []
    puts = []
    
    # FIXED: Iterate directly over option_chain, not kvp.Value
    for option in option_chain:
        # Filter for reasonable DTE (7-45 days)
        dte = (option.Expiry.date() - self.Time.date()).days
        if dte < 7 or dte > 45:
            continue
        
        # Check if we have price data
        if option.LastPrice <= 0 and option.BidPrice <= 0:
            continue
        
        # Separate calls and puts
        if option.Right == OptionRight.Call:
            calls.append(option)
        elif option.Right == OptionRight.Put:
            puts.append(option)
    
    # Sort by proximity to ATM
    calls.sort(key=lambda x: abs(x.Strike - spy_price))
    puts.sort(key=lambda x: abs(x.Strike - spy_price))
    
    return calls[:3], puts[:3]  # Return top 3 closest to ATM
