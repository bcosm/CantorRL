# 🎯 QuantConnect RL Hedging Algorithm - Strategic Debugging Guide

## 🔴 Critical Breakpoint Locations in main.py

I've added 18 strategic breakpoint markers throughout your algorithm. Here's how to use them:

### 🚀 **How to Set Breakpoints in VS Code:**
1. Open `quantconnect/main.py` 
2. Find the breakpoint comments (🔴 BREAKPOINT #X)
3. Click in the **left margin** next to the line numbers
4. Red dots will appear = breakpoints are set
5. Press **F5** to start debugging

---

## 🎯 **Breakpoint Strategy by Debugging Phase:**

### **Phase 1: Algorithm Setup & Initialization**
- **BREAKPOINT #1** (Line ~16): Algorithm startup - verify configuration
- **BREAKPOINT #2** (Line ~29): Model loading - catch model initialization issues

### **Phase 2: Data Processing & Market State**  
- **BREAKPOINT #3** (Line ~74): Data reception entry point
- **BREAKPOINT #4** (Line ~79): Warmup completion transition
- **BREAKPOINT #5** (Line ~84): SPY price data validation
- **BREAKPOINT #6** (Line ~93): Option chain processing

### **Phase 3: Core Trading Logic**
- **BREAKPOINT #7** (Line ~190): Main rebalancing entry point
- **BREAKPOINT #8** (Line ~197): Daily LSTM reset point

### **Phase 4: RL Model Pipeline**
- **BREAKPOINT #9** (Line ~258): Observation vector inspection
- **BREAKPOINT #10** (Line ~262): Model prediction critical point
- **BREAKPOINT #11** (Line ~269): Action analysis and validation
- **BREAKPOINT #12** (Line ~281): Trade execution initiation

### **Phase 5: Deep Feature Engineering**
- **BREAKPOINT #13** (Line ~293): Observation construction start
- **BREAKPOINT #14** (Line ~310): Market data feature engineering

### **Phase 6: Trade Execution Details**
- **BREAKPOINT #15** (Line ~420): Option trade execution start
- **BREAKPOINT #16** (Line ~428): Trade quantity validation  
- **BREAKPOINT #17** (Line ~432): Option symbol resolution
- **BREAKPOINT #18** (Line ~441): Call option trade execution

---

## 🔍 **What to Inspect at Each Breakpoint:**

### **Model Issues (Breakpoints #2, #10, #11):**
```python
# Inspect these variables:
self.model_wrapper.loaded          # Should be True
self.model_wrapper.model           # Should not be None
observation.shape                  # Should be (13,)
actions                           # Should be 2-element array
np.isfinite(observation).all()    # Should be True
```

### **Market Data Issues (Breakpoints #5, #6, #14):**
```python
# Inspect these variables:
self.last_price                   # Should be > 0
self.last_vol                     # Should be between 0.05-1.0
len(self.available_options)       # Should be > 0
current_price                     # Should match self.last_price
```

### **Trading Logic Issues (Breakpoints #15, #16, #17, #18):**
```python
# Inspect these variables:
call_trade, put_trade             # Should be reasonable integers
call_symbol, put_symbol           # Should not be None
self.current_call_contracts       # Track position sizes
self.current_put_contracts        # Track position sizes
```

### **Observation Issues (Breakpoints #9, #13, #14):**
```python
# Inspect observation components:
obs[0]  # norm_S_t (normalized stock price)
obs[1]  # norm_C_t (normalized call price)  
obs[2]  # norm_P_t (normalized put price)
obs[3]  # norm_call_held (normalized call position)
obs[4]  # norm_put_held (normalized put position)
obs[5]  # v_t (volatility)
obs[6]  # norm_time_to_end (episode progress)
obs[7:11] # Greeks (delta, gamma)
obs[11:13] # Lagged returns
```

---

## 🎮 **Debugging Workflow:**

### **Step 1: Basic Setup Verification**
1. Set breakpoints #1, #2
2. Run with F5
3. Verify model loads successfully

### **Step 2: Data Flow Verification** 
1. Set breakpoints #3, #4, #5, #6
2. Check market data is flowing correctly
3. Verify option chains are populated

### **Step 3: RL Pipeline Deep Dive**
1. Set breakpoints #9, #10, #11, #12
2. Step through observation → prediction → action → execution
3. Validate each transformation step

### **Step 4: Trade Execution Analysis**
1. Set breakpoints #15, #16, #17, #18  
2. Verify trade calculations and order placement
3. Check position tracking accuracy

---

## 🚨 **Common Issues to Debug:**

1. **Model Not Loading**: Check breakpoint #2
2. **Zero Actions**: Check breakpoints #10, #11
3. **NaN Observations**: Check breakpoints #13, #14
4. **No Option Data**: Check breakpoints #6, #17
5. **Trade Failures**: Check breakpoints #16, #18

---

## 🎯 **Pro Tips:**

- **Use Step Over (F10)** to move line by line
- **Use Step Into (F11)** to dive into model_wrapper methods
- **Watch variables** by right-clicking → "Add to Watch"
- **Evaluate expressions** in the Debug Console
- **Set conditional breakpoints** for specific scenarios

Your algorithm is now **fully instrumented for debugging**! 🚀
