# Smart Financial Coach 💰

An AI-enabled financial assistant developed as part of the Palo Alto Networks IT Software Engineer case challenge.  
The Smart Financial Coach helps users gain visibility into spending, detect patterns, and forecast account balances responsibly using a hybrid of deterministic ML and generative AI.

---

## 📌 Features
- **Transaction Categorization (planned)**  
- **Subscription Detector (planned)**  
- **Balance Forecasting** with ML regression  
- **AI Routing**: Orchestrator decides whether to use regression or conversational AI  
- **Responsible AI**: Local-first, toggleable AI, transparent forecasts  

---

## 🛠️ Tech Stack
- **Language:** Python 3.9+  
- **Libraries:** `requests`, `numpy`, `scikit-learn`, `matplotlib`  
- **AI Runtime:** [Ollama](https://ollama.ai/) (local LLM runtime)  
- **Interface:** Command-line interface (CLI)  
- **Data Input:** CSV transactions (`Date, Description, Amount, Running Bal.`)  

---

## 🚀 Setup

1. **Clone the repo**
```bash
git clone <your-repo-url>
cd FinCoach
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Ollama and pull a model**
```bash
ollama serve
ollama pull llama3.1:8b
```

---

## ▶️ Usage
Run the app with your transactions CSV:

```bash
python smart_fin_coach.py --file sample_transactions.csv
```

Example interaction:
```
You: How much money will I have in 30 days?
Assistant: My ML regression forecast for 2025-09-02 is $4,355.67 (degree=2; CV_RMSE=312.40).
```

---

## 📊 Demo Video
🎥 [Demo Video Link](https://example.com)  
*(Replace with actual unlisted YouTube or Vimeo link before submission)*

---

## 📑 Documentation
For detailed design decisions, see [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md).  
A PDF version is also available: `DESIGN_DOCUMENTATION.pdf`.

---

## 🔮 Future Enhancements
- Web or mobile interface  
- Real-time bank API integration  
- Seasonal ML forecasting (ARIMA, LSTM)  
- Subscription detection & budgeting features  
- Federated learning for privacy-preserving personalization  

---

*Prepared by: Naveen Nathan*
