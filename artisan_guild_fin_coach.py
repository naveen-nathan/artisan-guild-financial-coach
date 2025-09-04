#!/usr/bin/env python3
# Artisan Guild Fin Coach - CLI (ML Regression + Orchestrator, Date-aware)
# - Uses scikit-learn PolynomialFeatures + LinearRegression with CV degree selection (1..3)
# - Restores Ollama-based LLM orchestrator with conversation memory
# - Handles absolute dates (e.g., "August 10th") by asking the LLM to return target_date_iso
#   and we convert that to horizon_days using the last date in the dataset.
#
# Run:
#   python artisan_guild_fin_coach.py --file /path/to/transactions.csv
#
# Env (optional):
#   OLLAMA_URL   (default http://localhost:11434)
#   OLLAMA_MODEL (default llama3.1)

import os, sys, csv, json, argparse, datetime as dt, re, math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
TIMEOUT = 60

SYSTEM_PROMPT = """You are a Finance Chat Orchestrator for a Smart Financial Coach.
Your job: decide if the user's question is best answered by an ML regression forecast on account running balance,
or by a general conversational answer. Output STRICT JSON only with the schema below.

Schema:
{
  "task": "regression" | "general" | "clarify",
  "regression": {
    "target": "running_balance" | null,
    "horizon_days": number | null,
    "compare_to_amount": number | null,
    "target_date_iso": string | null,
    "notes": string | null
  },
  "need_more_info": boolean,
  "follow_up_question": string | null
}

Guidance:
- Choose "regression" for questions like: "What will my balance be in N days/weeks/months?" or
  "Will I have $X by <date or duration>?" or "How much money will I have on <date>?"
- If the question mentions an **absolute date** (e.g., "August 10th"), set regression.target_date_iso to "YYYY-MM-DD".
  If the year is not mentioned, assume the same year as the dataset's end_date (provided in prior context).
- If the question mentions a **duration** ("in two months", "in 30 days"), set regression.horizon_days (approximate months=30, weeks=7).
- If the question provides a threshold (e.g., "$5000 by next month?"), set compare_to_amount accordingly.
- If required information is missing (e.g., date or duration not provided), set need_more_info=true and ask a targeted follow_up_question.
- Otherwise, set task="general".
- IMPORTANT: Output ONLY the JSON object, no prose.
"""

@dataclass
class Transaction:
    date: dt.date
    description: str
    amount: float
    running_balance: Optional[float]

def parse_money(s: str) -> Optional[float]:
    if not s: return None
    t = s.strip().replace(",", "").replace("$", "").replace("−", "-")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True; t = t[1:-1]
    try:
        val = float(t)
        return -val if neg else val
    except: return None

def parse_date(s: str) -> dt.date:
    for f in ["%m/%d/%Y","%m/%d/%y","%Y-%m-%d"]:
        try: return dt.datetime.strptime(s.strip(), f).date()
        except: continue
    raise ValueError(f"Unrecognized date format: {s}")

def load_transactions(path: str) -> List[Transaction]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        def get(row, name):
            # try exact and case-insensitive header
            if name in row: return row[name]
            for i,h in enumerate(headers):
                if h == name.lower():
                    orig = reader.fieldnames[i]
                    return row.get(orig)
            return None
        for row in reader:
            date_s = get(row, "Date") or get(row, "date")
            desc = get(row, "Description") or get(row, "description") or ""
            amt_s = get(row, "Amount") or get(row, "amount")
            rb_s  = (get(row, "Running Bal.") or get(row, "Running Bal") or
                     get(row, "running bal.") or get(row, "running bal") or
                     get(row, "running_balance"))
            if not date_s or not amt_s: continue
            try: d = parse_date(date_s)
            except: continue
            amt = parse_money(amt_s)
            rb  = parse_money(rb_s) if rb_s else None
            rows.append(Transaction(d, desc, amt if amt is not None else 0.0, rb))
    rows.sort(key=lambda r: r.date)
    return rows

def build_balance_series(txns: List[Transaction]):
    if not txns: return []
    series = [{"date": t.date, "running_balance": t.running_balance}
              for t in txns if t.running_balance is not None]
    if len(series) >= max(2, len(txns)//3):
        # Dedup by date (last wins)
        ded = {}
        for p in series: ded[p["date"]] = p["running_balance"]
        return [{"date": d, "running_balance": rb} for d,rb in sorted(ded.items())]
    # Reconstruct cumulative from first known running balance, else from 0
    base_idx = next((i for i,t in enumerate(txns) if t.running_balance is not None), None)
    if base_idx is not None:
        out = []
        bal = txns[base_idx].running_balance or 0.0
        out.append({"date": txns[base_idx].date, "running_balance": bal})
        for j in range(base_idx+1, len(txns)):
            bal += (txns[j].amount or 0.0)
            out.append({"date": txns[j].date, "running_balance": bal})
        bal = txns[base_idx].running_balance or 0.0
        for j in range(base_idx-1, -1, -1):
            bal -= (txns[j+1].amount or 0.0)
            out.insert(0, {"date": txns[j].date, "running_balance": bal})
        # Dedup
        ded = {}
        for p in out: ded[p["date"]] = p["running_balance"]
        return [{"date": d, "running_balance": rb} for d,rb in sorted(ded.items())]
    # Fallback cumulative from 0 (unknown initial)
    bal=0.0; out=[]
    for t in txns:
        bal += (t.amount or 0.0)
        out.append({"date": t.date, "running_balance": bal})
    return out

def summarize_series(series):
    if not series:
        return {"points":0,"start_date":None,"end_date":None,"start_balance":None,"end_balance":None,"avg_daily_change":None}
    start,end = series[0], series[-1]
    days = max(1, (end["date"]-start["date"]).days)
    delta = end["running_balance"]-start["running_balance"]
    avg = delta/days
    return {
        "points": len(series),
        "start_date": start["date"].isoformat(),
        "end_date": end["date"].isoformat(),
        "start_balance": start["running_balance"],
        "end_balance": end["running_balance"],
        "avg_daily_change": avg
    }

def ml_forecast(series, horizon_days:int):
    if len(series)<3: return None
    base = series[0]["date"]
    X = np.array([(p["date"]-base).days for p in series]).reshape(-1,1)
    y = np.array([p["running_balance"] for p in series])
    best_deg,best_score=1,-1e18
    for deg in [1,2,3]:
        model=Pipeline([("poly",PolynomialFeatures(degree=deg,include_bias=False)),
                        ("lr",LinearRegression())])
        n_splits=min(5,len(X))
        if n_splits<2: continue
        kf=KFold(n_splits=n_splits,shuffle=True,random_state=42)
        score=cross_val_score(model,X,y,cv=kf,scoring="neg_mean_squared_error").mean()
        if score>best_score: best_score, best_deg=score,deg
    final=Pipeline([("poly",PolynomialFeatures(degree=best_deg,include_bias=False)),
                    ("lr",LinearRegression())])
    final.fit(X,y)
    last_x=X[-1,0]; fut=last_x+max(0,int(horizon_days))
    yhat=float(final.predict(np.array([[fut]]))[0])
    cv_rmse=math.sqrt(max(0.0,-best_score)) if best_score!=-1e18 else None
    return {"forecast":yhat,"degree":best_deg,"forecast_date":(base+dt.timedelta(days=int(fut))).isoformat(),"cv_rmse":cv_rmse}

def _ollama_health(url: str) -> dict:
    try:
        r = requests.get(f"{url}/api/version", timeout=10)
        if r.status_code == 200: return r.json()
    except Exception: pass
    return {}

def ollama_chat(url: str, model: str, messages: list) -> str:
    payload_chat = {"model": model, "messages": messages, "stream": False}
    try:
        resp = requests.post(f"{url}/api/chat", json=payload_chat, timeout=TIMEOUT)
        if resp.status_code == 404:
            # fallback to /api/generate (older Ollama)
            parts=[]
            for m in messages:
                parts.append(f"{m.get('role','user').upper()}: {m.get('content','')}")
            prompt="\n\n".join(parts)+"\n\nASSISTANT:"
            payload_gen={"model":model,"prompt":prompt,"stream":False}
            r2=requests.post(f"{url}/api/generate", json=payload_gen, timeout=TIMEOUT)
            r2.raise_for_status()
            d2=r2.json()
            return d2.get("response","") or str(d2)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        return json.dumps(data)
    except requests.exceptions.ConnectionError as ce:
        raise SystemExit(f"Could not reach Ollama at {url}. Is it running? (ollama serve)\nError: {ce}")
    except requests.HTTPError as he:
        info = _ollama_health(url)
        raise SystemExit(f"Ollama HTTP error: {he}\nServer info: {info if info else 'unavailable'}")

def try_extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        frag = m.group(0)
        try:
            return json.loads(frag)
        except:
            return None
    return None

def classify_orchestrate(messages: List[Dict[str, str]], ollama_url: str, model: str) -> dict:
    raw = ollama_chat(ollama_url, model, [{"role":"system","content":SYSTEM_PROMPT}] + messages)
    obj = try_extract_json(raw) or {}
    return obj

def main():
    ap = argparse.ArgumentParser(description="Smart Financial Coach — ML Regression + Orchestrator")
    ap.add_argument("--file", required=True, help="Path to CSV with headers: Date, Description, Amount, Running Bal.")
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name (default: llama3.1)")
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL (default: http://localhost:11434)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}"); sys.exit(1)
    txns = load_transactions(args.file)
    if not txns:
        print("No transactions parsed. Please check your CSV."); sys.exit(1)

    series = build_balance_series(txns)
    summary = summarize_series(series)

    print("\nSmart Financial Coach (CLI) — ML Regression + Orchestrator")
    print("Type 'exit' or 'quit' to end.\n")
    print(f"Loaded {len(txns)} transactions from {args.file}.")
    print(f"Data coverage: {summary['start_date']} → {summary['end_date']} | points={summary['points']}")
    if summary['start_balance'] is not None and summary['end_balance'] is not None:
        print(f"Start balance: ${summary['start_balance']:.2f} | End balance: ${summary['end_balance']:.2f}")
        if summary['avg_daily_change'] is not None:
            print(f"Avg daily change: ${summary['avg_daily_change']:.2f}/day")
    else:
        print("Running balances missing or partial; forecasts may be less accurate.")

    # Conversation memory
    convo: List[Dict[str,str]] = []
    data_context = (
        f"DATA SUMMARY: start={summary['start_date']}, end={summary['end_date']}, "
        f"points={summary['points']}, start_balance={summary['start_balance']}, "
        f"end_balance={summary['end_balance']}, avg_daily_change={summary['avg_daily_change']}."
    )
    last_series_date = series[-1]["date"] if series else None

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except EOFError:
            print(); break
        if user_q.lower() in ("exit","quit"):
            print("Goodbye!"); break

        # Add user message with data summary for context
        convo.append({"role":"user","content": user_q + "\n\n" + data_context})

        # First pass classification
        classification = classify_orchestrate(convo, args.url, args.model)

        # Clarify if needed
        if classification.get("need_more_info") and classification.get("follow_up_question"):
            print(f"\nAssistant: {classification['follow_up_question']}")
            ans = input("You: ").strip()
            convo.append({"role":"assistant","content": classification['follow_up_question']})
            convo.append({"role":"user","content": ans + "\n\n" + data_context})
            classification = classify_orchestrate(convo, args.url, args.model)

        task = classification.get("task")
        if task == "regression":
            reg = classification.get("regression") or {}
            horizon_days = reg.get("horizon_days")
            target_iso = reg.get("target_date_iso")
            compare_to = reg.get("compare_to_amount")

            # If target_date_iso present, convert to horizon_days based on last_series_date
            if target_iso and last_series_date:
                try:
                    target_dt = dt.date.fromisoformat(target_iso)
                    horizon_days = (target_dt - last_series_date).days
                except Exception:
                    pass

            if not isinstance(horizon_days, (int, float)):
                print("\nAssistant: I need a numeric time horizon to forecast. Try 'Will I have $5000 by 30 days?'")
                continue
            horizon_days = int(round(horizon_days))

            result = ml_forecast(series, horizon_days)
            if not result:
                print("\nAssistant: I couldn't build a reliable ML forecast (need at least 3 points with valid running balances).")
                continue

            yhat = result["forecast"]
            fdate = result["forecast_date"]
            deg   = result["degree"]
            rmse  = result["cv_rmse"]
            if isinstance(compare_to, (int, float)):
                verdict = "Yes" if yhat >= float(compare_to) else "No"
                gap = yhat - float(compare_to)
                print(f"\nAssistant: {verdict}. My ML regression forecast for {fdate} is ${yhat:,.2f}.")
                print(f"(Δ vs ${float(compare_to):,.2f}: {gap:+,.2f}; degree={deg}; CV_RMSE={rmse:.2f} if available)")
            else:
                extra = f"; CV_RMSE={rmse:.2f}" if rmse is not None else ""
                print(f"\nAssistant: My ML regression forecast for {fdate} is ${yhat:,.2f} (degree={deg}{extra}).")

            convo.append({"role":"assistant","content": f"Forecast: {fdate} → ${yhat:,.2f} (deg {deg}, RMSE {rmse})."})
            continue

        elif task == "general":
            # Ask LLM to answer concisely using data summary context; avoid inventing numbers
            prompt = {"role":"user","content": "Answer the last question concisely and practically. Use the data summary if relevant, but don't invent numbers beyond it."}
            try:
                text = ollama_chat(args.url, args.model, convo + [prompt])
                print(f"\nAssistant: {text.strip()}")
                convo.append({"role":"assistant","content": text})
            except Exception as e:
                print(f"\nAssistant: (Error reaching the local model: {e})")
            continue

        else:
            # Fallback: just let the LLM reply
            try:
                text = ollama_chat(args.url, args.model, convo)
                print(f"\nAssistant: {text.strip()}")
                convo.append({"role":"assistant","content": text})
            except Exception as e:
                print(f"\nAssistant: (Error reaching the local model: {e})")

if __name__ == "__main__":
    main()
