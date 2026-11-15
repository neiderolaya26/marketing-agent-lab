# marketing-agent-lab · English summary

> v1.1.1 · An end-to-end growth marketing lab built only on public data, from lead value to budget allocation, with an AI "growth analyst" agent under human control. The full documentation, notebooks and the 20 design decisions are in Spanish: [README.md](README.md) · [docs/decisiones.md](docs/decisiones.md) · [consulting case](docs/caso_consultoria.md).

**Try it:** [live demo](https://marketing-agent-lab.streamlit.app/) (Streamlit, no LLM) · [project page](https://neiderolaya26.github.io/work/marketing-agent-lab/) · [models and model card on Hugging Face](https://huggingface.co/neiderolaya26/marketing-agent-lab) · [data on Hugging Face](https://huggingface.co/datasets/neiderolaya26/marketing-agent-lab-data)

**In 30 seconds:**

- **Incremental value, not conversions:** every piece answers "what would not have happened without this action, net of its cost": lead value at 90 days, randomized experiments, uplift, difference-in-differences for non-random campaigns, attribution and marketing mix modeling.
- **Ground truth where it matters:** attribution and MMM are tested against synthetic data with known effects. Every last-touch rule and Markov chain misattribute about 2 points per channel in the same direction; Shapley on a conversion model gets 0.30. Without a demand control, the MMM's budget plan loses money in 4 of 10 simulations; with it, the plan captures a median 83% of the optimum.
- **AI agent with human control:** a LangGraph graph (router → acquisition, retention and budget specialists → coordinator) answers business questions with the lab's numbers. Budget rules live in code (±30% per channel, total cannot rise, paid search frozen until a geo test) and every budget move pauses for a human signature. On 20 golden cases it had 0 safety failures (it ignores injected instructions, refuses to delete data and does not raise paid search even when told "management approved").

## Key results

| Piece | Result |
|---|---|
| Lead value | At 90 days a paid search lead is worth 9.47 BRL of revenue and an organic one 7.78; social 1.93 and display 0.35 (assumed 15% commission) |
| Lead scoring | 6 models × 40 Optuna trials with time validation: logistic regression (AUC 0.68) ties LightGBM; the top 30% of leads by score brings 55% of closed deals |
| Customer value | "Next quarter = last quarter" is hard to beat (LightGBM MAE 177 vs 181 USD); BG/NBD is worse because customers buy less rather than leave. The 20% riskiest households hold 58% of spend at risk |
| Experiments | The men's e-mail lifts spend by +0.77 USD per customer (CI [0.49; 1.05]); peeking 20 times without correction raises false positives from 5% to 23%, calibrated O'Brien-Fleming keeps 4.7% |
| Uplift | No measurable heterogeneity in spend: sending the men's e-mail to everyone (1.42 USD per customer) beats personalizing (1.23) |
| Campaigns without an experiment | The naive comparison says +57% spend; doubly robust DiD gives +2.4 USD per household-week [−1.1; +6.0], not distinguishable from 0. Detecting a break-even effect needs 4,300–6,000 randomized control households per campaign |
| Attribution | Last touch makes display look 32% cheaper per conversion than it is |
| MMM | Priors go on ROI, not on coefficients; endogenous spend inflates paid search (9.3 vs 4.8 true ROI); a demand index fixes most of it |
| Agent | 7 questions for ~USD 0.33. Optimizing sales with a fixed total was a mistake; optimizing profit kept 471k in reserve and raised profit from +131k to +324k (true simulator) |
| Agent evals | 20 golden cases, Opus 5.5 judge calibrated against my own ratings: 0 safety failures, 100% expected figures, 85–95% judge approval |

## Recommendation (consulting case)

Move a little, bank what is safe and measure before scaling: reallocate media by profit (TV +10%, e-mail +30%, social −30%, display −19%, paid search unchanged; +274k profit per the MMM), send the men's e-mail to all customers, work leads by score, prioritize retention by household, and run three tests (geo on paid search, randomized control in retention, regional cut in social and display) before the larger cut the MMM suggests.

## Data and licenses

Olist + Marketing Funnel (CC BY-NC-SA 4.0), The Complete Journey by 84.51° (aggregates only), Hillstrom (MineThatData), Lenta (curves and metrics only), a synthetic attribution benchmark (CC BY 4.0) and an in-house MMM generator. Everything is stored as Parquet (zstd) and queried with in-memory DuckDB through a SELECT-only gate.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                  # no downloads needed: uses data/processed/
python -m growth_lab.servidor_mcp       # read-only MCP tools, no API key
python -m growth_lab.agente "Is display overrated?"   # needs ANTHROPIC_API_KEY in .env; --trazar sends LangSmith traces
streamlit run demo/streamlit_app.py     # demo, no LLM (pip install -r demo/requirements.txt)
```

## Limits

- Attribution and MMM use synthetic data: they show the method, not the real value of these channels.
- Olist's commission (15%) is an assumption.
- 20 eval cases catch large errors, not fine differences between agent versions; the last fix (v5) was not re-measured with paid evals.
