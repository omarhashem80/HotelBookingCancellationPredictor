# CI Pipeline

Runs automatically on every push to `main` or `master`.

---

## What it does (in order)

| Step | What happens |
|------|-------------|
| **Test** | Runs all tests in `tests/` |
| **Train** | Trains the XGBoost model |
| **Evaluate** | Evaluates the trained model and saves `reports/evaluation_report.json` |
| **Quality check** | Blocks deployment if accuracy < 98% |
| **Export** | Packages the model into the `Hotel-Booking/` folder |
| **Deploy** | Pushes `Hotel-Booking/` to the HuggingFace Space |

---

## Requirements

One secret must be set in GitHub → Settings → Secrets:

| Secret | What it is |
|--------|-----------|
| `HF_TOKEN` | HuggingFace write token for `aelsayed1/Hotel-Booking` |

---

## Deployed app

https://huggingface.co/spaces/aelsayed1/Hotel-Booking

---

## Skipping deployment locally

To run just the test + train steps locally:

```bash
make test
make train
```
