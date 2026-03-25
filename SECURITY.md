# Security Notes

## Secrets

Do not commit:
- `.env`
- API keys
- exchange credentials
- private tokens
- webhook secrets

Use local environment variables instead.

## Generated Artifacts

Do not commit:
- raw downloaded market data
- processed datasets
- trained model files
- generated signal snapshots
- local experiment outputs

Those files are for local use and are ignored by default.

## Public Issues And Pull Requests

Do not post:
- real API keys
- private infrastructure details
- internal server URLs
- screenshots containing secrets
- local account identifiers you do not want public

## Before You Push

Check:
1. `git status`
2. `.env` is not staged
3. no `data/raw/*.csv` files are staged unless intentionally publishing a cleaned sample
4. no `outputs/*.json` or `outputs/*.csv` files are staged
5. no `models/*.pkl` files are staged
