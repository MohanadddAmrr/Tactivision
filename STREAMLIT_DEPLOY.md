# Deploying the TactiVision Pro Showcase to Streamlit Cloud

This document explains how to deploy the **public showcase** (`streamlit_app.py`
at the repo root) to [Streamlit Community Cloud](https://share.streamlit.io).

> **Important:** This is *not* the full TactiVision pipeline. The full dashboard
> at `demo/dashboard_final.py` requires GPU + heavy ML dependencies and is meant
> to run locally. The showcase here is a lightweight, read-only visualisation
> of pre-computed match analysis results, so recruiters and reviewers can
> explore the project without a local setup.

## What's included

- **`streamlit_app.py`** — the showcase entry point (loaded by Streamlit Cloud by convention)
- **`requirements.txt`** — slim deps for the showcase (streamlit, pandas, plotly, pillow)
- **`requirements_full.txt`** — heavy deps for running the FULL pipeline locally (torch, tensorflow, ultralytics, opencv-contrib, easyocr…)
- **`.streamlit/config.toml`** — dark theme matching the project's visual language
- Pre-computed match data at `demo/demo_outputs/arsenal/` is served as-is

### Why two requirements files?

Streamlit Community Cloud auto-detects `requirements.txt` at repo root and installs it on every deploy. If we put the heavy ML dependencies there, every deploy would fail (torch alone is 2+ GB — exceeds the 1 GB memory cap). So the slim showcase deps are in `requirements.txt`, and the full-pipeline deps live in `requirements_full.txt`.

To install the full pipeline locally: `pip install -r requirements_full.txt`.

## Deploy in 3 clicks

### 1. Go to Streamlit Cloud

Open https://share.streamlit.io and click **"Create app"** → **"Deploy a public app from GitHub"**.

### 2. Configure the deployment

Fill in:

| Field | Value |
|---|---|
| **Repository** | `MohanadddAmrr/TactiVision` |
| **Branch** | `master` ⚠️ (NOT `main` — the empty-initial-commit `main` has no code) |
| **Main file path** | `streamlit_app.py` |
| **App URL** (custom subdomain) | `tactivision` (or any available slug) |

Click **"Advanced settings"** and set:

| Field | Value |
|---|---|
| **Python version** | `3.11` ⚠️ **(NOT 3.14 — it's too new, half the packages don't have wheels for it yet)** |

Leave Secrets blank.

### 3. Click Deploy and wait

First build takes 3-7 minutes while Streamlit Cloud installs dependencies and
spins up the container. You'll land on the live app URL when it's ready.

Your final URL will look like:  `https://tactivision.streamlit.app`

## Updating the showcase

Every push to `master` (or whichever branch you deployed) automatically triggers
a redeploy. You never need to touch the Streamlit Cloud dashboard again.

## Resource budget

- **Memory:** Showcase uses ~300 MB (well under the 1 GB free-tier cap)
- **Cold start:** ~15-25 seconds
- **Payload:** ~7 MB of pre-computed heatmap PNGs served statically

## Troubleshooting

### "ModuleNotFoundError" after deploy

Streamlit Cloud's auto-detected `requirements.txt` should be the slim showcase file (not the full-pipeline one). Verify:
- `requirements.txt` at the repo root is ~5 lines (streamlit, pandas, numpy, plotly, Pillow)
- `requirements_full.txt` is the long heavy file

If they're swapped, rename them.

### App shows "No pre-computed matches found"

The `demo/demo_outputs/arsenal/` folder is missing from the deployed commit.
Check that `git ls-files demo/demo_outputs/arsenal/metrics.json` returns a path
locally. If not, add it with `git add -f demo/demo_outputs/arsenal/*`.

### Deploy times out at `pip install`

- Python version is set too new (3.14 won't compile numpy/pandas yet). Set to **3.11**.
- Or a dependency pin is wrong — check logs for the exact failing package.

### Changes don't appear after push

Streamlit Cloud sometimes caches. Force a reboot:
- Dashboard → the app's three-dot menu → "Reboot app"

## Files reference

```
tactivision/
├── streamlit_app.py           # Showcase entry point (what Streamlit Cloud runs)
├── requirements.txt           # Slim deps for Streamlit Cloud showcase
├── requirements_full.txt      # Full-pipeline deps (local use only, has torch etc)
├── .streamlit/
│   └── config.toml            # Theme + runtime config
├── demo/
│   ├── demo_outputs/
│   │   └── arsenal/           # Pre-computed match data (served by showcase)
│   │       ├── metrics.json
│   │       ├── heatmap_team_A.png
│   │       ├── heatmap_team_B.png
│   │       ├── heatmap_global.png
│   │       └── heatmap_player_*.png (hundreds)
│   └── dashboard_final.py     # Full-pipeline dashboard (local only, GPU recommended)
└── services/                  # Full-pipeline services (local only)
```

## Linking the live URL back to your profile

Once the app is live, update these places with the Streamlit URL:

- **CV:** replace the `[coming soon]` placeholder under the TactiVision project
  with the live link
- **LinkedIn Featured section:** add the live URL as a Featured link
- **GitHub TactiVision repo README:** replace the "coming soon" demo link in the hero
- **GitHub profile README** (`MohanadddAmrr/MohanadddAmrr`): add it next to the
  TactiVision line

---

Built as part of the TactiVision Pro graduation project at Misr International
University, 2026. Showcase authored by Mohanad Amr.
