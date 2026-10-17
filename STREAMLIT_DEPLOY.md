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
- **`requirements_streamlit.txt`** — slim dependency set (no torch / tensorflow / ultralytics)
- **`.streamlit/config.toml`** — dark theme matching the project's visual language
- Pre-computed match data at `demo/demo_outputs/arsenal/` is served as-is

## Deploy in 3 clicks

### 1. Go to Streamlit Cloud and start a new app

Open https://share.streamlit.io and click **"New app"**.

### 2. Configure the deployment

Fill in:

| Field | Value |
|---|---|
| **Repository** | `MohanadddAmrr/TactiVision` |
| **Branch** | `main` (merge this PR first) |
| **Main file path** | `streamlit_app.py` |
| **App URL** (custom subdomain) | `tactivision` (or any available slug) |

Click **"Advanced settings"** and set:

| Field | Value |
|---|---|
| **Python version** | `3.11` |
| **Requirements file** | `requirements_streamlit.txt` |

> If "Requirements file" is not visible in the UI, Streamlit Cloud will
> auto-detect `requirements.txt` first. In that case, after deployment,
> go to **Settings → Secrets** and add `requirements_file = "requirements_streamlit.txt"`,
> or temporarily rename the full `requirements.txt` on the deployment branch.

### 3. Click Deploy and wait

First build takes 3-7 minutes while Streamlit Cloud installs dependencies and
spins up the container. You'll land on the live app URL when it's ready.

Your final URL will look like:  `https://tactivision.streamlit.app`

## Updating the showcase

Every push to `main` (or whichever branch you deployed) automatically triggers
a redeploy. You never need to touch the Streamlit Cloud dashboard again.

## Resource budget

- **Memory:** Showcase uses ~300 MB (well under the 1 GB free-tier cap)
- **Cold start:** ~15-25 seconds
- **Payload:** ~7 MB of pre-computed heatmap PNGs served statically

## Troubleshooting

### "ModuleNotFoundError" after deploy

Streamlit Cloud picked up the wrong `requirements.txt`. Check:
- Advanced settings → "Requirements file" is set to `requirements_streamlit.txt`
- OR rename the top-level `requirements.txt` → `requirements_full.txt` so the
  slim file becomes the default

### App shows "No pre-computed matches found"

The `demo/demo_outputs/arsenal/` folder is missing from the deployed commit.
Check that `git ls-files demo/demo_outputs/arsenal/metrics.json` returns a path
locally. If not, add it with `git add -f demo/demo_outputs/arsenal/*`.

### Deploy times out at `pip install`

Something in `requirements_streamlit.txt` is pulling heavy transitive deps.
Pin versions tighter or check `pip-tools` output.

### Changes don't appear after push

Streamlit Cloud sometimes caches. Force a reboot:
- Dashboard → the app's three-dot menu → "Reboot app"

## Files reference

```
tactivision/
├── streamlit_app.py           # Showcase entry point (what Streamlit Cloud runs)
├── requirements_streamlit.txt # Slim deps for the showcase
├── requirements.txt           # Full-pipeline deps (local use only)
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
