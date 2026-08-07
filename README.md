# 🌊 SubsurfAI — Physics-Aware Subsurface Interpretation Platform

**Author:** Abdulrahman Al-Fakih · KFUPM / Geoscience AI Lab  
**Live demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://subsurfai.streamlit.app)

SubsurfAI is an open-source, physics-constrained well log interpretation platform for the Dutch North Sea. It combines classical petrophysical equations (Archie's law, Wyllie time-average, NPHI-RHOB crossplot) with AI-powered analysis using Claude.

---

## What it does

- **Upload any LAS well log** → instant petrophysical interpretation
- **6 bundled Dutch North Sea wells** (F-block, NLOG) for instant demo
- **Physics-constrained lithofacies** (Shale / Sandstone / Limestone / Coal)
- **Reservoir quality** — porosity (φ_eff), water saturation (Sw), uncertainty bands
- **AI Chat** — ask questions about your well in plain English, get expert answers
- **Multi-well comparison** — compare up to 4 wells side by side
- **Export** — download interpreted CSV or LAS with new curves

---

## Foundation model

GlobalWellFM — trained on 163,000 depth samples from 3 countries:  
🤗 [Aljaser2030/GlobalWellFM](https://huggingface.co/Aljaser2030/GlobalWellFM) · F1 = 0.9494

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/SubsurfAI
cd SubsurfAI
pip install -r requirements.txt
streamlit run app.py
```

Set your Anthropic API key in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Citation

```
Al-Fakih, A. (2026). SubsurfAI: A Physics-Constrained AI Platform for Well Log
Interpretation in the Dutch North Sea. KFUPM / Geoscience AI Lab.
```

---

## License

MIT — free to use, modify, and distribute with attribution.
