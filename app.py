"""
SubsurfAI — Physics-Aware Subsurface Interpretation Platform
Real Dutch North Sea data (1,066 NLOG F-block wells) + physics-constrained AI
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import io, os, sys

sys.path.insert(0, os.path.dirname(__file__))

# Load .env if present (local dev), then Streamlit secrets (cloud), then env var
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from petromind import load_las, generate_demo_las, interpret, zone_summary, answer_question, build_log_figure, build_crossplot
from petromind.demo_wells import DEMO_WELLS, get_las_path, well_map_data
from petromind.claude_chat import ask_claude

# API key: Streamlit Cloud secrets → env var → empty
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SubsurfAI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0A1628; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #38BDF8 !important; font-size:0.9rem; }
.chat-user   { background:#1E40AF; color:#E0F2FE; border-radius:12px 12px 2px 12px;
               padding:10px 14px; margin:6px 0; max-width:85%; float:right; clear:both; }
.chat-ai     { background:#1E293B; color:#E2E8F0; border-radius:12px 12px 12px 2px;
               padding:10px 14px; margin:6px 0; max-width:85%; float:left; clear:both; }
.well-badge  { display:inline-block; background:#0F172A; border:1px solid #38BDF8;
               color:#38BDF8; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; }
.metric-val  { font-size:1.5rem; font-weight:700; color:#38BDF8; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "wells": {},          # {name: {"df_raw":..., "df":..., "meta":...}}
    "active_well": None,
    "chat_history": [],
    "config": {"rw":0.05,"a":1.0,"m":2.0,"n":2.0,"rho_ma":2.65,"rho_fl":1.0,
                "dt_ma":55.5,"dt_fl":189.0,"gr_clean":30,"gr_shale":120},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def add_well(name, df_raw, meta):
    df = interpret(df_raw, st.session_state.config)
    st.session_state.wells[name] = {"df_raw": df_raw, "df": df, "meta": meta}
    st.session_state.active_well = name

def active_df():
    if st.session_state.active_well and st.session_state.active_well in st.session_state.wells:
        return st.session_state.wells[st.session_state.active_well]["df"]
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 SubsurfAI")
    st.markdown("*Physics-aware subsurface platform*")
    st.caption("Real NLOG Dutch North Sea data · 1,066 wells")
    st.divider()

    # ── Demo wells ──
    st.markdown("### 🗄️ NLOG Demo Wells")
    st.caption("Real F-block wells, Netherlands North Sea")
    demo_choice = st.selectbox("Select well", ["— pick a well —"] + list(DEMO_WELLS.keys()), label_visibility="collapsed")
    if st.button("▶ Load Demo Well", use_container_width=True) and demo_choice != "— pick a well —":
        with st.spinner(f"Loading {demo_choice.split('(')[0].strip()}…"):
            try:
                path = get_las_path(demo_choice)
                df_raw, meta = load_las(path)
                add_well(demo_choice.split(" (")[0], df_raw, meta)
                st.success(f"✓ Loaded — {len(st.session_state.wells[demo_choice.split(' (')[0]]['df'])} rows")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("▶ Synthetic Demo (4-zone)", use_container_width=True):
        df_raw, meta = generate_demo_las()
        add_well("Synthetic Well", df_raw, meta)
        st.success("Synthetic 4-zone well loaded")

    st.divider()

    # ── Upload own data ──
    st.markdown("### 📂 Upload Your LAS")
    uploaded = st.file_uploader("LAS file", type=["las","LAS"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Reading…"):
            try:
                df_raw, meta = load_las(uploaded)
                add_well(uploaded.name.replace(".las","").replace(".LAS",""), df_raw, meta)
                st.success(f"✓ {uploaded.name} loaded")
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Loaded wells ──
    if st.session_state.wells:
        st.markdown("### 📋 Loaded Wells")
        for wname in list(st.session_state.wells.keys()):
            col1, col2 = st.columns([3,1])
            is_active = wname == st.session_state.active_well
            label = f"{'▶ ' if is_active else ''}{wname}"
            if col1.button(label, key=f"sw_{wname}", use_container_width=True):
                st.session_state.active_well = wname
                st.rerun()
            if col2.button("✕", key=f"rm_{wname}"):
                del st.session_state.wells[wname]
                if st.session_state.active_well == wname:
                    st.session_state.active_well = next(iter(st.session_state.wells), None)
                st.rerun()
        st.divider()

    # ── Archie config ──
    st.markdown("### ⚙️ Archie / Wyllie")
    st.session_state.config["rw"]  = st.number_input("Rw (Ω·m)", 0.001,10.0, st.session_state.config["rw"], 0.005, format="%.3f")
    st.session_state.config["m"]   = st.number_input("m (cementation)", 1.0, 3.0, st.session_state.config["m"], 0.1)
    st.session_state.config["n"]   = st.number_input("n (saturation)", 1.0, 3.0, st.session_state.config["n"], 0.1)
    if st.button("🔄 Re-interpret all wells", use_container_width=True):
        for wname, wdata in st.session_state.wells.items():
            wdata["df"] = interpret(wdata["df_raw"], st.session_state.config)
        st.success("All wells re-interpreted!")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🌊 SubsurfAI — Subsurface AI Platform")
st.caption("Real NLOG Dutch North Sea data · Physics-constrained interpretation · Uncertainty quantification")

df = active_df()

if df is None:
    # Landing page
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("#### 🗺️ North Sea Map\nExplore 1,066 real NLOG wells. Click to load any well instantly.")
    c2.markdown("#### 📈 Log Viewer\nMulti-track display with GR, RHOB, NPHI, RT, φ, Sw and uncertainty ribbons.")
    c3.markdown("#### 🧠 AI Physics\nArchie's law, Wyllie time-average, NPHI-RHOB crossplot — hard physics, not hallucination.")
    c4.markdown("#### 💬 Knowledge Chat\nAsk about porosity, lithology, fluid contacts in plain English.")
    st.info("👈 Load a demo NLOG well or upload your own LAS file to begin.")
    # Show map even before loading a well
    st.markdown("---")
    st.markdown("### 🗺️ Dutch North Sea — Available Demo Wells")
    _map_data = well_map_data()
    import plotly.graph_objects as go
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=[w["lat"] for w in _map_data],
        lon=[w["lon"] for w in _map_data],
        mode="markers+text",
        text=[w["name"] for w in _map_data],
        textposition="top center",
        marker=dict(size=14, color="#38BDF8", opacity=0.9),
        customdata=[[w["depth_range"], w["valid_rows"], w["description"]] for w in _map_data],
        hovertemplate="<b>%{text}</b><br>Depth: %{customdata[0]}<br>Samples: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=54.9, lon=4.9), zoom=6),
        margin=dict(l=0,r=0,t=0,b=0), height=420,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
depth_min = df["DEPTH"].min(); depth_max = df["DEPTH"].max()
net_pay = df[df["FLUID_FLAG"]=="hydrocarbon"]["DEPTH"].count() * ((depth_max-depth_min)/max(len(df),1))
avg_phi = df["PHI_EFF"].mean() if "PHI_EFF" in df else float("nan")
avg_sw  = df["SW"].mean()      if "SW"      in df else float("nan")
lith_mode = df["LITHOLOGY"].mode().iloc[0] if "LITHOLOGY" in df else "—"

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Well", st.session_state.active_well or "—")
m2.metric("Depth (m)", f"{depth_min:.0f}–{depth_max:.0f}")
m3.metric("Net pay", f"{net_pay:.1f} m")
m4.metric("Avg φ_eff", f"{avg_phi:.3f}")
m5.metric("Avg Sw", f"{avg_sw:.3f}")
m6.metric("Dom. Lithology", lith_mode.capitalize())
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_log, tab_xplot, tab_multi, tab_zone, tab_chat, tab_export = st.tabs([
    "🗺️ Well Map", "📈 Log View", "✖ Crossplot", "⚖ Multi-Well", "🗂 Zone Summary", "💬 AI Chat", "⬇ Export"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — NORTH SEA MAP
# ═══════════════════════════════════════════════════════════
with tab_map:
    import plotly.graph_objects as go
    map_data = well_map_data()

    # Highlight loaded wells
    loaded_names = set(st.session_state.wells.keys())

    fig_map = go.Figure()
    # All demo wells
    fig_map.add_trace(go.Scattermapbox(
        lat=[w["lat"] for w in map_data],
        lon=[w["lon"] for w in map_data],
        mode="markers+text",
        name="Demo wells (NLOG)",
        text=[w["name"] for w in map_data],
        textposition="top center",
        marker=dict(size=16, color=["#22C55E" if w["name"] in loaded_names else "#38BDF8" for w in map_data], opacity=0.9),
        customdata=[[w["depth_range"], w["valid_rows"], w["description"], w["nlog_url"]] for w in map_data],
        hovertemplate="<b>%{text}</b><br>Depth: %{customdata[0]}<br>Samples: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=54.92, lon=4.85), zoom=6.5),
        margin=dict(l=0,r=0,t=30,b=0),
        height=500,
        title="Dutch North Sea F-Block Wells (NLOG) — green = loaded",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Well info cards
    st.markdown("#### Well Details")
    cols = st.columns(3)
    for i, w in enumerate(map_data):
        with cols[i % 3]:
            loaded = w["name"] in loaded_names
            st.markdown(f"""
**{'✅ ' if loaded else ''}{"🟦"} {w['name']}**  
📍 {w['lat']:.3f}°N, {w['lon']:.3f}°E · Block {w['block']}  
📏 {w['depth_range']} · {w['valid_rows']:,} samples  
{w['description']}  
[📂 View on NLOG ↗]({w['nlog_url']})
""")

# ═══════════════════════════════════════════════════════════
# TAB 2 — LOG VIEW
# ═══════════════════════════════════════════════════════════
with tab_log:
    col_ctrl, col_fig = st.columns([1, 5])
    with col_ctrl:
        d_lo = st.number_input("Top (m)",    float(depth_min), float(depth_max), float(depth_min), 10.0)
        d_hi = st.number_input("Bottom (m)", float(depth_min), float(depth_max), float(depth_max), 10.0)
    with col_fig:
        fig = build_log_figure(df, depth_range=(d_lo, d_hi))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — CROSSPLOT
# ═══════════════════════════════════════════════════════════
with tab_xplot:
    avail = [c for c in df.columns if c != "DEPTH"]
    cx, cy = st.columns(2)
    x_col = cx.selectbox("X axis", avail, index=avail.index("NPHI") if "NPHI" in avail else 0)
    y_col = cy.selectbox("Y axis", avail, index=avail.index("RHOB") if "RHOB" in avail else 1)
    if x_col != y_col:
        st.plotly_chart(build_crossplot(df, x_col, y_col), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — MULTI-WELL COMPARISON
# ═══════════════════════════════════════════════════════════
with tab_multi:
    if len(st.session_state.wells) < 2:
        st.info("Load at least 2 wells from the sidebar to compare them side by side.")
    else:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        well_names = list(st.session_state.wells.keys())
        selected = st.multiselect("Select wells to compare (2–4)", well_names, default=well_names[:min(4,len(well_names))])
        curve = st.selectbox("Curve to compare", ["GR","RHOB","NPHI","PHI_EFF","SW","VCL"], index=0)

        if len(selected) >= 2:
            fig_m = make_subplots(rows=1, cols=len(selected), shared_yaxes=True,
                                  subplot_titles=selected, horizontal_spacing=0.02)
            COLORS = ["#38BDF8","#22C55E","#F59E0B","#EC4899"]
            for i, wname in enumerate(selected):
                wdf = st.session_state.wells[wname]["df"]
                if curve in wdf.columns:
                    fig_m.add_trace(go.Scatter(
                        x=wdf[curve], y=wdf["DEPTH"], mode="lines",
                        name=wname, line=dict(color=COLORS[i % 4], width=1.2),
                        showlegend=True,
                    ), row=1, col=i+1)
                    fig_m.update_xaxes(title_text=curve, row=1, col=i+1, side="top")

            fig_m.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
            fig_m.update_layout(height=650, plot_bgcolor="white", paper_bgcolor="white",
                                 margin=dict(l=60,r=20,t=60,b=20))
            st.plotly_chart(fig_m, use_container_width=True)

            # Summary stats table
            rows = []
            for wname in selected:
                wdf = st.session_state.wells[wname]["df"]
                rows.append({
                    "Well": wname,
                    "Depth (m)": f"{wdf['DEPTH'].min():.0f}–{wdf['DEPTH'].max():.0f}",
                    "Avg GR": f"{wdf['GR'].mean():.1f}" if "GR" in wdf else "—",
                    "Avg φ_eff": f"{wdf['PHI_EFF'].mean():.3f}" if "PHI_EFF" in wdf else "—",
                    "Avg Sw": f"{wdf['SW'].mean():.3f}" if "SW" in wdf else "—",
                    "Net pay (m)": f"{wdf[wdf['FLUID_FLAG']=='hydrocarbon']['DEPTH'].count() * ((wdf['DEPTH'].max()-wdf['DEPTH'].min())/max(len(wdf),1)):.1f}",
                    "Dom. Lith.": wdf["LITHOLOGY"].mode().iloc[0].capitalize() if "LITHOLOGY" in wdf else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Well"), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — ZONE SUMMARY
# ═══════════════════════════════════════════════════════════
with tab_zone:
    n_zones = st.number_input("Number of zones", 1, 8, 3, 1)
    cols = st.columns(3)
    zones = []
    for i in range(int(n_zones)):
        with cols[i % 3]:
            st.markdown(f"**Zone {i+1}**")
            zt = st.number_input(f"Top (m)", float(depth_min), float(depth_max),
                                  float(depth_min + i*((depth_max-depth_min)/int(n_zones))), key=f"zt{i}")
            zb = st.number_input(f"Bottom (m)", float(depth_min), float(depth_max),
                                  float(depth_min + (i+1)*((depth_max-depth_min)/int(n_zones))), key=f"zb{i}")
            zones.append((zt, zb))

    if st.button("Compute Zone Summaries"):
        rows = []
        for i, (zt, zb) in enumerate(zones):
            s = zone_summary(df, zt, zb)
            s["Zone"] = f"Zone {i+1}  ({zt:.0f}–{zb:.0f} m)"
            rows.append(s)
        zdf = pd.DataFrame(rows).set_index("Zone")
        zdf[zdf.select_dtypes(float).columns] = zdf.select_dtypes(float).round(4)
        st.dataframe(zdf, use_container_width=True)
        for i, r in enumerate(rows):
            sw = r.get("avg_SW", 1.0) or 1.0
            phi = r.get("avg_PHI_EFF", 0.0) or 0.0
            flag = "🟢 Potential pay" if sw < 0.5 and phi > 0.1 else ("🟡 Transition" if sw < 0.7 else "🔵 Wet/tight")
            st.markdown(f"**Zone {i+1}:** {flag} — φ_eff={phi:.3f}, Sw={sw:.3f}")

# ═══════════════════════════════════════════════════════════
# TAB 6 — CLAUDE AI CHAT
# ═══════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 🤖 SubsurfAI Chat — Powered by Claude")
    st.caption("Claude reads your entire interpreted well log and answers as an expert petrophysicist.")

    if not ANTHROPIC_API_KEY:
        st.warning("No API key found. Add your Anthropic key to the `.env` file in the SubsurfAI folder.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🌊"):
                    st.markdown(msg["content"])

        # Quick questions
        st.markdown("**Quick questions:**")
        qcols = st.columns(3)
        quick_qs = [
            "What is the best reservoir interval in this well?",
            "Are there hydrocarbon indicators? Give depths.",
            "What is the dominant lithology and why?",
            "Estimate net pay thickness with cutoffs φ>0.1 and Sw<0.5",
            "What is the porosity range and what does it mean?",
            "Compare this well to typical Dutch North Sea Rotliegend reservoirs.",
        ]
        for idx, q in enumerate(quick_qs):
            if qcols[idx % 3].button(q, key=f"qq_{idx}", use_container_width=True):
                st.session_state._pending_q = q

        user_q = st.chat_input("Ask anything about this well…")
        pending = getattr(st.session_state, "_pending_q", None)
        if pending:
            user_q = pending
            st.session_state._pending_q = None

        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant", avatar="🌊"):
                with st.spinner("Claude is reading your well data…"):
                    try:
                        answer = ask_claude(
                            question=user_q,
                            df=df,
                            well_name=st.session_state.active_well or "Well",
                            chat_history=st.session_state.chat_history[:-1],
                            api_key=ANTHROPIC_API_KEY,
                        )
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"API error: {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})

        if st.session_state.chat_history:
            if st.button("🗑 Clear chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 7 — EXPORT
# ═══════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### Export interpreted data")
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("**CSV — full interpreted log**")
        csv_buf = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv_buf,
                           file_name=f"{st.session_state.active_well or 'well'}_interpreted.csv",
                           mime="text/csv", use_container_width=True)

    with col_e2:
        st.markdown("**LAS — with interpreted curves appended**")
        try:
            import lasio
            # Build new LAS with interpreted curves added
            raw_data = st.session_state.wells[st.session_state.active_well]["df_raw"]
            las_out = lasio.LASFile()
            las_out.well["WELL"] = lasio.HeaderItem("WELL", value=st.session_state.active_well or "WELL")
            las_out.well["STRT"] = lasio.HeaderItem("STRT", unit="m", value=float(df["DEPTH"].min()))
            las_out.well["STOP"] = lasio.HeaderItem("STOP", unit="m", value=float(df["DEPTH"].max()))
            las_out.well["STEP"] = lasio.HeaderItem("STEP", unit="m", value=round(float(df["DEPTH"].diff().median()),4))
            las_out.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)

            add_curves = ["DEPTH","GR","RHOB","NPHI","RT","DT","VCL","PHI_D","PHI_EFF","SW","UNC_PHI","UNC_SW"]
            units = {"DEPTH":"m","GR":"gAPI","RHOB":"g/cm3","NPHI":"v/v","RT":"ohm.m","DT":"us/ft",
                     "VCL":"v/v","PHI_D":"v/v","PHI_EFF":"v/v","SW":"v/v","UNC_PHI":"v/v","UNC_SW":"v/v"}
            descs = {"VCL":"Volume of clay","PHI_D":"Density porosity","PHI_EFF":"Effective porosity",
                     "SW":"Water saturation (Archie)","UNC_PHI":"Porosity uncertainty 1-sigma",
                     "UNC_SW":"Sw uncertainty 1-sigma"}
            for c in add_curves:
                if c in df.columns:
                    arr = df[c].fillna(-999.25).values
                    las_out.append_curve(c, arr, unit=units.get(c,""), descr=descs.get(c,c))

            buf = io.StringIO()
            las_out.write(buf)
            las_bytes = buf.getvalue().encode()
            st.download_button("⬇ Download LAS", las_bytes,
                               file_name=f"{st.session_state.active_well or 'well'}_interpreted.las",
                               mime="text/plain", use_container_width=True)
        except Exception as e:
            st.warning(f"LAS export requires lasio: {e}")

    st.markdown("---")
    st.markdown("**Preview — first 20 rows of interpreted output**")
    display_cols = [c for c in ["DEPTH","GR","RHOB","NPHI","RT","VCL","PHI_EFF","SW","UNC_PHI","LITHOLOGY","FLUID_FLAG"]
                    if c in df.columns]
    st.dataframe(df[display_cols].head(20).round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# WELL BROWSER — appended section; add "🗂 Well Browser" as an extra sidebar
# option. Import at top of existing app.py in a real merge.
# For now, run as: streamlit run well_browser_page.py
# ══════════════════════════════════════════════════════════════════════════════
