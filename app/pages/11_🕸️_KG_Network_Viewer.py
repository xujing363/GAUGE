import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.kg_tools import build_drug_subgraph  # noqa: E402

common.configure_page("KG Network Viewer")
bundle = common.sidebar_mode_selector()

st.title("🕸️ Knowledge-Graph Network Viewer")
st.write(
    "GAUGE's prior knowledge comes from three structured knowledge graphs. This page renders "
    "the actual graph neighbourhood around a chosen drug — the same nodes and edges the "
    "model's source-attention mechanism reasons over."
)

SOURCE_COLORS = {"ChEMBL": "#66c2a5", "DRKG": "#fc8d62", "PrimeKG": "#8da0cb"}

demo = st.button("▶️ Try the Erlotinib demo", key="kg_demo")
lib = bundle.drug_library.sort_values("DRUG_NAME")
default_name = "Erlotinib" if "Erlotinib" in lib["DRUG_NAME"].values else lib["DRUG_NAME"].iloc[0]
drug_name = st.selectbox(
    "Drug", lib["DRUG_NAME"].tolist(), index=lib["DRUG_NAME"].tolist().index(default_name) if demo else 0, key="kg_drug"
)
hops = st.slider("Neighbourhood hops", 1, 2, 1, key="kg_hops")
max_nodes = st.slider("Max nodes to display", 20, 200, 60, key="kg_max_nodes")

drug_row = lib.loc[lib["DRUG_NAME"] == drug_name].iloc[0]
drug_id = int(drug_row["DRUG_ID"])

kg = bundle.artifacts.kg_graph
coverage = kg.coverage
cov_row = coverage.loc[coverage["DRUG_ID"] == drug_id]
if cov_row.empty:
    st.warning("This drug has no knowledge-graph coverage in the current bundle.")
    st.stop()
cov_row = cov_row.iloc[0]

c1, c2, c3 = st.columns(3)
for src, col in zip(bundle.meta.get("kg_sources", []), (c1, c2, c3)):
    has = bool(cov_row.get(f"has_{src}", False))
    degree = cov_row.get(f"graph_degree_{src}", 0)
    col.metric(src, "covered" if has else "absent", f"degree {int(degree)}" if has else "")

if st.button("🚀 Render network", type="primary"):
    # Shared with the GAUGE Assistant's kg_neighborhood tool (single source of truth).
    G = build_drug_subgraph(bundle, drug_id, hops=hops, max_nodes=max_nodes)
    for n, data in G.nodes(data=True):
        if data.get("node_type") == "drug":
            data["name"] = drug_name

    if G.number_of_nodes() <= 1:
        st.info("No edges found for this drug in the bundled (filtered) edge table at this hop depth.")
    else:
        pos = nx.spring_layout(G, seed=7, k=1.2 / max(G.number_of_nodes() ** 0.5, 1))
        edge_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(width=1.2, color=SOURCE_COLORS.get(data.get("source"), "#999")),
                    hoverinfo="text", text=data.get("relation", ""), showlegend=False,
                )
            )
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n, data in G.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            is_drug = data.get("node_type") == "drug"
            node_text.append(f"{data.get('name')} ({data.get('node_type')})")
            node_color.append("#e41a1c" if is_drug else SOURCE_COLORS.get(data.get("source"), "#999"))
            node_size.append(22 if is_drug else 10)
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=[d.split(" (")[0] for d in node_text],
            textposition="top center", hovertext=node_text, hoverinfo="text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
        )
        fig = go.Figure(data=[*edge_traces, node_trace])
        fig.update_layout(
            showlegend=False, height=600, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title=f"{drug_name} — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        )
        st.plotly_chart(fig, use_container_width=True)
        legend_html = " &nbsp;&nbsp; ".join(
            f"<span style='color:{c}'>●</span> {s}" for s, c in SOURCE_COLORS.items()
        )
        st.markdown(f"{legend_html} &nbsp;&nbsp; <span style='color:#e41a1c'>●</span> {drug_name} (drug node)", unsafe_allow_html=True)

        with st.expander("Raw neighbourhood edge table"):
            edges_df = pd.DataFrame(
                [{"src": u, "dst": v, "relation": d.get("relation"), "source": d.get("source")} for u, v, d in G.edges(data=True)]
            )
            st.dataframe(edges_df, use_container_width=True)
            st.download_button(
                "⬇️ Download edges (CSV)", edges_df.to_csv(index=False).encode(),
                file_name=f"gauge_kg_network_{drug_name}.csv", mime="text/csv",
            )
