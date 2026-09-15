"""
fedsuper_simulation/src/graph_visualizer.py
Interactive Plotly Network Flow Graph Visualizer for Privacy-Preserving Federated SUPER.

Strictly visually differentiates between 3 architectural data streams:
1. Stream 1 (Local Private Data - Zero Egress):
   - Coral Orange (#FF7043) glowing markers and circular sandbox halos enclosing each client node.
   - User interaction histories, raw ratings D_u, and latent factors p_u remain strictly on-device.
2. Stream 2 (DP-Clipped Gradients):
   - Electric Neon Cyan (#00E5FF) directed transmission lines and in-flight gradient flow packets.
   - Differentially private item parameter updates with L2 gradient clipping (C) and Gaussian noise perturbation (sigma).
3. Stream 3 (LLM Semantic Blueprints):
   - Electric Magenta (#D500F9) high-energy beam and semantic pulse markers.
   - High-dimensional semantic taste embeddings streamed from the LLM Knowledge Base into the central server.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go


# =============================================================================
# COLOR PALETTE CONSTANTS
# =============================================================================
COLOR_STREAM_1_ORANGE = "#FF7043"   # Vibrant Coral Orange (Local Private Enclave - Zero Egress)
COLOR_STREAM_2_CYAN = "#00E5FF"     # Electric Neon Cyan (DP-Clipped Gradients Uplink)
COLOR_STREAM_3_MAGENTA = "#D500F9"  # Electric Magenta (LLM Semantic Blueprints)
COLOR_SERVER_BLUE = "#1E88E5"       # Electric Royal Blue (Central Parameter Server)
COLOR_CLIENT_ACTIVE = "#00E676"     # Neon Emerald Green (Active Client Core)
COLOR_CLIENT_INACTIVE = "#546E7A"   # Subtle Slate Gray (Standby Client Core)
COLOR_BG_PAPER = "#0E1117"          # Streamlit Dark Paper Background
COLOR_BG_PLOT = "#161B22"           # Elevated Dark Plot Background
COLOR_GRID = "#30363D"              # Low-Contrast Border / Grid
COLOR_TEXT_PRIMARY = "#E6EDF3"      # High-Legibility Light Text
COLOR_TEXT_MUTED = "#8B949E"        # Muted Secondary Text
COLOR_EDGE_FAINT = "rgba(100, 116, 139, 0.15)"  # Translucent Topology Mesh

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


# =============================================================================
# THEME HELPER
# =============================================================================
def apply_dark_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 520,
    margin: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Applies the unified cyber dark layout theme to a Plotly network figure.
    """
    if margin is None:
        margin = dict(l=20, r=20, t=50 if title else 25, b=50)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG_PAPER,
        plot_bgcolor=COLOR_BG_PLOT,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family=FONT_FAMILY, size=15, color=COLOR_TEXT_PRIMARY),
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top"
        ) if title else None,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXT_PRIMARY, size=12),
        margin=margin,
        height=height,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[-1.55, 1.55],
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[-1.35, 1.65],
            scaleanchor="x",
            scaleratio=1.0,
            fixedrange=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(22, 27, 34, 0.85)",
            bordercolor="rgba(48, 54, 61, 0.7)",
            borderwidth=1,
            font=dict(size=10, color=COLOR_TEXT_PRIMARY)
        ),
        hoverlabel=dict(
            bgcolor="#1F2937",
            bordercolor="#374151",
            font=dict(family=FONT_FAMILY, size=12, color="#F3F4F6")
        ),
        hovermode="closest"
    )
    return fig


# =============================================================================
# 1. TOPOLOGY & COORDINATE ENGINE
# =============================================================================
def get_network_topology_layout(
    num_clients: int,
    radius: float = 1.0,
    llm_offset: Tuple[float, float] = (0.0, 1.35),
    layout_type: str = "circular"
) -> Dict[str, Any]:
    """
    Computes 2D Cartesian coordinates for Central Server, LLM Knowledge Base, and Client nodes.

    Parameters:
        num_clients: Number of client nodes K to position (must handle 0, 1, or large K).
        radius: Orbit radius R for client ring (default 1.0).
        llm_offset: (x, y) coordinates for LLM Knowledge Base (default (0.0, 1.35)).
        layout_type: 'circular' (full 360-deg orbit) or 'horseshoe' (open top arc).

    Returns:
        Dict with keys:
            'server': {'x': 0.0, 'y': 0.0, 'id': 0}
            'llm': {'x': llm_offset[0], 'y': llm_offset[1], 'id': -1}
            'clients': List[Dict[str, Any]] containing 'id', 'x', 'y', 'theta'
    """
    server_dict = {"x": 0.0, "y": 0.0, "id": 0}
    llm_dict = {"x": float(llm_offset[0]), "y": float(llm_offset[1]), "id": -1}
    clients = []

    if num_clients <= 0:
        return {
            "server": server_dict,
            "llm": llm_dict,
            "clients": []
        }

    if num_clients == 1:
        clients.append({
            "id": 0,
            "x": 0.0,
            "y": -float(radius),
            "theta": -np.pi / 2.0
        })
        return {
            "server": server_dict,
            "llm": llm_dict,
            "clients": clients
        }

    for i in range(num_clients):
        if layout_type == "horseshoe":
            # Horseshoe arc with top clearance beneath LLM node
            theta = -np.pi / 2.0 + (i - (num_clients - 1) / 2.0) * (1.6 * np.pi / max(num_clients - 1, 1))
        else:
            # Full 360-degree circular orbit
            theta = -np.pi / 2.0 + (2.0 * np.pi * i / num_clients)

        x = float(radius * np.cos(theta))
        y = float(radius * np.sin(theta))
        clients.append({
            "id": i,
            "x": x,
            "y": y,
            "theta": float(theta)
        })

    return {
        "server": server_dict,
        "llm": llm_dict,
        "clients": clients
    }


def compute_radial_layout(
    num_clients: int,
    radius: float = 1.0,
    server_pos: Tuple[float, float] = (0.0, 0.0),
    llm_pos: Tuple[float, float] = (0.0, 1.35)
) -> Dict[str, Any]:
    """
    Backward-compatibility alias and helper for compute_radial_layout.
    """
    topology = get_network_topology_layout(num_clients, radius=radius, llm_offset=llm_pos)
    client_positions = [(c["x"], c["y"]) for c in topology["clients"]]
    angles = [c["theta"] for c in topology["clients"]]
    return {
        "server": server_pos,
        "llm": llm_pos,
        "clients": client_positions,
        "angles": angles,
        "radius": radius
    }


# =============================================================================
# 2. STREAM 1: LOCAL PRIVATE DATA (ZERO EGRESS HALOS & NODES)
# =============================================================================
def create_client_sandbox_halos_trace(
    topology: Dict[str, Any],
    dataset: Optional[Any] = None,
    active_client_ids: Optional[List[int]] = None,
    dp_clip_norm: float = 1.0
) -> go.Scatter:
    """
    Constructs Stream 1 circular privacy sandbox halos (#FF7043) enclosing each client.
    Represents on-device local enclaves with zero egress of private user data.
    """
    clients = topology.get("clients", [])
    if not clients:
        return go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=28, color="rgba(255, 112, 67, 0.25)", line=dict(color=COLOR_STREAM_1_ORANGE, width=2.5)),
            name="🔒 Stream 1: Local Private Enclave (Zero Egress)",
            showlegend=True
        )

    active_set = set(active_client_ids or [])
    xs = [c["x"] for c in clients]
    ys = [c["y"] for c in clients]
    hover_texts = []

    for c in clients:
        cid = c["id"]
        status_str = "ACTIVE (DP Uplink)" if cid in active_set else "STANDBY (Enclave Only)"
        
        # Extract metadata if available
        interaction_cnt = 25
        preferred_genres = "Sci-Fi, Action, Drama"
        head_p, torso_p, tail_p = 33.3, 33.3, 33.4

        if dataset is not None:
            if hasattr(dataset, "train_matrix") and cid < dataset.train_matrix.shape[0]:
                interaction_cnt = int(np.sum(dataset.train_matrix[cid]))
            if hasattr(dataset, "user_metadata") and hasattr(dataset.user_metadata, "iloc") and cid < len(dataset.user_metadata):
                row = dataset.user_metadata.iloc[cid]
                interaction_cnt = int(row.get("activity_level", interaction_cnt))
                preferred_genres = str(row.get("preferred_genres", preferred_genres))
                head_p = float(row.get("pop_u_true", 0.35)) * 100.0
                torso_p = float(row.get("pop_u_torso", 0.30)) * 100.0
                tail_p = float(row.get("pop_u_tail", 0.35)) * 100.0

        hover_text = (
            f"<b>🔒 Federated Client #{cid}</b><br>"
            f"• <b>Status</b>: {status_str}<br>"
            f"• <b>Local Interactions</b>: {interaction_cnt} items (Private D_u)<br>"
            f"• <b>Taste Profile</b>: {preferred_genres}<br>"
            f"• <b>Target Blueprint P_u</b>: Head {head_p:.1f}%, Torso {torso_p:.1f}%, Tail {tail_p:.1f}%<br>"
            f"• <b>Differential Privacy</b>: L2 Clip ≤ {dp_clip_norm:.2f}<br>"
            f"• <b>Data Privacy</b>: <span style='color:{COLOR_STREAM_1_ORANGE};'><b>ZERO EGRESS (Never Leaves Device)</b></span>"
        )
        hover_texts.append(hover_text)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            size=30,
            color="rgba(255, 112, 67, 0.22)",
            line=dict(color=COLOR_STREAM_1_ORANGE, width=2.5),
            symbol="circle"
        ),
        name="🔒 Stream 1: Local Private Enclave (Zero Egress)",
        hoverinfo="text",
        hovertext=hover_texts,
        legendgroup="stream1",
        showlegend=True
    )


# Alias
create_sandbox_halos_trace = create_client_sandbox_halos_trace


def create_client_nodes_trace(
    topology: Dict[str, Any],
    dataset: Optional[Any] = None,
    active_client_ids: Optional[List[int]] = None,
    dp_clip_norm: float = 1.0,
    dp_sigma: float = 0.0,
    dp_enabled: bool = True
) -> go.Scatter:
    """
    Constructs Client Core Node markers (active glowing emerald #00E676,
    standby slate #546E7A, bordered by white/orange).
    """
    clients = topology.get("clients", [])
    if not clients:
        return go.Scatter(x=[None], y=[None], mode="markers", showlegend=False)

    active_set = set(active_client_ids or [])
    xs = [c["x"] for c in clients]
    ys = [c["y"] for c in clients]
    colors = [COLOR_CLIENT_ACTIVE if c["id"] in active_set else COLOR_CLIENT_INACTIVE for c in clients]
    sizes = [16 if c["id"] in active_set else 12 for c in clients]
    texts = [f"C{c['id']}" for c in clients]
    hover_texts = []

    for c in clients:
        cid = c["id"]
        is_act = cid in active_set
        act_text = "ACTIVE (Transmitting DP Gradients)" if is_act else "STANDBY (Idle Enclave)"
        dp_text = f"L2 Clip ≤ {dp_clip_norm:.2f}, σ = {dp_sigma:.3f}" if dp_enabled else "DP Disabled"
        hover_texts.append(
            f"<b>Client Device #{cid}</b><br>"
            f"• State: {act_text}<br>"
            f"• Privacy Mode: {dp_text}<br>"
            f"• Core Memory: Local Enclave p_u"
        )

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color="#FFFFFF", width=1.5),
            symbol="circle"
        ),
        text=texts,
        textposition="bottom center",
        textfont=dict(color="#FFFFFF", size=10, family=FONT_FAMILY),
        name="Client Devices",
        hoverinfo="text",
        hovertext=hover_texts,
        legendgroup="topology",
        showlegend=False
    )


# =============================================================================
# 3. STREAM 2: DP-CLIPPED GRADIENT UPLINKS
# =============================================================================
def create_gradient_flow_traces(
    topology: Dict[str, Any],
    active_client_ids: Optional[List[int]] = None,
    dp_clip_norm: float = 1.0,
    dp_sigma: float = 0.0,
    dp_epsilon: float = 4.0
) -> List[go.Scatter]:
    """
    Constructs Stream 2 Cyan (#00E5FF) directed transmission lines and in-flight flow
    particle markers between active sampled clients and the central server.
    """
    server = topology.get("server", {"x": 0.0, "y": 0.0})
    clients = {c["id"]: c for c in topology.get("clients", [])}
    active_ids = active_client_ids or []

    edge_x = []
    edge_y = []
    particle_x = []
    particle_y = []
    particle_hovers = []

    for cid in active_ids:
        if cid in clients:
            c = clients[cid]
            cx, cy = c["x"], c["y"]
            sx, sy = server["x"], server["y"]
            
            # Transmission line from client to server
            edge_x.extend([cx, sx, None])
            edge_y.extend([cy, sy, None])

            # In-flight gradient flow particles at fractions
            for frac in [0.35, 0.70]:
                px = cx + frac * (sx - cx)
                py = cy + frac * (sy - cy)
                particle_x.append(px)
                particle_y.append(py)
                particle_hovers.append(
                    f"<b>🛡️ Stream 2: DP Gradient Uplink</b><br>"
                    f"• Source: Client #{cid} → Central Server<br>"
                    f"• L2 Clip Bound: C = {dp_clip_norm:.2f}<br>"
                    f"• Perturbation: Gaussian Noise σ = {dp_sigma:.3f}<br>"
                    f"• Privacy Budget: ε = {dp_epsilon:.2f}"
                )

    traces = []
    # Master Line Trace
    line_trace = go.Scatter(
        x=edge_x if edge_x else [None],
        y=edge_y if edge_y else [None],
        mode="lines",
        line=dict(color=COLOR_STREAM_2_CYAN, width=3.2),
        name="🛡️ Stream 2: DP-Clipped Gradients (Uplink)",
        hoverinfo="text",
        hovertext=[
            f"<b>🛡️ Stream 2: DP Gradient Uplink Channel</b><br>"
            f"• L2 Clip Bound: C = {dp_clip_norm:.2f}<br>"
            f"• Gaussian Noise: σ = {dp_sigma:.3f}<br>"
            f"• Privacy Budget: ε = {dp_epsilon:.2f}"
        ] * max(len(edge_x) // 3, 1),
        legendgroup="stream2",
        showlegend=True
    )
    traces.append(line_trace)

    # In-Flight Particle Trace
    if particle_x:
        particle_trace = go.Scatter(
            x=particle_x,
            y=particle_y,
            mode="markers",
            marker=dict(
                size=8,
                color=COLOR_STREAM_2_CYAN,
                line=dict(color="#FFFFFF", width=1.2),
                symbol="diamond"
            ),
            name="🛡️ In-Flight DP Gradients",
            hoverinfo="text",
            hovertext=particle_hovers,
            legendgroup="stream2",
            showlegend=False
        )
        traces.append(particle_trace)

    return traces


# =============================================================================
# 4. STREAM 3: LLM SEMANTIC BLUEPRINT STREAM
# =============================================================================
def create_llm_stream_trace(
    topology: Dict[str, Any],
    llm_lambda: float = 0.70
) -> List[go.Scatter]:
    """
    Constructs Stream 3 Magenta (#D500F9) high-energy beam and pulse markers connecting
    the LLM Knowledge Base to the Central Parameter Server.
    """
    server = topology.get("server", {"x": 0.0, "y": 0.0})
    llm = topology.get("llm", {"x": 0.0, "y": 1.35})

    sx, sy = server["x"], server["y"]
    lx, ly = llm["x"], llm["y"]

    # Beam line trace
    beam_trace = go.Scatter(
        x=[lx, sx],
        y=[ly, sy],
        mode="lines+markers",
        line=dict(color=COLOR_STREAM_3_MAGENTA, width=4.5),
        marker=dict(size=[14, 10], color=COLOR_STREAM_3_MAGENTA, symbol=["diamond", "triangle-down"]),
        name=f"🧠 Stream 3: LLM Blueprints (λ={llm_lambda:.2f})",
        hoverinfo="text",
        hovertext=[
            f"<b>🧠 Stream 3: LLM Semantic Stream</b><br>"
            f"• Origin: LLM Knowledge Base<br>"
            f"• Target: Central Parameter Server<br>"
            f"• Fusion Weight (λ): {llm_lambda:.2f}<br>"
            f"• Mode: Active Blueprint Stream (#D500F9)",
            f"<b>🛡️ Central Server Hub</b><br>Receiving LLM Semantic Blueprints"
        ],
        legendgroup="stream3",
        showlegend=True
    )

    # In-flight semantic pulse particles along the beam
    particle_t = np.linspace(0.20, 0.80, 4)
    particle_x = [lx + t * (sx - lx) for t in particle_t]
    particle_y = [ly + t * (sy - ly) for t in particle_t]
    pulse_trace = go.Scatter(
        x=particle_x,
        y=particle_y,
        mode="markers",
        marker=dict(
            size=9,
            color=COLOR_STREAM_3_MAGENTA,
            line=dict(color="#FFFFFF", width=1.5),
            symbol="circle"
        ),
        name="🧠 Semantic Vector Packets",
        hoverinfo="text",
        hovertext=[
            f"<b>🧠 LLM Semantic Vector Packet #{i+1}</b><br>"
            f"• Vector Dim: 32-d<br>"
            f"• Standardization: Intra-Pool Z-Score"
            for i in range(len(particle_t))
        ],
        legendgroup="stream3",
        showlegend=False
    )

    return [beam_trace, pulse_trace]


# =============================================================================
# 5. INFRASTRUCTURE & AGGREGATION NODES
# =============================================================================
def create_server_node_trace(
    topology: Dict[str, Any],
    current_round: int = 0,
    active_count: int = 0,
    num_items: int = 100,
    llm_lambda: float = 0.70,
    calibration_alpha: float = 0.40
) -> go.Scatter:
    """
    Constructs Central Parameter Server node trace (#1E88E5) at coordinate (0, 0).
    """
    server = topology.get("server", {"x": 0.0, "y": 0.0})
    server_hover = (
        f"<b>🛡️ Central Parameter Server</b><br>"
        f"• <b>Simulation Round</b>: Round {current_round}<br>"
        f"• <b>Active Uplinks</b>: {active_count} Client Enclaves<br>"
        f"• <b>Global Catalog</b>: {num_items} Items<br>"
        f"• <b>Aggregation Engine</b>: Sparse FedAvg (Touched Items Only)<br>"
        f"• <b>SUPER Calibration Weight</b>: α = {calibration_alpha:.2f}<br>"
        f"• <b>LLM Semantic Fusion Weight</b>: λ = {llm_lambda:.2f}"
    )

    return go.Scatter(
        x=[server["x"]],
        y=[server["y"]],
        mode="markers+text",
        marker=dict(
            size=36,
            color=COLOR_SERVER_BLUE,
            line=dict(color="#FFFFFF", width=3.0),
            symbol="hexagon"
        ),
        text=["Server"],
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=12, family=FONT_FAMILY),
        name="Central Server (FedAvg Hub)",
        hoverinfo="text",
        hovertext=[server_hover],
        legendgroup="topology",
        showlegend=True
    )


def create_llm_node_trace(
    topology: Dict[str, Any],
    llm_dim: int = 32,
    llm_lambda: float = 0.70,
    num_genres: int = 6
) -> go.Scatter:
    """
    Constructs LLM Semantic Knowledge Base diamond node (#D500F9) at coordinate (0, 1.35).
    """
    llm = topology.get("llm", {"x": 0.0, "y": 1.35})
    llm_hover = (
        f"<b>🧠 LLM Semantic Knowledge Base</b><br>"
        f"• <b>Representation Space</b>: {llm_dim}-dimensional Semantic Embeddings<br>"
        f"• <b>Semantic Anchors</b>: {num_genres} Genre Centroids<br>"
        f"• <b>Fusion Coefficient (λ)</b>: {llm_lambda:.2f}<br>"
        f"• <b>Transmission Stream</b>: <span style='color:{COLOR_STREAM_3_MAGENTA};'><b>Active Blueprint Stream (#D500F9)</b></span>"
    )

    return go.Scatter(
        x=[llm["x"]],
        y=[llm["y"]],
        mode="markers+text",
        marker=dict(
            size=34,
            color=COLOR_STREAM_3_MAGENTA,
            line=dict(color="#FFFFFF", width=2.8),
            symbol="diamond"
        ),
        text=["LLM Base"],
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=11, family=FONT_FAMILY),
        name="LLM Knowledge Base",
        hoverinfo="text",
        hovertext=[llm_hover],
        legendgroup="topology",
        showlegend=True
    )


def create_background_mesh_trace(
    topology: Dict[str, Any]
) -> go.Scatter:
    """
    Constructs subtle idle network topology lines (translucent slate) connecting all clients to the server.
    """
    server = topology.get("server", {"x": 0.0, "y": 0.0})
    clients = topology.get("clients", [])
    mesh_x = []
    mesh_y = []

    for c in clients:
        mesh_x.extend([server["x"], c["x"], None])
        mesh_y.extend([server["y"], c["y"], None])

    return go.Scatter(
        x=mesh_x if mesh_x else [None],
        y=mesh_y if mesh_y else [None],
        mode="lines",
        line=dict(color=COLOR_EDGE_FAINT, width=1.2),
        hoverinfo="none",
        name="Network Topology Mesh",
        legendgroup="topology",
        showlegend=False
    )


# =============================================================================
# 6. MASTER VISUALIZER ENTRYPOINT
# =============================================================================
def render_network_flow_figure(
    dataset: Optional[Any] = None,
    active_client_ids: Optional[List[int]] = None,
    current_round: int = 0,
    dp_clip_norm: float = 1.0,
    dp_epsilon: float = 4.0,
    dp_sigma: float = 0.0,
    llm_lambda: float = 0.70,
    calibration_alpha: float = 0.40,
    theme: str = "plotly_dark"
) -> go.Figure:
    """
    Assembles and returns the complete 2D interactive Plotly Network Flow Figure.
    Features:
      - 3 strictly differentiated streams:
        1. Orange (#FF7043): Local Private Data Enclaves (Zero Egress).
        2. Cyan (#00E5FF): DP-Clipped Gradient Uplink Channel.
        3. Magenta (#D500F9): LLM Semantic Blueprint Stream.
      - Cyberpunk dark layout (#0E1117 / #161B22).
      - Interactive legend grouping for stream toggling.
      - Rich custom hover tooltips on all nodes, halos, and streams.
      - 1:1 aspect ratio locking.
      - Defensive fallback handling for empty or None states.

    Parameters:
        dataset: SyntheticDataset or None.
        active_client_ids: List of client IDs participating in current communication round.
        current_round: Current federated communication round number.
        dp_clip_norm: Differential privacy L2 clipping threshold (C).
        dp_epsilon: Target DP privacy budget (epsilon).
        dp_sigma: Differential privacy noise scale (sigma).
        llm_lambda: LLM semantic fusion weight (lambda).
        calibration_alpha: SUPER popularity calibration weight (alpha).
        theme: Plotly layout template name.

    Returns:
        go.Figure: Interactive Plotly visualization.
    """
    # 1. Extract or synthesize client count safely
    if dataset is not None:
        if hasattr(dataset, "num_users"):
            num_clients = int(dataset.num_users)
        elif hasattr(dataset, "train_matrix"):
            num_clients = int(dataset.train_matrix.shape[0])
        elif hasattr(dataset, "user_metadata"):
            num_clients = len(dataset.user_metadata)
        elif isinstance(dataset, int):
            num_clients = dataset
        else:
            num_clients = 20
    else:
        num_clients = 20

    # 2. Extract catalog size safely
    num_items = 100
    if dataset is not None and hasattr(dataset, "num_items"):
        num_items = int(dataset.num_items)
    elif dataset is not None and hasattr(dataset, "item_metadata"):
        num_items = len(dataset.item_metadata)

    # 3. Filter active client IDs
    if active_client_ids is None:
        active_client_ids = [0, 1, 2] if num_clients >= 3 else ([0] if num_clients == 1 else [])
    
    valid_active_ids = [cid for cid in active_client_ids if 0 <= cid < num_clients]

    # 4. Generate topology layout
    topology = get_network_topology_layout(num_clients=num_clients, radius=1.0)

    # 5. Build Figure Traces
    fig = go.Figure()

    # Layer 1: Background idle mesh
    fig.add_trace(create_background_mesh_trace(topology))

    # Layer 2: Stream 3 — LLM Semantic Blueprint Beam & Pulses (Magenta #D500F9)
    for trace in create_llm_stream_trace(topology, llm_lambda=llm_lambda):
        fig.add_trace(trace)

    # Layer 3: Stream 2 — DP-Clipped Gradient Uplinks (Cyan #00E5FF)
    for trace in create_gradient_flow_traces(
        topology,
        active_client_ids=valid_active_ids,
        dp_clip_norm=dp_clip_norm,
        dp_sigma=dp_sigma,
        dp_epsilon=dp_epsilon
    ):
        fig.add_trace(trace)

    # Layer 4: Stream 1 — Local Private Data Sandboxes (Orange #FF7043)
    fig.add_trace(create_client_sandbox_halos_trace(
        topology,
        dataset=dataset,
        active_client_ids=valid_active_ids,
        dp_clip_norm=dp_clip_norm
    ))

    # Layer 5: Client Core Node Markers
    fig.add_trace(create_client_nodes_trace(
        topology,
        dataset=dataset,
        active_client_ids=valid_active_ids,
        dp_clip_norm=dp_clip_norm,
        dp_sigma=dp_sigma,
        dp_enabled=dp_clip_norm > 0
    ))

    # Layer 6: Central Parameter Server Node (#1E88E5)
    fig.add_trace(create_server_node_trace(
        topology,
        current_round=current_round,
        active_count=len(valid_active_ids),
        num_items=num_items,
        llm_lambda=llm_lambda,
        calibration_alpha=calibration_alpha
    ))

    # Layer 7: LLM Knowledge Base Node (#D500F9)
    fig.add_trace(create_llm_node_trace(
        topology,
        llm_dim=32,
        llm_lambda=llm_lambda,
        num_genres=6
    ))

    # 6. Apply Cyber Dark Theme & Aspect Ratio
    apply_dark_theme(
        fig,
        title=f"Privacy-Preserving Federated Architecture & Data Flow (Round {current_round})",
        height=540
    )

    return fig
