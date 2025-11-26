import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="SVD Analysis Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #ffffff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #2c3e50; }
    .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #2c3e50; }
    .formula-box { background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin: 10px 0; font-family: "Courier New", monospace; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("SVD Matrix Decomposition for Image Compression")

# --- Configuration ---
DEFAULT_IMAGE = "DT_photo.png"

# --- Initialize Session State ---
if 'k_val' not in st.session_state:
    st.session_state.k_val = 1

@st.cache_data
def compute_svd(img_array):
    U, s, Vt = np.linalg.svd(img_array, full_matrices=False)
    return U, s, Vt

def find_elbow_point(x_values, y_values):
    p1 = np.array([x_values[0], y_values[0]])
    p2 = np.array([x_values[-1], y_values[-1]])
    max_dist = 0
    elbow_idx = 0
    for i in range(len(x_values)):
        p3 = np.array([x_values[i], y_values[i]])
        dist = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i
    return elbow_idx

# --- Main Logic ---
if os.path.exists(DEFAULT_IMAGE):
    # Process the fixed image
    with st.spinner(f"Processing {DEFAULT_IMAGE}..."):
        image_original = Image.open(DEFAULT_IMAGE)
        image_gray = image_original.convert('L')
        img_array = np.array(image_gray) / 255.0
        m, n = img_array.shape
        
        U, s, Vt = compute_svd(img_array)
        max_rank = len(s)
        total_energy = np.sum(s**2)
        
    # Analysis Metrics
    cumulative_energy = np.cumsum(s**2) / total_energy
    elbow_idx = find_elbow_point(np.arange(len(cumulative_energy)), cumulative_energy)
    k_elbow = elbow_idx + 1
    k_90 = np.argmax(cumulative_energy >= 0.90) + 1
    k_95 = np.argmax(cumulative_energy >= 0.95) + 1

    # Initialize k_val if needed
    if 'k_val_initialized' not in st.session_state:
        st.session_state.k_val = k_elbow
        st.session_state.k_val_initialized = True

    def update_k_slider(): st.session_state.k_val = st.session_state.slider_k
    def update_k_number(): st.session_state.k_val = st.session_state.number_k

    # Rank Selection Control
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 2])
    with col_ctrl1:
        st.slider("Select Rank k", min_value=1, max_value=max_rank, value=st.session_state.k_val, key='slider_k', on_change=update_k_slider)
    with col_ctrl2:
        st.number_input("Exact k", min_value=1, max_value=max_rank, value=st.session_state.k_val, key='number_k', on_change=update_k_number)
    with col_ctrl3:
        st.markdown("##### **Optimal Suggestions:**")
        c_rec1, c_rec2, c_rec3 = st.columns(3)
        c_rec1.metric("Elbow Point", f"k={k_elbow}")
        c_rec2.metric("90% Energy", f"k={k_90}")
        c_rec3.metric("95% Energy", f"k={k_95}")

    k = st.session_state.k_val

    # Reconstruction
    reconstructed_img = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    reconstructed_display = np.clip(reconstructed_img * 255, 0, 255).astype(np.uint8)
    storage_ratio = (k * (m + n + 1)) / (m * n) * 100
    lost_energy = total_energy - np.sum(s[:k]**2)
    distortion = (lost_energy / total_energy) * 100 

    # 2. Visual Comparison
    st.markdown("### 1. Visual Reconstruction & Error Analysis")
    c1, c2, c3 = st.columns(3)
    with c1: st.image(image_gray, use_container_width=True, caption=f"Original ({m}x{n})")
    with c2: st.image(reconstructed_display, use_container_width=True, caption=f"Reconstructed (k={k})")
    with c3:
        error_img = np.abs(img_array - reconstructed_img)
        error_display = error_img / error_img.max() if error_img.max() > 0 else error_img
        st.image(error_display, use_container_width=True, clamp=True, caption="Error Heatmap (|A - A_k|)")

    # 3. Math & Formulas
    st.markdown("### 2. Mathematical Analysis")
    st.latex(r"\text{Storage Ratio} = \frac{" + f"{k}({m}+{n}+1)" + r"}{" + f"{m} \times {n}" + r"} = " + f"{storage_ratio:.2f}" + r"\%")
    st.latex(r"\text{Distortion} = \frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} = \frac{" + f"{lost_energy:.2f}" + r"}{" + f"{total_energy:.2f}" + r"} = " + f"{distortion:.2f}" + r"\%")

    # 4. Charts
    st.markdown("### 3. Statistical Charts")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(y=s[:50], mode='lines+markers', name='Singular Values'))
        fig_scree.add_trace(go.Scatter(x=[k-1], y=[s[k-1]], mode='markers', marker=dict(size=10, color='red'), name='Current k'))
        fig_scree.add_trace(go.Scatter(x=[k_elbow-1], y=[s[k_elbow-1]], mode='markers', marker=dict(size=10, color='green', symbol='star'), name='Elbow'))
        fig_scree.update_layout(title="Scree Plot (Top 50)", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scree, use_container_width=True)
    with col_chart2:
        check_points = np.unique(np.linspace(1, max_rank, 50, dtype=int).tolist() + [k, k_elbow, k_90, k_95])
        check_points.sort()
        rd_x = [(cp * (m + n + 1) / (m*n) * 100) for cp in check_points]
        rd_y = [(np.sum(s[cp:]**2) / total_energy * 100) for cp in check_points]
        fig_rd = go.Figure()
        fig_rd.add_trace(go.Scatter(x=rd_x, y=rd_y, mode='lines', name='R-D Curve'))
        fig_rd.add_trace(go.Scatter(x=[storage_ratio], y=[distortion], mode='markers', marker=dict(size=10, color='red'), name='Current'))
        fig_rd.add_trace(go.Scatter(x=[(k_elbow*(m+n+1)/(m*n)*100)], y=[(np.sum(s[k_elbow:]**2)/total_energy*100)], mode='markers', marker=dict(size=10, color='green', symbol='star'), name='Elbow'))
        fig_rd.update_layout(title="Rate-Distortion Curve", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rd, use_container_width=True)
        st.download_button(label="Download R-D Curve (HTML)", data=fig_rd.to_html(), file_name="rd_curve.html", mime="text/html")

    # 5. Rank-1 Layers
    st.markdown("---")
    st.markdown("### 4. Rank-1 Layers")
    st.markdown("Visualizing $A = \sum \sigma_i u_i v_i^T$: Image = Sum of Rank-1 Layers.")
    c_layer_sel, c_layer_view = st.columns([1, 2])
    with c_layer_sel:
        st.markdown("#### Explore Layers")
        layer_idx = st.slider("Select Layer Index (i)", 1, max_rank, 1) - 1
        layer_matrix = s[layer_idx] * np.outer(U[:, layer_idx], Vt[layer_idx, :])
        layer_display = ((layer_matrix - layer_matrix.min()) / (layer_matrix.max() - layer_matrix.min()) * 255).astype(np.uint8)
        st.image(layer_display, use_container_width=True, caption=f"Visual of Layer {layer_idx+1}")
        st.metric(f"Singular Value σ_{layer_idx+1}", f"{s[layer_idx]:.2f}")
    with c_layer_view:
        st.markdown("#### Underlying Basis Vectors")
        st.caption(f"Layer {layer_idx+1} is the outer product of these two vectors.")
        u_explanation = ("**For i=1:** Represents dominant vertical brightness/structure. "
                       "**For i>1:** Captures vertical textures/edges.")
        v_explanation = ("**For i=1:** Represents dominant horizontal brightness/structure. "
                       "**For i>1:** Captures horizontal textures/edges.")
        col_vec1, col_vec2 = st.columns(2)
        with col_vec1:
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(y=U[:, layer_idx], mode='lines', name=f'u_{layer_idx+1}', line=dict(color='blue')))
            fig_u.update_layout(title=f"u_{layer_idx+1} (Column Basis)", height=200, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
            st.plotly_chart(fig_u, use_container_width=True)
            st.info(u_explanation)
        with col_vec2:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(y=Vt[layer_idx, :], mode='lines', name=f'v_{layer_idx+1}', line=dict(color='green')))
            fig_v.update_layout(title=f"v_{layer_idx+1} (Row Basis)", height=200, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
            st.plotly_chart(fig_v, use_container_width=True)
            st.info(v_explanation)

    # 6. Matrix Details
    st.markdown("---")
    st.markdown("### 5. Matrix Values (Truncated)")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.caption(f"U (First {k} cols)")
        st.dataframe(pd.DataFrame(U[:, :k]), height=200)
    with mc2:
        st.caption(f"Sigma (Top {k})")
        st.dataframe(pd.DataFrame(s[:k], columns=["Val"]), height=200)
    with mc3:
        st.caption(f"V^T (First {k} rows)")
        st.dataframe(pd.DataFrame(Vt[:k, :]), height=200)
        
    # 7. PPT Export
    st.markdown("---")
    st.markdown("### 6. PowerPoint Export Tools")
    if st.button("Generate Interactive Reconstruction Animation"):
        frames_k = np.unique(np.geomspace(1, max_rank, 30, dtype=int))
        fig_anim = go.Figure(
            data=[go.Heatmap(z=np.flipud(img_array), colorscale='gray', showscale=False)],
            layout=go.Layout(
                title="SVD Reconstruction Animation",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False, scaleanchor="x"),
                width=600, height=600 * (m/n),
                updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]
            )
        )
        frames = []
        steps = []
        with st.spinner("Rendering animation frames..."):
            for k_frame in frames_k:
                rec = U[:, :k_frame] @ np.diag(s[:k_frame]) @ Vt[:k_frame, :]
                rec_flipped = np.flipud(rec) 
                frame = go.Frame(data=[go.Heatmap(z=rec_flipped, colorscale='gray', showscale=False, zmin=0, zmax=1)], name=str(k_frame))
                frames.append(frame)
                step = dict(method="animate", args=[[str(k_frame)], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0))], label=str(k_frame))
                steps.append(step)
        fig_anim.frames = frames
        fig_anim.layout.sliders = [dict(active=0, steps=steps, currentvalue={"prefix": "Rank k: "})]
        st.plotly_chart(fig_anim, use_container_width=True)
        st.download_button(label="Download Animation HTML", data=fig_anim.to_html(), file_name="svd_animation.html", mime="text/html")

else:
    st.error(f"Error: Image file '{DEFAULT_IMAGE}' not found. Please ensure it is in the GitHub repository.")
