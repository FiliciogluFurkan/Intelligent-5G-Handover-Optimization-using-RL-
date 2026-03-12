import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from environment import HandoverEnv
from stable_baselines3 import DQN, PPO
import time
import os

st.set_page_config(page_title="5G Handover Optimization", layout="wide")

# Title
st.title("🚀 Intelligent 5G Handover Optimization Dashboard")
st.markdown("---")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Controls")
algorithm = st.sidebar.selectbox("Select Algorithm", ["Baseline", "DQN", "PPO"])
num_users = st.sidebar.slider("Number of Users", 5, 30, 15)
simulation_speed = st.sidebar.slider("Simulation Speed", 1, 10, 5)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = HandoverEnv(num_users=num_users)
    st.session_state.env.reset()
    st.session_state.step_count = 0
    st.session_state.metrics = {
        'handovers': [],
        'avg_sinr': [],
        'energy': [],
        'ping_pongs': []
    }

# Load model
@st.cache_resource
def load_model(algo):
    if algo == "DQN" and os.path.exists("models/dqn_handover.zip"):
        return DQN.load("models/dqn_handover")
    elif algo == "PPO" and os.path.exists("models/ppo_handover.zip"):
        return PPO.load("models/ppo_handover")
    return None

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Network Visualization")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    env = st.session_state.env
    
    # Draw base stations
    bs_colors = ['red', 'blue', 'green']
    for bs in env.base_stations:
        circle = plt.Circle(bs.position, 150, color=bs_colors[bs.bs_id], alpha=0.1)
        ax.add_patch(circle)
        ax.plot(bs.position[0], bs.position[1], 'o', color=bs_colors[bs.bs_id], 
                markersize=20, label=f'BS{bs.bs_id+1}')
        ax.text(bs.position[0], bs.position[1]-20, f'BS{bs.bs_id+1}', 
                ha='center', fontsize=12, fontweight='bold')
    
    # Draw users
    for user in env.users:
        if user.user_type == "pedestrian":
            marker, color, size = 'o', 'orange', 100
        elif user.user_type == "vehicle":
            marker, color, size = 's', 'purple', 120
        else:  # emergency
            marker, color, size = '^', 'red', 150
        
        # Get connected BS color
        bs_color = bs_colors[user.connected_bs]
        ax.scatter(user.position[0], user.position[1], 
                  marker=marker, c=color, s=size, edgecolors=bs_color, linewidths=3)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plot_placeholder = st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Real-time Metrics")
    
    # Metrics display
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Handovers", st.session_state.env.total_handovers)
        st.metric("Ping-Pongs", st.session_state.env.ping_pong_count)
    with metric_col2:
        st.metric("Time Steps", st.session_state.step_count)
        st.metric("Emergency Disc.", st.session_state.env.emergency_disconnections)
    
    # User type legend
    st.markdown("### 👥 User Types")
    st.markdown("🟠 **Pedestrian** (5 km/h)")
    st.markdown("🟣 **Vehicle** (60 km/h)")
    st.markdown("🔺 **Emergency** (120 km/h)")
    
    st.markdown("---")
    st.markdown("### 📡 Base Stations")
    for bs in env.base_stations:
        load = bs.get_load()
        st.progress(load, text=f"BS{bs.bs_id+1}: {len(bs.connected_users)} users ({load*100:.0f}%)")

# Performance graphs
st.markdown("---")
st.subheader("📈 Performance Trends")

graph_col1, graph_col2, graph_col3 = st.columns(3)

with graph_col1:
    if len(st.session_state.metrics['handovers']) > 0:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(st.session_state.metrics['handovers'], color='blue')
        ax1.set_title('Cumulative Handovers')
        ax1.set_xlabel('Steps')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()

with graph_col2:
    if len(st.session_state.metrics['avg_sinr']) > 0:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(st.session_state.metrics['avg_sinr'], color='green')
        ax2.set_title('Average SINR (dB)')
        ax2.set_xlabel('Steps')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()

with graph_col3:
    if len(st.session_state.metrics['energy']) > 0:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(st.session_state.metrics['energy'], color='red')
        ax3.set_title('Total Energy Consumption')
        ax3.set_xlabel('Steps')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()

# Control buttons
st.markdown("---")
button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button("▶️ Run Simulation", use_container_width=True):
        model = load_model(algorithm)
        
        for _ in range(simulation_speed * 10):
            # Get action
            if algorithm == "Baseline":
                user = env.users[env.current_user_idx]
                sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
                action = np.argmax(sinrs)
            else:
                if model:
                    obs = env._get_observation()
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    st.error(f"{algorithm} model not found! Train the model first.")
                    break
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            st.session_state.step_count += 1
            
            # Update metrics
            avg_sinr = np.mean([bs.calculate_sinr(u.position) 
                               for u in env.users for bs in env.base_stations 
                               if u.connected_bs == bs.bs_id])
            total_energy = sum(bs.total_energy for bs in env.base_stations)
            
            st.session_state.metrics['handovers'].append(env.total_handovers)
            st.session_state.metrics['avg_sinr'].append(avg_sinr)
            st.session_state.metrics['energy'].append(total_energy)
            st.session_state.metrics['ping_pongs'].append(env.ping_pong_count)
            
            if terminated or truncated:
                break
        
        st.rerun()

with button_col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.env.reset()
        st.session_state.step_count = 0
        st.session_state.metrics = {
            'handovers': [],
            'avg_sinr': [],
            'energy': [],
            'ping_pongs': []
        }
        st.rerun()

with button_col3:
    if st.button("📊 Compare All Methods", use_container_width=True):
        st.switch_page("pages/comparison.py")
