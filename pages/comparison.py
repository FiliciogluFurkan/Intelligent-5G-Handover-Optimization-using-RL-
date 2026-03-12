import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from environment import HandoverEnv
from stable_baselines3 import DQN, PPO
import os

st.set_page_config(page_title="Method Comparison", layout="wide")

st.title("📊 Algorithm Comparison: Baseline vs DQN vs PPO")
st.markdown("---")

def evaluate_method(method_name, env, num_episodes=5):
    """Evaluate a method"""
    results = {
        'handovers': [],
        'avg_sinr': [],
        'energy': [],
        'ping_pongs': [],
        'emergency_disc': []
    }
    
    for episode in range(num_episodes):
        env.reset()
        episode_sinr = []
        
        for step in range(500):
            user = env.users[env.current_user_idx]
            
            # Get action based on method
            if method_name == "Baseline":
                sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
                action = np.argmax(sinrs)
            elif method_name == "DQN":
                if os.path.exists("models/dqn_handover.zip"):
                    model = DQN.load("models/dqn_handover")
                    obs = env._get_observation()
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    return None
            else:  # PPO
                if os.path.exists("models/ppo_handover.zip"):
                    model = PPO.load("models/ppo_handover")
                    obs = env._get_observation()
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    return None
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Track SINR
            bs = env.base_stations[user.connected_bs]
            sinr = bs.calculate_sinr(user.position)
            episode_sinr.append(sinr)
            
            if terminated or truncated:
                break
        
        results['handovers'].append(env.total_handovers)
        results['avg_sinr'].append(np.mean(episode_sinr))
        results['energy'].append(sum(bs.total_energy for bs in env.base_stations))
        results['ping_pongs'].append(env.ping_pong_count)
        results['emergency_disc'].append(env.emergency_disconnections)
    
    return {k: np.mean(v) for k, v in results.items()}

# Run comparison
if st.button("🚀 Run Comparison", use_container_width=True):
    with st.spinner("Running simulations..."):
        env = HandoverEnv(num_users=15)
        
        baseline_results = evaluate_method("Baseline", env)
        dqn_results = evaluate_method("DQN", env)
        ppo_results = evaluate_method("PPO", env)
        
        if baseline_results and dqn_results and ppo_results:
            # Display results
            st.success("✅ Comparison completed!")
            
            # Metrics comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔵 Baseline")
                st.metric("Avg Handovers", f"{baseline_results['handovers']:.1f}")
                st.metric("Avg SINR", f"{baseline_results['avg_sinr']:.2f} dB")
                st.metric("Energy", f"{baseline_results['energy']:.2f}")
                st.metric("Ping-Pongs", f"{baseline_results['ping_pongs']:.0f}")
                st.metric("Emergency Disc.", f"{baseline_results['emergency_disc']:.0f}")
            
            with col2:
                st.markdown("### 🟢 DQN")
                st.metric("Avg Handovers", f"{dqn_results['handovers']:.1f}",
                         delta=f"{dqn_results['handovers']-baseline_results['handovers']:.1f}")
                st.metric("Avg SINR", f"{dqn_results['avg_sinr']:.2f} dB",
                         delta=f"{dqn_results['avg_sinr']-baseline_results['avg_sinr']:.2f}")
                st.metric("Energy", f"{dqn_results['energy']:.2f}",
                         delta=f"{dqn_results['energy']-baseline_results['energy']:.2f}")
                st.metric("Ping-Pongs", f"{dqn_results['ping_pongs']:.0f}",
                         delta=f"{dqn_results['ping_pongs']-baseline_results['ping_pongs']:.0f}")
                st.metric("Emergency Disc.", f"{dqn_results['emergency_disc']:.0f}",
                         delta=f"{dqn_results['emergency_disc']-baseline_results['emergency_disc']:.0f}")
            
            with col3:
                st.markdown("### 🟣 PPO")
                st.metric("Avg Handovers", f"{ppo_results['handovers']:.1f}",
                         delta=f"{ppo_results['handovers']-baseline_results['handovers']:.1f}")
                st.metric("Avg SINR", f"{ppo_results['avg_sinr']:.2f} dB",
                         delta=f"{ppo_results['avg_sinr']-baseline_results['avg_sinr']:.2f}")
                st.metric("Energy", f"{ppo_results['energy']:.2f}",
                         delta=f"{ppo_results['energy']-baseline_results['energy']:.2f}")
                st.metric("Ping-Pongs", f"{ppo_results['ping_pongs']:.0f}",
                         delta=f"{ppo_results['ping_pongs']-baseline_results['ping_pongs']:.0f}")
                st.metric("Emergency Disc.", f"{ppo_results['emergency_disc']:.0f}",
                         delta=f"{ppo_results['emergency_disc']-baseline_results['emergency_disc']:.0f}")
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Visual Comparison")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            methods = ['Baseline', 'DQN', 'PPO']
            colors = ['gray', 'green', 'purple']
            
            # Handovers
            axes[0, 0].bar(methods, [baseline_results['handovers'], 
                                     dqn_results['handovers'], 
                                     ppo_results['handovers']], color=colors)
            axes[0, 0].set_title('Average Handovers (Lower is Better)', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # SINR
            axes[0, 1].bar(methods, [baseline_results['avg_sinr'], 
                                     dqn_results['avg_sinr'], 
                                     ppo_results['avg_sinr']], color=colors)
            axes[0, 1].set_title('Average SINR (Higher is Better)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('dB')
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # Ping-Pongs
            axes[1, 0].bar(methods, [baseline_results['ping_pongs'], 
                                     dqn_results['ping_pongs'], 
                                     ppo_results['ping_pongs']], color=colors)
            axes[1, 0].set_title('Ping-Pong Count (Lower is Better)', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Emergency Disconnections
            axes[1, 1].bar(methods, [baseline_results['emergency_disc'], 
                                     dqn_results['emergency_disc'], 
                                     ppo_results['emergency_disc']], color=colors)
            axes[1, 1].set_title('Emergency Disconnections (Lower is Better)', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.error("❌ Models not found! Please train DQN and PPO models first using train.py")

st.markdown("---")
if st.button("⬅️ Back to Dashboard"):
    st.switch_page("dashboard.py")
