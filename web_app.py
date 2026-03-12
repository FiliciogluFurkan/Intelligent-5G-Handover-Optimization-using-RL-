from flask import Flask, render_template, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from environment import HandoverEnv
from stable_baselines3 import DQN, PPO
import os

app = Flask(__name__)

# Global environment
env = HandoverEnv(num_users=15)
env.reset()
step_count = 0

def get_network_plot():
    """Generate network visualization"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw base stations with coverage
    bs_colors = ['red', 'blue', 'green']
    for bs in env.base_stations:
        circle = plt.Circle(bs.position, 150, color=bs_colors[bs.bs_id], alpha=0.15)
        ax.add_patch(circle)
        ax.plot(bs.position[0], bs.position[1], 'o', color=bs_colors[bs.bs_id], 
                markersize=25, markeredgecolor='black', markeredgewidth=2)
        ax.text(bs.position[0], bs.position[1]-25, f'BS{bs.bs_id+1}', 
                ha='center', fontsize=14, fontweight='bold')
    
    # Draw users
    for user in env.users:
        if user.user_type == "pedestrian":
            marker, color, size = 'o', 'orange', 150
            label = '🚶'
        elif user.user_type == "vehicle":
            marker, color, size = 's', 'purple', 180
            label = '🚗'
        else:  # emergency
            marker, color, size = '^', 'red', 200
            label = '🚑'
        
        bs_color = bs_colors[user.connected_bs]
        ax.scatter(user.position[0], user.position[1], 
                  marker=marker, c=color, s=size, edgecolors=bs_color, linewidths=4)
        ax.text(user.position[0], user.position[1]+15, label, 
                ha='center', fontsize=10)
    
    ax.set_xlim(-10, 510)
    ax.set_ylim(-10, 510)
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title('5G Network Topology - Real-time User Movement', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Pedestrian (5 km/h)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=12, label='Vehicle (60 km/h)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=12, label='Emergency (120 km/h)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_state')
def get_state():
    """Get current environment state"""
    img = get_network_plot()
    
    # Calculate metrics
    avg_sinr = np.mean([env.base_stations[u.connected_bs].calculate_sinr(u.position) 
                        for u in env.users])
    total_energy = sum(bs.total_energy for bs in env.base_stations)
    
    bs_loads = []
    for bs in env.base_stations:
        bs_loads.append({
            'id': bs.bs_id + 1,
            'users': len(bs.connected_users),
            'load': bs.get_load() * 100
        })
    
    return jsonify({
        'image': img,
        'step': step_count,
        'handovers': env.total_handovers,
        'ping_pongs': env.ping_pong_count,
        'emergency_disc': env.emergency_disconnections,
        'avg_sinr': round(avg_sinr, 2),
        'total_energy': round(total_energy, 2),
        'bs_loads': bs_loads
    })

@app.route('/step/<algorithm>')
def step_simulation(algorithm):
    """Execute simulation step"""
    global step_count
    
    # Get action based on algorithm
    if algorithm == "baseline":
        user = env.users[env.current_user_idx]
        sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
        action = int(np.argmax(sinrs))
    elif algorithm == "dqn":
        if os.path.exists("models/dqn_handover.zip"):
            model = DQN.load("models/dqn_handover")
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            return jsonify({'error': 'DQN model not found'}), 404
    elif algorithm == "ppo":
        if os.path.exists("models/ppo_handover.zip"):
            model = PPO.load("models/ppo_handover")
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            return jsonify({'error': 'PPO model not found'}), 404
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    # Execute step
    env.step(action)
    step_count += 1
    
    return jsonify({'success': True})

@app.route('/reset')
def reset_simulation():
    """Reset environment"""
    global step_count
    env.reset()
    step_count = 0
    return jsonify({'success': True})

@app.route('/comparison')
def comparison():
    """Show comparison results"""
    return render_template('comparison.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
