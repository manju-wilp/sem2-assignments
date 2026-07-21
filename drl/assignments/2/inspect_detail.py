"""Deeper inspection to explain thruster-usage and Q-value trends honestly."""
import pickle
import numpy as np

d = pickle.load(open("training_results.pkl", "rb"))
order = ["DQN-Orig", "DDQN-Orig", "DQN-Mod", "DDQN-Mod"]

print("Per-episode detail (mean over last 100 episodes):")
print(f'{"Config":<12}{"Steps":>9}{"Thrust":>9}{"Thr/Step":>10}{"Reward":>10}{"Land%":>8}')
print("-" * 58)
for tag in order:
    r = d[tag]
    steps = np.mean(r["steps"][-100:])
    thr = np.mean(r["thrusters"][-100:])
    print(f'{tag:<12}{steps:>9.1f}{thr:>9.1f}{thr / steps:>10.3f}'
          f'{np.mean(r["rewards"][-100:]):>10.2f}{np.mean(r["landings"][-100:]) * 100:>8.1f}')

print("\nAbsolute |DQN - DDQN| predicted-Q gap (last 100):")
for env in ["Orig", "Mod"]:
    qd = np.mean(d[f"DQN-{env}"]["avg_q"][-100:])
    qdd = np.mean(d[f"DDQN-{env}"]["avg_q"][-100:])
    print(f"  {env:<5}: |gap| = {abs(qd - qdd):6.2f}  (DQN={qd:7.2f}, DDQN={qdd:7.2f})")
