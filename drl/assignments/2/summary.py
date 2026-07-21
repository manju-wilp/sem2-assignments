"""Print a summary of the trained metrics from training_results.pkl."""
import pickle
import numpy as np

d = pickle.load(open("training_results.pkl", "rb"))
order = ["DQN-Orig", "DDQN-Orig", "DQN-Mod", "DDQN-Mod"]

print(f'{"Config":<12}{"AvgR(last100)":>14}{"Land%":>8}{"AvgQ":>9}{"Thrust":>8}{"Min":>7}')
print("-" * 58)
for tag in order:
    r = d[tag]
    print(f'{tag:<12}'
          f'{np.mean(r["rewards"][-100:]):>14.2f}'
          f'{np.mean(r["landings"][-100:]) * 100:>8.1f}'
          f'{np.mean(r["avg_q"][-100:]):>9.2f}'
          f'{np.mean(r["thrusters"][-100:]):>8.1f}'
          f'{r["minutes"]:>7.1f}')

print("\nReward (100-ep moving average) peak vs final:")
for tag in order:
    rw = d[tag]["rewards"]
    ma = [np.mean(rw[max(0, i - 99):i + 1]) for i in range(len(rw))]
    print(f"  {tag:<12} peak100={max(ma):8.2f}  final100={ma[-1]:8.2f}")

print("\nOver-estimation check (DQN AvgQ should exceed DDQN):")
for env in ["Orig", "Mod"]:
    qd = np.mean(d[f"DQN-{env}"]["avg_q"][-100:])
    qdd = np.mean(d[f"DDQN-{env}"]["avg_q"][-100:])
    print(f"  {env:<5}: DQN AvgQ={qd:7.2f}  DDQN AvgQ={qdd:7.2f}  gap={qd - qdd:6.2f}")
