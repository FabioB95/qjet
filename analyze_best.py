# analyze_best.py
import json, os
import numpy as np
import matplotlib.pyplot as plt

def bitstring_to_spins(bitstring, K):
    """
    Converte un bitstring ('0101') in array di spin ±1.
    NB: IBM ordina i qubit al contrario → prendiamo ultimi K bit.
    """
    bit = bitstring.replace(" ", "")[-K:]
    spins = np.array([+1 if b == "0" else -1 for b in bit[::-1]], dtype=int)
    return spins

def analyze_counts(counts_file, K, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(counts_file) as f:
        counts = json.load(f)

    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    # Salva top-5 configurazioni
    top_items = list(probs_sorted.items())[:5]
    spins_list = []
    for bitstring, p in top_items:
        spins = bitstring_to_spins(bitstring, K)
        spins_list.append((spins, p))
        print(f"{bitstring} -> spins={spins} prob={p:.3f}")

    # Salva come npy
        # Salva come npy (dtype=object per liste eterogenee)
    np.save(os.path.join(out_dir, f"top_spins.npy"),
            np.array(spins_list, dtype=object), allow_pickle=True)

    # Oppure, per più chiarezza, salva in JSON
    spins_json = [
        {"bitstring": b, "spins": spins.tolist(), "prob": float(p)}
        for (spins, p), (b, _) in zip(spins_list, top_items)
    ]
    with open(os.path.join(out_dir, f"top_spins.json"), "w") as f:
        json.dump(spins_json, f, indent=2)


    # Barplot probabilità
    plt.figure(figsize=(8,5))
    labels = [b for b, _ in top_items]
    values = [p for _, p in top_items]
    plt.bar(labels, values)
    plt.xlabel("Bitstring")
    plt.ylabel("Probability")
    plt.title(f"Top configurations (K={K})")
    plt.savefig(os.path.join(out_dir, f"top_configs.png"))
    plt.close()
    print(f"Salvati risultati in {out_dir}")

if __name__ == "__main__":
    configs = [
        ("figures/K8_counts.json", 8, "figures/K8_analysis"),
        ("figures/K12_counts.json", 12, "figures/K12_analysis"),
        ("figures/K16_counts.json", 16, "figures/K16_analysis"),
    ]
    for counts_file, K, out_dir in configs:
        analyze_counts(counts_file, K, out_dir)
