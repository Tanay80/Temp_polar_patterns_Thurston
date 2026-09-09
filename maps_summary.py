import matplotlib.pyplot as plt

models = [
    r"$\mathbb{R} \times S^2$",
    r"$\mathbb{R} \times \mathbb{H}^2$",
    r"$\widetilde{U \left(\mathbb{H}^2 \right)}$",
    "Nil",
    "Solv"
]

T = [1.63e-7, 1.74e-5, 2.02e-5, 1.67e-5, 3.11e-5]
P = [3.27e-7, 3.60e-7, 9.69e-7, 3.60e-7, 1.11e-6]
Q = [3.14e-7, 3.46e-7, 9.69e-7, 3.46e-7, 1.03e-6]
U = [3.15e-7, 3.48e-7, 5.18e-7, 3.50e-7, 1.06e-6]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

quantities = [
    (T, "Average T", "red"),
    (P, "Average P", "green"),
    (Q, "Average Q", "yellow"),
    (U, "Average U", "blue")
]

for ax, (values, label, color) in zip(axes.flat, quantities):
    bars = ax.bar(models, values, color=color)
    scaled_labels = [f"{v * 1e5:.2f}" for v in values]
    ax.bar_label(bars, labels=scaled_labels,padding=3, fontsize=8)
    #ax.set_yscale("log")
    ax.set_ylabel(label)
    #ax.set_title(label)
    ax.tick_params(axis="x")
    ax.grid(True)

plt.savefig("maps_summary.png", dpi=300, bbox_inches="tight")
