import stim
import numpy as np
import sinter
import math
from toric_code import ToricCode
from typing import List
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Generating Toric Code circuits...")
    tasks = [
        sinter.Task(
            circuit=ToricCode(d=d, rounds=d*3, noise_prob=noise).circuit,
            json_metadata={'d': d, 'p': noise},
        )
        for d in [3, 5, 7]
        for noise in np.linspace(0.001, 0.02, 10)
    ]

    # Run
    print(f"Starting Sinter collection with {len(tasks)} tasks...")
    collected_stats = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=10000,
        max_errors=1000,
        print_progress=True # Progress bar
    )

    # Plot
    print(r"\n--- SIMULATION COMPLETE ---")
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax.loglog()
    ax.set_title("Toric Code Error Rates (Circuit-Level Noise)")
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
    plt.show()
