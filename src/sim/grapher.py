import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_price_paths(npy_file="data/paths.npy",
                     output_file="data/price_paths_plot.png",
                     show_plot=False,
                     max_paths=None,
                     highlight_paths=5):
    print(f"Loading price paths from {npy_file}...")
    try:
        paths = np.load(npy_file)
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return

    num_paths, num_steps = paths.shape
    print(f"Found {num_paths} paths with {num_steps} time steps each")

    time_steps = np.arange(num_steps)
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))

    if max_paths is not None and max_paths < num_paths:
        plot_paths = max_paths
    else:
        plot_paths = num_paths

    cmap_bg = plt.cm.get_cmap('cool', plot_paths)
    for i in range(plot_paths):
        plt.plot(time_steps, paths[i], color=cmap_bg(i/plot_paths),
                 linewidth=0.8, alpha=0.4)

    if highlight_paths > 0:
        highlight_indices = np.linspace(0, plot_paths - 1,
                                        min(highlight_paths, plot_paths),
                                        dtype=int)
        cmap = plt.cm.get_cmap('plasma', len(highlight_indices))
        for idx, i in enumerate(highlight_indices):
            plt.plot(time_steps, paths[i],
                     color=cmap(idx),
                     linewidth=2.5, alpha=1.0,
                     label=f"Path {i+1}")

    mean_path = np.mean(paths[:plot_paths], axis=0)
    median_path = np.median(paths[:plot_paths], axis=0)
    upper_path = np.quantile(paths[:plot_paths], 0.95, axis=0)
    lower_path = np.quantile(paths[:plot_paths], 0.05, axis=0)

    plt.plot(time_steps, mean_path, color='#FF5733', linewidth=3, label="Mean")
    plt.plot(time_steps, median_path, color='#33FFF5', linewidth=3, label="Median")
    plt.plot(time_steps, upper_path, color='#33FF57',
             linewidth=2, linestyle='--', label="95th Percentile")
    plt.plot(time_steps, lower_path, color='#F033FF',
             linewidth=2, linestyle='--', label="5th Percentile")

    plt.xlabel('Time Steps', fontsize=14, color='white')
    plt.ylabel('Price', fontsize=14, color='white')
    plt.title('Simulated Price Paths', fontsize=16, color='white')
    plt.grid(True, alpha=0.2, color='gray')

    legend = plt.legend(loc='best', facecolor='black', edgecolor='gray')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='none')

    if show_plot:
        plt.show()

    plt.close()
    print("Plot created successfully!")

def main():
    parser = argparse.ArgumentParser(
        description='Plot price paths from .npy file')
    parser.add_argument('--input', type=str, default='data/paths.npy',
                        help='Input .npy file containing price paths')
    parser.add_argument('--output', type=str, default='price_paths_plot.png',
                        help='Output image file')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot in addition to saving it')
    parser.add_argument('--max-paths', type=int, default=None,
                        help='Maximum number of paths to plot')
    parser.add_argument('--highlight', type=int, default=5,
                        help='Number of paths to highlight')

    args = parser.parse_args()

    plot_price_paths(
        npy_file=args.input,
        output_file=args.output,
        show_plot=args.show,
        max_paths=args.max_paths,
        highlight_paths=args.highlight
    )

if __name__ == "__main__":
    main()
