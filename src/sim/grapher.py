import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_price_paths(csv_file="price_paths.csv", output_file="price_paths_plot.png", 
                     show_plot=False, max_paths=None, highlight_paths=5):
    """
    Plot price paths from a CSV file and save the visualization.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing price paths
    output_file : str
        Path to save the output plot image
    show_plot : bool
        Whether to display the plot in addition to saving it
    max_paths : int or None
        Maximum number of paths to plot (None for all)
    highlight_paths : int
        Number of paths to highlight with distinct colors
    """
    # Read price paths
    print(f"Reading price paths from {csv_file}...")
    try:
        paths_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Get dimensions
    num_paths, num_steps = paths_df.shape
    print(f"Found {num_paths} paths with {num_steps} time steps each")
    
    # Create x-axis (time steps)
    time_steps = np.arange(num_steps)
    
    # Set dark style for the plot
    plt.style.use('dark_background')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Limit paths if requested
    if max_paths is not None and max_paths < num_paths:
        plot_paths = min(max_paths, num_paths)
    else:
        plot_paths = num_paths
    
    # Plot background paths with more vivid colors
    cmap_bg = plt.cm.get_cmap('cool', plot_paths)
    for i in range(plot_paths):
        plt.plot(time_steps, paths_df.iloc[i], color=cmap_bg(i/plot_paths), 
                 linewidth=0.8, alpha=0.4)
    
    # Highlight a few select paths with distinct colors
    if highlight_paths > 0:
        # Choose evenly spaced paths to highlight
        highlight_indices = np.linspace(0, plot_paths-1, min(highlight_paths, plot_paths), dtype=int)
        
        # Use a colormap for the highlighted paths
        cmap = plt.cm.get_cmap('plasma', len(highlight_indices))
        
        for idx, i in enumerate(highlight_indices):
            plt.plot(time_steps, paths_df.iloc[i], color=cmap(idx), linewidth=2.5, alpha=1.0, 
                     label=f"Path {i+1}")
    
    # Calculate statistics across all paths
    mean_path = paths_df.iloc[:plot_paths].mean(axis=0)
    median_path = paths_df.iloc[:plot_paths].median(axis=0)
    upper_path = paths_df.iloc[:plot_paths].quantile(0.95, axis=0)
    lower_path = paths_df.iloc[:plot_paths].quantile(0.05, axis=0)
    
    # Plot statistical lines with more vivid colors against dark background
    plt.plot(time_steps, mean_path, color='#FF5733', linewidth=3, label="Mean")
    plt.plot(time_steps, median_path, color='#33FFF5', linewidth=3, label="Median")
    plt.plot(time_steps, upper_path, color='#33FF57', linewidth=2, linestyle='--', label="95th Percentile")
    plt.plot(time_steps, lower_path, color='#F033FF', linewidth=2, linestyle='--', label="5th Percentile")
    
    # Add labels and title
    plt.xlabel('Time Steps', fontsize=14, color='white')
    plt.ylabel('Price', fontsize=14, color='white')
    plt.title('Simulated Price Paths', fontsize=16, color='white')
    plt.grid(True, alpha=0.2, color='gray')
    
    # Add legend with transparent background
    legend = plt.legend(loc='best', facecolor='black', edgecolor='gray')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with transparent=False to ensure dark background is preserved
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    
    if show_plot:
        plt.show()
    
    plt.close()
    print("Plot created successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot price paths from CSV file')
    parser.add_argument('--input', type=str, default='price_paths.csv',
                        help='Input CSV file containing price paths (default: price_paths.csv)')
    parser.add_argument('--output', type=str, default='price_paths_plot.png',
                        help='Output image file (default: price_paths_plot.png)')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot in addition to saving it')
    parser.add_argument('--max-paths', type=int, default=None,
                        help='Maximum number of paths to plot (default: all)')
    parser.add_argument('--highlight', type=int, default=5,
                        help='Number of paths to highlight (default: 5)')
    
    args = parser.parse_args()
    
    # Call the plotting function
    plot_price_paths(
        csv_file=args.input,
        output_file=args.output,
        show_plot=args.show,
        max_paths=args.max_paths,
        highlight_paths=args.highlight
    )

if __name__ == "__main__":
    main()
