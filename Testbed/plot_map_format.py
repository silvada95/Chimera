import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Read the CSV file
df = pd.read_csv('testbed_results.csv')

def plot_combined(df):
    # Create vectors for each data entry's mAP50-95 values
    map_vtei = df[df['data'] == 'pedro_vtei.yaml']['mAP50-95'].values
    map_mdes = df[df['data'] == 'pedro_mdes.yaml']['mAP50-95'].values
    map_shist = df[df['data'] == 'pedro_shist.yaml']['mAP50-95'].values
    map_taf = df[df['data'] == 'pedro_taf.yaml']['mAP50-95'].values

    # Initialize counts
    count_vtei = np.sum((map_vtei > map_mdes) & (map_vtei > map_shist) & (map_vtei > map_taf))
    count_mdes = np.sum((map_mdes > map_vtei) & (map_mdes > map_shist) & (map_mdes > map_taf))
    count_shist = np.sum((map_shist > map_vtei) & (map_shist > map_mdes) & (map_shist > map_taf))
    count_taf = np.sum((map_taf > map_vtei) & (map_taf > map_mdes) & (map_taf > map_shist))
    
    print(count_vtei, count_mdes, count_shist, count_taf)
    
    # Prepare data for the bar plot
    counts = [count_vtei, count_mdes, count_shist, count_taf]
    labels = ['VTEI', 'MDES', 'SHIST', 'TAF']
    colors = ['#FFFF00', '#FF0000', '#0000FF', '#00FF00']  # Colors for each bar

    # Create a DataFrame to store counts for box plot alignments
    counts_df = pd.DataFrame({
        'data_label': labels,
        'counts': counts
    })

    # Prepare data for box plot using the original values
    data_entries = ["pedro_mdes.yaml", "pedro_vtei.yaml", "pedro_taf.yaml", "pedro_shist.yaml"]
    data_labels = {
        "pedro_mdes.yaml": "MDES",
        "pedro_vtei.yaml": "VTEI",
        "pedro_taf.yaml": "TAF",
        "pedro_shist.yaml": "SHIST"
    }

    # Filter the DataFrame to include only the relevant data for box plots
    filtered_df = df[df['data'].isin(data_entries)]
    filtered_df['data_label'] = filtered_df['data'].map(data_labels)

    # Create a combined figure for the bar and box plots with reduced dimensions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))  # Reduced size from (10, 12) to (5, 6)

    # Bar plot
    ax1.bar(labels, counts, color=colors)
    ax1.set_title('Highest mAP Cases', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Data Encoding', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(axis='y')

    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=12, fontweight='bold')
    
    # Set y-axis format to show 2 decimal places
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Box plot - Use predefined order of labels and ensure the same colors
    sns.boxplot(ax=ax2, data=filtered_df, x='data_label', y='mAP50-95', 
                order=labels, palette=colors)
    ax2.set_title('mAP by Data Encoding', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Data Encoding', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mAP', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    ax2.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=12, fontweight='bold')

    # Set y-axis format to show 2 decimal places for the box plot
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.show()

# Call the function to plot the combined plots
plot_combined(df)
