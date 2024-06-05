import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    # Load the CSV file into a DataFrame
    return pd.read_csv(file_path, header=None, names=['value', 'file1', 'file2'])

def plot_boxplot(dataframes, file_names, output_fil, names):
    # Concatenate all DataFrames into one with an additional 'file' column
    all_data = pd.concat([df.assign(file=name) for df, name in zip(dataframes, names)], ignore_index=True)
    
    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='file', y='value', data=all_data)
    
    # Set plot labels and title
    plt.xlabel('Artist')
    plt.ylabel('Distance')
    plt.title('Perceptual Distance between Images')
    plt.xticks(rotation=45)
    
    # Save plot to file
    plt.tight_layout()
    plt.savefig(output_file)

if __name__ == "__main__":
    # List of file paths
    file_paths = ['data/similar_Albrecht_Dürer.txt', 'data/similar_Camille_Pissarro.txt', 'data/similar_Vincent_van_Gogh.txt']
    names = ['Albrecht Dürer', 'Camille Pissarro', 'Vincent van Gogh']
    
    # Load data from each file
    dataframes = [load_data(file_path) for file_path in file_paths]
    
    # Output file path
    output_file = 'runs/figures/boxplot.png'
    
    # Plot the boxplot and save it to a file
    plot_boxplot(dataframes, file_paths, output_file, names)
    
    print(f'Box plot saved to {output_file}')
