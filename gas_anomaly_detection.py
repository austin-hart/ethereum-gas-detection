# Import necessary libraries
from web3 import Web3  # Web3: A Python library for connecting to and performing operations on Ethereum-based blockchains.
import pandas as pd  # pandas: A data manipulation and analysis library.
import matplotlib.pyplot as plt  # matplotlib.pyplot: A plotting library for creating static, interactive, and animated visualizations in Python.
import scipy.stats as stats  # scipy.stats: A module for a large number of probability distributions and statistical functions.
from sklearn.ensemble import IsolationForest  # IsolationForest: An algorithm for anomaly detection.

# Constants
ALCHEMY_API_KEY = "NY7FmEw1VgeB5J9nASAUNW3XwpX9yFGr"  # Alchemy API Key: Unique identifier for accessing the Alchemy API services.
ALCHEMY_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"  # Alchemy URL: Endpoint for connecting to the Ethereum blockchain via Alchemy.
NUM_BLOCKS = 1024  # Number of blocks: Determines how many blocks of Ethereum data we will retrieve and analyze.

# Establish a connection to Ethereum blockchain using Web3
web3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
assert web3.is_connected(), "Failed to connect to Ethereum blockchain. Check the Alchemy API Key and URL."

# Retrieve historical data for the Ethereum blockchain
# 'NUM_BLOCKS' specifies how many blocks worth of data we want to analyze.
# 'latest' refers to the most recent blocks in the blockchain.
# [25, 75] are percentiles for calculating the fee distribution within each block.
hist = web3.eth.fee_history(NUM_BLOCKS, 'latest', [25,75])

def format_output(data, num_blocks):
    """
    Formats and structures historical blockchain data for analysis.

    Args:
    - data (dict): Raw blockchain data. Contains detailed information about each block.
    - num_blocks (int): Number of blocks to process.

    Returns:
    - List[dict]: A list of dictionaries, each representing formatted data for a single block.
    """
    formatted_data = []
    for i in range(num_blocks):
        # Process each block to extract and format relevant information
        block = {
            'blockNumber': int(data['oldestBlock']) + i,  # Block number: Sequential identifier of the block in the blockchain.
            'reward': [round(int(r) / 10**9) for r in data['reward'][i]],  # Reward: The reward for each block, scaled to Gwei for readability.
            'baseFeePerGas': round(int(data['baseFeePerGas'][i]) / 10**9),  # Base Fee Per Gas: The minimum gas price for inclusion in this block.
            'gasUsedRatio': data['gasUsedRatio'][i],  # Gas Used Ratio: The ratio of gas used in the block to the total gas limit.
        }
        formatted_data.append(block)
    return formatted_data

# Convert the formatted blockchain data into a DataFrame for easier manipulation and analysis
df = pd.DataFrame(format_output(hist, NUM_BLOCKS))

# DataFrame 'df' now contains structured data on Ethereum blocks, including block number, rewards, base fee, and gas used ratio.

# Display the latest 5 blocks from the data to get a glimpse of the most recent Ethereum blockchain activity.
print('5 most recent blocks:')
print(df.tail())
print()

# Perform and display descriptive statistical analysis on Ethereum gas prices.
# This includes mean, median, standard deviation, and quartiles, which provide insights into the distribution and central tendency of gas prices.
print("Descriptive Analysis of Ethereum Gas Prices from last 1024 blocks (in Gwei):")
descriptive_stats = df['baseFeePerGas'].describe(percentiles=[0.25, 0.5, 0.75])
print(descriptive_stats.to_string())
print()

# Calculate and print the correlation coefficient between gas used ratio and gas price.
correlation = df['baseFeePerGas'].corr(df['gasUsedRatio'])
print("Correlation between gas used ratio and gas price (in Gwei):", correlation)
print()

# Calculate and print skewness and kurtosis of the gas distribution.
# Skewness measures the asymmetry of the data distribution. Kurtosis indicates the 'tailedness' of the distribution.
print('Skewness and Kurtosis of Gas Prices:')
print(f"Skewness: {df['baseFeePerGas'].skew()}")
print(f"Kurtosis: {df['baseFeePerGas'].kurtosis()}")
print()

# Initialize the Isolation Forest algorithm for anomaly detection.
# contamination=0.01 indicates the proportion of outliers in the data set is expected to be around 1%.
#1% percent was chosen because there is extreme volatility in the ethereum network
clf = IsolationForest(contamination=0.01)
df['anomaly'] = clf.fit_predict(df[['baseFeePerGas']])  # Apply the algorithm to the 'baseFeePerGas' column.
anomalous_blocks = df[df['anomaly'] == -1]['blockNumber']  # Identify blocks that are considered anomalies.

# Display blocks identified as having anomalous gas activity, providing insights into unusual blockchain events.
print('Blocks with anomalous gas activity: ')
print(anomalous_blocks.to_string(index=False))
print()

# Visualization Section
fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # Create a figure with 4 subplots.

# Scatter plot: Visualizes the relationship between Base Fee Per Gas and Gas Used Ratio.
axs[0].scatter(df['baseFeePerGas'], df['gasUsedRatio'])
axs[0].set_xlabel('Base Fee Per Gas (Gwei)')
axs[0].set_ylabel('Gas Used Ratio')
axs[0].set_title('Scatter Plot of Gas Used Ratio vs. Base Fee Per Gas')

# Histogram: Shows the frequency distribution of the Base Fee Per Gas.
axs[1].hist(df['baseFeePerGas'], bins=20, edgecolor='black')
axs[1].set_xlabel('Base Fee Per Gas (Gwei)')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Base Fee Per Gas')

# Q-Q Plot: Assesses if the Base Fee Per Gas data follows a normal distribution.
stats.probplot(df['baseFeePerGas'], dist="norm", plot=axs[2])
axs[2].set_title('Q-Q Plot for Base Fee Per Gas')

# Anomaly Detection Visualization: Differentiates between normal data points and anomalies in the Base Fee Per Gas.
axs[3].scatter(df.index, df['baseFeePerGas'], color='blue', label='Normal')
axs[3].scatter(df[df['anomaly'] == -1].index, df[df['anomaly'] == -1]['baseFeePerGas'], color='red', label='Anomaly')
axs[3].set_xlabel('Block Number')
axs[3].set_ylabel('Base Fee Per Gas (Gwei)')
axs[3].set_title('Anomaly Detection in Gas Prices')
axs[3].legend()

plt.tight_layout()  # Adjusts the layout for a neat and readable presentation.
plt.show()  # Displays the figure with all subplots.
