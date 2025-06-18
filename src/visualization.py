import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_ratio(df, group_col):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=group_col, 
        y='LossRatio', 
        data=df.groupby(group_col)['LossRatio'].mean().reset_index()
    )
    plt.title(f'Loss Ratio by {group_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_risk_distribution(df, col, bins=20):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=bins, kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    return plt.gcf()