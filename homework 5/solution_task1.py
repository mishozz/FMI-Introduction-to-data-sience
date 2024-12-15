import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_REGIONS = [
    "Americas",
    "Asia Pacific",
    "Europe and Central Asia",
    "Middle East and North Africa",
    "Sub-Saharan Africa"
]

def init_data(corruption_df, gdp_happiness_df):
    corruption_long = pd.melt(
        corruption_df, 
        id_vars=['Country', 'Country Code', 'Region'], 
        value_vars=['CPI 2014 Score', 'CPI 2015 Score', 'CPI 2016 Score'],
        var_name='Year', 
        value_name='Corruption_Score'
    )
    corruption_long['Year'] = corruption_long['Year'].str.extract(r'(\d+)').astype(int)
    corruption_long = corruption_long.rename(columns={
        'Country': 'Entity', 
        'Country Code': 'Code'
    })

    # Preprocess GDP and Happiness Data
    gdp_happiness_filtered = gdp_happiness_df[
        (gdp_happiness_df['Year'].isin([2014, 2015, 2016])) & 
        (gdp_happiness_df['Cantril ladder score'].notna()) &
        (gdp_happiness_df['GDP per capita, PPP (constant 2017 international $)'].notna())
    ]
    gdp_happiness_filtered = gdp_happiness_filtered.rename(columns={
        'Cantril ladder score': 'Happiness_Score',
        'GDP per capita, PPP (constant 2017 international $)': 'GDP_Per_Capita'
    })

    # Merge datasets
    return pd.merge(
        gdp_happiness_filtered, 
        corruption_long, 
        on=['Entity', 'Code', 'Year'], 
        how='inner'
    )

def analyze_data(df):
    plot_heatmap(df)
    scatter_plot(df)
    plot_regressions(df)

def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['Happiness_Score', 'Corruption_Score', 'GDP_Per_Capita']].corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0, 
        square=True
    )
    plt.title('Correlation between Happiness, Corruption, and GDP', fontsize=15)
    plt.tight_layout()
    plt.show()

def scatter_plot(df):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['Corruption_Score'], 
        df['Happiness_Score'], 
        c=df['GDP_Per_Capita'], 
        cmap='viridis', 
        alpha=0.7
    )
    plt.colorbar(scatter, label='GDP Per Capita')
    plt.title('Happiness vs Corruption Perception (2014-2016)', fontsize=15)
    plt.xlabel('Corruption Perception Index Score', fontsize=12)
    plt.ylabel('Happiness Ladder Score', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_regressions(df):
    plt.figure(figsize=(15, 5))    
    plt.subplot(1, 3, 1)
    sns.regplot(
        x='Corruption_Score', 
        y='Happiness_Score', 
        data=df,
        scatter_kws={'alpha':0.5},
        line_kws={'color': 'red'}
    )
    plt.title('Regression: Corruption vs Happiness')
    
    plt.subplot(1, 3, 2)
    sns.regplot(
        x='GDP_Per_Capita', 
        y='Happiness_Score', 
        data=df,
        scatter_kws={'alpha':0.5},
        line_kws={'color': 'green'}
    )
    plt.title('Regression: GDP vs Happiness')

    plt.subplot(1, 3, 3)
    sns.regplot(
        x='Corruption_Score', 
        y='GDP_Per_Capita', 
        data=df,
        scatter_kws={'alpha':0.5},
        line_kws={'color': 'blue'}
    )
    plt.title('Regression: Corruption vs GDP')
    plt.tight_layout()
    plt.show()


def analyze_data_per_regions(df):
    regional_data = df[df['Region'].isin(TARGET_REGIONS)]

    region_analysis = regional_data.groupby('Region').agg({
        'Happiness_Score': 'mean',
        'Corruption_Score': 'mean',
        'GDP_Per_Capita': 'mean'
    }).reset_index()

    plot_bars(
        data=region_analysis,
        x='Region',
        y='Happiness_Score',
        palette='viridis',
        title='Average Happiness Score by Region',
        ylabel='Happiness Score'
    )

    plot_bars(
        data=region_analysis,
        x='Region',
        y='Corruption_Score',
        palette='coolwarm',
        title='Average Corruption Score by Region',
        ylabel='Corruption Score'
    )

    plot_bars(
        data=region_analysis,
        x='Region',
        y='GDP_Per_Capita',
        palette='mako',
        title='Average GDP by Region',
        ylabel='GDP_Per_Capita'
    )

    
def plot_bars(data, x, y, palette, title, ylabel):
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=data,
        x=x,
        y=y,
        palette=palette,
        hue=x,
        legend=False
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=30, fontsize=12, ha='right')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(x, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    corruption_df = pd.read_csv("corruption.csv")
    gdp_vs_happiness_df = pd.read_csv("gdp-vs-happiness.csv")
    data = init_data(corruption_df, gdp_vs_happiness_df)
    analyze_data(data)
    analyze_data_per_regions(data)
