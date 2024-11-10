import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LIFE_EXPENTANCY_CSV = 'life-expectancy.csv'
GDP_PER_CAPITA_CSV = 'gdp-per-capita-worldbank.csv'
LIFE_EXPENTANCY = 'Life_Expectancy'
GDP_PER_CAPITA = 'GDP_per_capita'
ENTITY = 'Entity'
CODE = 'Code'
YEAR = 'Year'
YEARS_TO_ANALYZE = [1990, 1998, 2008, 2018, 2019, 2020, 2021]

def init_data():
    life_expectancy = pd.read_csv(LIFE_EXPENTANCY_CSV)
    gdp = pd.read_csv(GDP_PER_CAPITA_CSV)

    life_expectancy = life_expectancy.rename(columns={
        'Period life expectancy at birth - Sex: all - Age: 0': LIFE_EXPENTANCY
    })
    gdp = gdp.rename(columns={
        'GDP per capita, PPP (constant 2017 international $)': GDP_PER_CAPITA
    })

    return pd.merge(life_expectancy, gdp, 
                        on=[ENTITY, CODE, YEAR],
                        how='inner')
    

def filter_data_by_year(data, year):
    filtered_data = data[data[YEAR] == year]
    return filtered_data.dropna()

def format_func(value, tick_number):
    return f'{int(value):,}'

def plot_data_by_year(data, year):
    plt.figure(figsize=(12, 8))
    plt.scatter(data[GDP_PER_CAPITA], 
            data[LIFE_EXPENTANCY],
            alpha=0.5,
            c='blue',
            label='Countries')
    z = np.polyfit(data[GDP_PER_CAPITA], data[LIFE_EXPENTANCY], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(data[GDP_PER_CAPITA].min(),
                        data[GDP_PER_CAPITA].max(),
                        100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
            label=f'Trend line (y = {z[0]:.2f}x + {z[1]:.2f})')
    plt.title(f'GDP per Capita vs Life Expectancy {year} year', fontsize=14, pad=15)
    plt.xlabel('GDP per Capita in $', fontsize=12)
    plt.ylabel('Life Expectancy in years)', fontsize=12)

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    

def get_hight_expectancy_low_gdp_countries(data):
    high_life_exp_countries = get_high_end_countries_by_life_exp(data)
    low_gdp_countries = get_low_end_countries_by_gdp(data)

    return pd.merge(high_life_exp_countries, low_gdp_countries, how='inner', on=[ENTITY, CODE, YEAR, GDP_PER_CAPITA, LIFE_EXPENTANCY])

def get_high_gdp_low_life_exp_countries(data):
    high_gdp_countries = get_high_end_countries_by_gdp(data)
    low_life_exp_countries = get_low_end_countries_by_life_exp(data)

    return pd.merge(high_gdp_countries, low_life_exp_countries, how='inner', on=[ENTITY, CODE, YEAR, GDP_PER_CAPITA, LIFE_EXPENTANCY])


def get_high_end_countries_by_gdp(data):
    threshold = calculate_high_gdp_threshold(data[GDP_PER_CAPITA])
    high_life_exp_countries = data[data[GDP_PER_CAPITA] > threshold].copy()

    return high_life_exp_countries.loc[high_life_exp_countries.groupby('Entity')[GDP_PER_CAPITA].idxmax()]

def get_low_end_countries_by_gdp(data):
    threshold = calculate_low_gdp_threshold(data[GDP_PER_CAPITA])
    low_life_exp_countries = data[data[GDP_PER_CAPITA] < threshold].copy()

    return low_life_exp_countries.loc[low_life_exp_countries.groupby('Entity')[GDP_PER_CAPITA].idxmin()]

def get_high_end_countries_by_life_exp(data):
    threshold = calculate_high_life_exp_threshold(data[LIFE_EXPENTANCY])
    high_life_exp_countries = data[data[LIFE_EXPENTANCY] > threshold].copy()

    return high_life_exp_countries.loc[high_life_exp_countries.groupby('Entity')[LIFE_EXPENTANCY].idxmax()]

def get_low_end_countries_by_life_exp(data):
    threshold = calculate_low_life_exp_threshold(data[LIFE_EXPENTANCY])
    low_life_exp_countries = data[data[LIFE_EXPENTANCY] < threshold].copy()

    return low_life_exp_countries.loc[low_life_exp_countries.groupby('Entity')[LIFE_EXPENTANCY].idxmin()]

def get_countries_with_higher_than_one_std(data, metric_key):
    threshold = calculate_threshold_higher_than_one_std(data[metric_key])
    return data[data[metric_key] > threshold]

def calculate_threshold_higher_than_one_std(values):
    return values.mean() + values.std()

def calculate_high_life_exp_threshold(values):
    return values.quantile(0.85)

def calculate_low_life_exp_threshold(values):
    return values.quantile(0.35)

def calculate_high_gdp_threshold(values):
    return values.median()*1.5

def calculate_low_gdp_threshold(values):
    return values.median()*0.6

def run_statistics_for(years):
    data = init_data()
    
    for year in years:
        filtered_data = filter_data_by_year(data, year)
    
        # b)
        higher_than_one_std_conties = get_countries_with_higher_than_one_std(filtered_data, LIFE_EXPENTANCY)
        print(f'\nLife expectancy threshold higher than 1 SD above mean for {year}:')
        print(higher_than_one_std_conties[[ENTITY, LIFE_EXPENTANCY]].to_string(index=False))

        # c)
        print(f'\nCountries with high life expectancy and low GDP per capita for {year}:')
        print(get_hight_expectancy_low_gdp_countries(filtered_data).to_string(index=False))

        # d)
        high_gdp_low_life_exp_countries = get_high_gdp_low_life_exp_countries(filtered_data)
        print(f'\nCountries with high GDP per capita and low life expectancy for {year}:')
        print(high_gdp_low_life_exp_countries.to_string(index=False))
        correlation = np.corrcoef(filtered_data[GDP_PER_CAPITA], 
                                filtered_data[LIFE_EXPENTANCY])[0,1]
        print("\nCorrelation coefficient between GDP per capita and Life Expectancy:", 
            round(correlation, 3))

        # a)
        plot_data_by_year(filtered_data, year)


    print('\nPlotting...')
    plt.show()

run_statistics_for(YEARS_TO_ANALYZE)
