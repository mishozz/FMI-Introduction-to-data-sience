import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

VILLAS_CSV = 'villas_metadata.csv'
LIVING_AREA = 'Living_area'
SELLING_PRICE = 'Selling_price'
LIVING_AREA_PREDICTIONS = [100, 150, 200]

def init_data():
    data = pd.read_csv(VILLAS_CSV)
    data = data[[LIVING_AREA, SELLING_PRICE]].dropna()
    living_areas = data[[LIVING_AREA]]
    selling_prices = data[SELLING_PRICE]
    return data, living_areas, selling_prices

def plot_data(X, y, model):
    slope = model.coef_[0]
    intercept = model.intercept_

    living_areas = pd.DataFrame({LIVING_AREA: LIVING_AREA_PREDICTIONS})
    predicted_prices = model.predict(living_areas)
    print(f"Predicted prices for 100 m², 150 m², and 200 m²: {predicted_prices}")

    y_pred = model.predict(X)
    residuals = y - y_pred
    
    _, (pl1, pl2) = plt.subplots(1, 2, figsize=(15, 6))

    pl1.scatter(data[LIVING_AREA], data[SELLING_PRICE], color='blue', label='Data Points', alpha=0.6)
    pl1.axline(xy1=(0, intercept), slope=slope, color='red', label=f'$y = {slope:.1f}x {intercept:+.1f}$')
    pl1.set_xlabel('Living Area (m²)')
    pl1.set_ylabel('Selling Price')
    pl1.set_title('Living Area vs Selling Price with Regression Line')
    pl1.ticklabel_format(axis='y', style='plain')
    pl1.legend()

    # Plot the second graph: Residual Plot
    pl2.scatter(X, residuals, color='blue', alpha=0.6)
    pl2.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    pl2.set_xlabel('Living Area (m²)')
    pl2.set_ylabel('Residuals (Actual - Predicted Price)')
    pl2.set_title('Residual Plot')
    pl2.ticklabel_format(axis='y', style='plain')
    pl2.legend()

    plt.show()


if __name__ == '__main__':
    data, X, y = init_data()
    model = LinearRegression()
    model.fit(X, y)
    plot_data(X, y, model)
