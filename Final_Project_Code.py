import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def remove_current_year(firesDf):
    """Remove current year data and calculate Total Area Burnt."""
    current_year = pd.to_datetime('today').year
    firesDf = firesDf[firesDf['Year'] != current_year]
    firesDf['Total Area Burnt'] = firesDf['Annual number of fires'] * firesDf['Annual area burnt per wildfire']
    return firesDf

def fit_linear_model(X, y):
    """Fit a linear regression model and return predictions."""
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X)

def plot_data(x, y, y_pred, title, xlabel, ylabel):
    """Plot data with a fitted linear model."""
    plt.figure(figsize=(10,6))
    plt.plot(x, y, marker='X', label='Actual Data')  # Plot the actual data
    plt.plot(x, y_pred, color='red', label='Fitted Line')  # Plot the fitted line
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def main(filepath):
    """Main function to load data, process it, fit model, and plot results."""
    # Load data
    firesDf = load_data(filepath)
    
    # Process data
    firesDf = remove_current_year(firesDf)
    
    # Plot Total Area Burnt
    total_area_burnt_by_year = firesDf.groupby('Year')['Total Area Burnt'].sum()
    X = total_area_burnt_by_year.index.values.reshape(-1, 1)
    y = total_area_burnt_by_year.values
    y_pred = fit_linear_model(X, y)
    plot_data(total_area_burnt_by_year.index, total_area_burnt_by_year.values, y_pred, 
              'Total Area Burnt Each Year', 'Year', 'Total Area Burnt (Hectares)')
    
    # Plot Average Area Burnt per Wildfire
    average_area_burnt_per_wildfire = firesDf.groupby('Year')['Annual area burnt per wildfire'].mean()
    X = average_area_burnt_per_wildfire.index.values.reshape(-1, 1)
    y = average_area_burnt_per_wildfire.values
    y_pred = fit_linear_model(X, y)
    plot_data(average_area_burnt_per_wildfire.index, average_area_burnt_per_wildfire.values, y_pred, 
              'Average Area Burnt Per Wildfire Each Year', 'Year', 'Average Area Burnt (Hectares)')
    
    # Plot Total Wildfires
    total_wildfires = firesDf.groupby('Year')['Annual number of fires'].sum()
    X = total_wildfires.index.values.reshape(-1, 1)
    y = total_wildfires.values
    y_pred = fit_linear_model(X, y)
    plot_data(total_wildfires.index, total_wildfires.values, y_pred, 
              'Total Wildfires Each Year', 'Year', 'Total Wildfires')

if __name__ == '__main__':
    filepath = 'annual-area-burnt-per-wildfire-vs-number-of-fires.csv'
    main(filepath)