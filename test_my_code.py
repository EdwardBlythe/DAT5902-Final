import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from Final_Project_Code import load_data, remove_current_year, fit_linear_model, plot_data  # Replace `your_module` with the actual module name

class TestDataProcessing(unittest.TestCase):
    
    

    # ensures the program reacts correctly to being provided invalid file-paths
    def test_read_data(self):
        
        with self.assertRaises(FileNotFoundError):
            load_data('imaginary_file.csv')

    # ensures current year is removed correctly
    def test_remove_current_year(self):
        # Creating a mock dataframe
        df = pd.DataFrame({
            'Year': [2021, 2022, 2023, 2024, 2025],
            'Annual number of fires': [10, 15, 20, 15, 10],
            'Annual area burnt per wildfire': [100, 150, 200, 150, 10]
        })
        processed_df = remove_current_year(df)
        
        self.assertNotIn(pd.to_datetime('today').year, processed_df['Year'].values)


    #Ensures read data contains expected columns
    def test_contains_expected_columns(self):
        df = pd.read_csv("annual-area-burnt-per-wildfire-vs-number-of-fires.csv")
        expected_values = ['Entity','Code','Year','Annual number of fires','Annual area burnt per wildfire','World regions according to OWID']
        self.assertEqual(list(df.columns), expected_values)


    @patch('matplotlib.pyplot.show')
    def test_plot_data(self, mock_show):
        # set dummy data to plot
        x = np.array([1, 2, 3])
        y = np.array([1, 3, 2])
        y_pred = np.array([2, 2, 2])
        
        plot_data(x, y, y_pred, 'Test Title', 'Year', 'Burnt Area')
        
        # Check if show() is called in the above call
        mock_show.assert_called_once()

        
if __name__ == '__main__':
    unittest.main()
