import pandas as pd

class TechnicalIndicators:
    def __init__(self, data):
        """
        Initialize with historical stock data.
        :param data: A pandas DataFrame containing 'Open', 'High', 'Low', 'Price' columns.
        """
        self.data = data
        self.price_column = 'Price'  # Assuming 'Price' is equivalent to the close price
        
    def moving_average(self, window=20):
        """
        Calculate the moving average.
        :param window: The window size for the moving average (default is 20).
        :return: A pandas Series with the moving average.
        """
        return self.data[self.price_column].rolling(window=window).mean()
    
    def exponential_moving_average(self, window=20):
        """
        Calculate the exponential moving average.
        :param window: The window size for the EMA (default is 20).
        :return: A pandas Series with the exponential moving average.
        """
        return self.data[self.price_column].ewm(span=window, adjust=False).mean()
    
    def relative_strength_index(self, window=14):
        """
        Calculate the Relative Strength Index (RSI).
        :param window: The window size for the RSI (default is 14).
        :return: A pandas Series with the RSI.
        """
        delta = self.data[self.price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, short_window=12, long_window=26, signal_window=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        :param short_window: The short window size (default is 12).
        :param long_window: The long window size (default is 26).
        :param signal_window: The signal line window size (default is 9).
        :return: A pandas DataFrame with MACD and Signal line.
        """
        ema_short = self.data[self.price_column].ewm(span=short_window, adjust=False).mean()
        ema_long = self.data[self.price_column].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal': signal})
    
    def bollinger_bands(self, window=20, num_std=2):
        """
        Calculate the Bollinger Bands.
        :param window: The window size for the moving average (default is 20).
        :param num_std: The number of standard deviations for the bands (default is 2).
        :return: A pandas DataFrame with Upper Band, Lower Band, and Moving Average.
        """
        rolling_mean = self.data[self.price_column].rolling(window=window).mean()
        rolling_std = self.data[self.price_column].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return pd.DataFrame({
            'Upper Band': upper_band,
            'Lower Band': lower_band,
            'Moving Average': rolling_mean
        })
    
    def get_all_indicators(self):
        """
        Calculate all the technical indicators and add them to the data.
        :return: A pandas DataFrame with all indicators added as columns.
        """
        self.data['Moving Average'] = self.moving_average()
        self.data['Exponential Moving Average'] = self.exponential_moving_average()
        self.data['RSI'] = self.relative_strength_index()
        macd_data = self.macd()
        self.data['MACD'] = macd_data['MACD']
        self.data['MACD Signal'] = macd_data['Signal']
        bollinger_data = self.bollinger_bands()
        self.data['Upper Band'] = bollinger_data['Upper Band']
        self.data['Lower Band'] = bollinger_data['Lower Band']
        self.data['Bollinger Moving Average'] = bollinger_data['Moving Average']
        return self.data