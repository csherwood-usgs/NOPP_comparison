# nheri_funcs
import pandas as pd
import os

def nheri_rwg_read_csv(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Combine the 'Date' and 'Time (GMT)' columns into a single 'time' column
    df['time'] = pd.to_datetime(df['date'] + ' ' + df[' time'])
    
    # Drop the original 'Date' and 'Time (GMT)' columns and other unnecessary columns
    # (note the annoying leading space for column names)
    df.drop(columns=["date", " time"], inplace=True)

    df.rename(columns={' pressure': 'pressure'}, inplace=True)
    
    # Initialize variables to track the current second and the number of records for that second
    current_second = None
    records_in_current_second = 0
    timestamps = []
    
    # Set the base time as the time of the first record
    base_time = df['time'].iloc[0]  # Base time is the timestamp of the first record
    
    # Iterate through the records to assign fractional timestamps
    for i, row in df.iterrows():
        record_time = row['time']
        
        # If the second has changed, process the previous second
        if current_second is None or (record_time - current_second).seconds >= 1:
            # If we're already processing a second, assign fractional timestamps to it
            if current_second is not None:
                time_increment = 1 / records_in_current_second  # Fraction of second
                for j in range(records_in_current_second):
                    fractional_time = base_time + pd.Timedelta(seconds=j * time_increment)
                    timestamps.append(fractional_time)
            
            # Update current_second and reset the records counter for the new second
            current_second = record_time
            base_time = record_time  # Set the base time for the new second
            records_in_current_second = 0
        
        # Increment the counter for records in the current second
        records_in_current_second += 1
    
    # After the loop, process the last second's records
    if current_second is not None:
        time_increment = 1 / records_in_current_second
        for j in range(records_in_current_second):
            fractional_time = base_time + pd.Timedelta(seconds=j * time_increment)
            timestamps.append(fractional_time)
    
    # Add the calculated timestamps to the dataframe
    df['timestamp'] = timestamps
    
    return df


def smooth_pressure_data(df, window=5):
    """
    Applies a running mean to the 'pressure' column of the dataframe,
    without introducing a lag.
    """
    dfs = df.copy()
    dfs['pressure'] = dfs['pressure'].rolling(window=window, center=True).mean()
    return dfs


def resample_data(df, freq='0.25s'):
    """
    Resamples the 'pressure' data to a new time series with a freq (default: 4 samples per second
    i.e., every 0.25 seconds) using linear interpolation in time.
    """
    dfc = df.copy()
    # Generate a new time index at 0.25 second intervals starting from the first timestamp
    new_time_index = pd.date_range(start=dfc['timestamp'].iloc[0], end=dfc['timestamp'].iloc[-1], freq=freq)
    
    # Create a new DataFrame with the new time index, reindexing the existing data
    df_resampled = dfc.set_index('timestamp').reindex(new_time_index)

    # Perform linear interpolation to fill in the NaN values for 'pressure'
    df_resampled['pressure'] = df_resampled['pressure'].interpolate(method='linear')

    # Reset the index, making 'timestamp' a column again
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'timestamp'}, inplace=True)
    
    return df_resampled


def read_met_data(filename):
    ''' 
    Read the NOS met data...keep pressure only
    Return an pandas dataframe
    
    First two lines of the .csv file look like this:
    "Date","Time (GMT)","Wind Speed (m/s)","Wind Dir (deg)","Wind Gust (m/s)","Air Temp (°C)","Baro (mb)","Humidity (%)","Visibility (km)"
    "2024/09/24","00:00","1.8","315","3.2","28.8","1011.5","-","-"
    '''
    
    df = pd.read_csv(filename)
    # Combine the 'Date' and 'Time (GMT)' columns into a single 'time' column
    df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time (GMT)'])    
    # Drop the original 'Date' and 'Time (GMT)' columns
    df.drop(columns=["Date", "Time (GMT)","Wind Speed (m/s)","Wind Dir (deg)","Wind Gust (m/s)","Air Temp (°C)","Humidity (%)","Visibility (km)"], inplace=True)
    # Rename the "Baro (mb)" column to "BP"
    df.rename(columns={'Baro (mb)': 'BP'}, inplace=True)
    return df