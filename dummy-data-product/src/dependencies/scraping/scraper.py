import geemap.eefolium as geemap
import ee
from geopy.geocoders import Nominatim
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize



def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df



def t_modis_to_celsius(t_modis):
    """Converts MODIS LST units to degrees Celsius."""
    t_celsius =  0.02*t_modis - 273.15
    return t_celsius

def scaper():
    dataset = 'MODIS/061/MOD11A1'
    #dataset = input("\n Enter the name of dataset (eg): ")
    print("\n Dataset we are using is ",dataset)
    
    i_date = input("\n Enter the initial date (Year-month-date)(eg. 2018-01-01 ) : ")
    f_date = input("\n Enter the final date (Year-month-date)(eg. 2022-01-01 ) : ")

    lat = float(input("\n Enter the latitude (eg. 40.6473 ) : "))
    lon = float(input("\n Enter the longitude (eg. -100.0231 ) : "))
    
    print("\n Enter Temprature Range: ")
    min_temp_celcius = float(input("\n Minimum Temprature (eg. -15 ): "))
    min_temp_MODIS = ((min_temp_celcius + 273.15)/0.02)
    max_temp_celcius = float(input("\n Maximum Temprature (eg. 35.45 ): "))
    max_temp_MODIS = ((max_temp_celcius + 273.15)/0.02)

    lst = ee.ImageCollection(dataset)
    lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)
    us_poi = ee.Geometry.Point(lon, lat)
    scale = 1000  # scale in meters ( resolution)

    # Calculate and print the mean value of the LST collection at the point.
    lst_us_point = lst.mean().sample(us_poi, scale).first().get('LST_Day_1km').getInfo()
    avg_temp = round(lst_us_point*0.02 -273.15, 2)
    print('\n Average daytime LST at the point:', avg_temp, '°C')
    lst_us_poi = lst.getRegion(us_poi, scale).getInfo()
    
    lst_df_us = ee_array_to_df(lst_us_poi,['LST_Day_1km'])


    # Apply the function to get temperature in celsius.
    lst_df_us['LST_Day_1km_celsius'] = lst_df_us['LST_Day_1km'].apply(t_modis_to_celsius)
    lst_df_us = lst_df_us.drop(['LST_Day_1km'], axis = 1)
    
    # saving dataframe into exel file
    lst_df_us.to_excel(r'output\LST_US.xlsx')
    
    lst_df_us.to_csv(r'output\mesh_timeseries.csv',index= False)
    
    # creating data_info file
    
    geolocator = Nominatim(user_agent="geoapiExercises")
    Latitude = str(lat)
    Longitude = str(lon)
  
    location = geolocator.reverse(Latitude+","+Longitude)
    address = location.raw['address']
    country = address.get('country', '')
    
    print("\n Country as per cordinates: ",country)
    
    data_info = {'dataset_name':[dataset],
                 'country_name':[country],
                 'avg_temp_celcius':[avg_temp],
                 'in_date':[i_date],
                 'f_date':[f_date],
                 'latitude':[Latitude],
                 'longitude':[Longitude],
                 'min_temp':[min_temp_celcius],
                 'max_temp':[max_temp_celcius],
                 'min_temp_MODIS':[min_temp_MODIS],
                 'max_temp_MODIS':[max_temp_MODIS]
                 }
    
    data_info = pd.DataFrame(data_info)
    
    data_info.to_excel(r'output\data_info.xlsx',index= False)
    
def images_scraper():
    lst_df_us = pd.read_csv(r"output\mesh_timeseries.csv")
    
    # Fitting curves.
    ## First, extract x values (times) from the dfs.
    x_data_u = np.asanyarray(lst_df_us['time'].apply(float))  # time

    ## Secondly, extract y values (LST) from the dfs.
    y_data_u = np.asanyarray(lst_df_us['LST_Day_1km_celsius'].apply(float))  # Temperature in degree celcius

    ## Then, define the fitting function with parameters.
    def fit_func(t, lst0, delta_lst, tau, phi):
        return lst0 + (delta_lst/2)*np.sin(2*np.pi*t/tau + phi)

    ## Optimize the parameters using a good start p0.
    lst0 = 20
    delta_lst = 40
    tau = 365*24*3600*1000   # milliseconds in a year
    phi = 2*np.pi*4*30.5*3600*1000/tau  # offset regarding when we expect LST(t)=LST0

    params_u, params_covariance_u = optimize.curve_fit(
        fit_func, x_data_u, y_data_u, p0=[lst0, delta_lst, tau, phi])

    # Subplots.
    fig, ax = plt.subplots(figsize=(14, 6))

    # Add scatter plots.
    ax.scatter(lst_df_us['datetime'], lst_df_us['LST_Day_1km_celsius'],
            c='black', alpha=0.2, label='Data')


    # Add fitting curves.
    ax.plot(lst_df_us['datetime'],
            fit_func(x_data_u, params_u[0], params_u[1], params_u[2], params_u[3]),
            label='(fitted)', color='black', lw=2.5)


    # Add some parameters.
    ax.set_title('Daytime Land Surface Temperature at given cordinates', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Temperature [C]', fontsize=14)
    ax.set_ylim(-0, 40)
    ax.grid(lw=0.2)
    ax.legend(fontsize=14, loc='lower right')

    
    plt.savefig(r'output\graph.png', bbox_inches='tight')
    
def map_scraper():
    data_info = pd.read_excel(r"output\data_info.xlsx")

    data_info = pd.DataFrame(data=data_info, dtype=object)
    
    # Map with Land Surface Temperature 

    dataset = ee.ImageCollection(data_info.dataset_name[0]) \
    .filter(ee.Filter.date(data_info.in_date[0], data_info.f_date[0]))

    dataset_region = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    US_Border = dataset_region.filter(ee.Filter.eq('country_na',data_info.country_name[0]))


    landSurfaceTemperature = dataset.select('LST_Day_1km')
    landSurfaceTemperatureVis = {
        'min': data_info.min_temp_MODIS[0],
        'max': data_info.max_temp_MODIS[0], 
        'palette':['040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                'ff0000', 'de0101', 'c21301', 'a71001', '911003'
                ]
        }

    Map = geemap.Map()
    Map.setCenter(data_info.longitude[0],  data_info.latitude[0], 4)
    Map.addLayer(landSurfaceTemperature, landSurfaceTemperatureVis,'Land Surface Temperature')
    Map.addLayer(US_Border);
    Map.save(r'output\my_lc_interactive_map.html')
            