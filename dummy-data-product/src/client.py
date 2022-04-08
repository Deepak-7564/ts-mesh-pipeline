import dependencies.scraping as scraping

import ee


# Google  earth engine authrentication

ee.Authenticate()
ee.Initialize()
    
# import os
# #to get the current working directory
# directory = os.getcwd()

print("\n Please enter following information to get result data and map: \n")

scraping.scaper()
print("\n Data is Downloaded...... \n")
scraping.images_scraper()
print("Image of Graph is created..... \n")
scraping.map_scraper()
print("Map downloaded.... \n")

print('Task Completed. Please find results in output folder.....')