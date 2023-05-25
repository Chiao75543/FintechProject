import tejapi as tej
import pandas as pd
tej.ApiConfig.api_key = "C2waEjXBx7SBJug2fRgQ8evS2tHrFz"

district = ['100', '103', '104', '105', '106', '108', '110', '111', '112', '114', '115', '116']
df_aaprrent = pd.DataFrame(tej.get('TWN/AAPRRENT',district = district, tsign ='土地',mdate={'gt':'2020-01-01',},paginate=True))
df_aaprtran = pd.DataFrame(tej.get('TWN/AAPRTRAN',district = district, tsign ='土地',mdate={'gt':'2020-01-01'},paginate=True))
# 這個沒有data
# df_alandtr = pd.DataFrame(tej.get('TWN/ALANDTR',coid='A',mdate={'gt':'2018-01-01'},paginate=True))

df_aaprrent.to_csv("AAPRRENT.csv")
df_aaprtran.to_csv("AAPRTRAN.csv")


