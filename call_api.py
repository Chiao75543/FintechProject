import tejapi as tej
import pandas as pd
tej.ApiConfig.api_key = "C2waEjXBx7SBJug2fRgQ8evS2tHrFz"
# data = tej.get("TWN/AAPRTRAN",coid="A",ann_date={"gt":"2023-01-01"},opts={"sort":"tot_prc.desc"},paginate=True)

df_aaprrent = pd.DataFrame(tej.get('TWN/AAPRRENT',tsign ='土地',mdate={'gt':'2020-01-01'},paginate=True))
df_aaprtran = pd.DataFrame(tej.get('TWN/AAPRTRAN',tsign ='土地',mdate={'gt':'2020-01-01'},paginate=True))
# df_alandtr = pd.DataFrame(tej.get('TWN/ALANDTR',coid='A',mdate={'gt':'2018-01-01'},paginate=True))

df_aaprrent.to_csv("AAPRRENT.csv")
df_aaprtran.to_csv("AAPRTRAN.csv")


