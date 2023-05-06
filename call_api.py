import tejapi as tej
tej.ApiConfig.api_key = "C2waEjXBx7SBJug2fRgQ8evS2tHrFz"
data = tej.get("TWN/AAPRTRAN",coid="A",ann_date={"gt":"2023-01-01"},opts={"sort":"tot_prc.desc"},paginate=True)


#hihi
print(data.head())