Title:NYC Cab Project
Date:2016-08-15
Category: Machine Learning

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import shapefile
import fiona
from itertools import chain
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from descartes import PolygonPatch
import datetime
%matplotlib inline

FILE = "/Users/GCleverly/Documents/GA-DS-15/yellow_tripdata_2014-01.csv"

cab_data = pd.read_csv(FILE, nrows = 500)
```


```python
print(cab_data.columns)
total_max = max(cab_data[' total_amount'])
cab_data.head()
```

    Index(['vendor_id', ' pickup_datetime', ' dropoff_datetime',
           ' passenger_count', ' trip_distance', ' pickup_longitude',
           ' pickup_latitude', ' rate_code', ' store_and_fwd_flag',
           ' dropoff_longitude', ' dropoff_latitude', ' payment_type',
           ' fare_amount', ' surcharge', ' mta_tax', ' tip_amount',
           ' tolls_amount', ' total_amount'],
          dtype='object')





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>rate_code</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>surcharge</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CMT</td>
      <td>2014-01-09 20:45:25</td>
      <td>2014-01-09 20:52:31</td>
      <td>1</td>
      <td>0.7</td>
      <td>-73.994770</td>
      <td>40.736828</td>
      <td>1</td>
      <td>N</td>
      <td>-73.982227</td>
      <td>40.731790</td>
      <td>CRD</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.40</td>
      <td>0</td>
      <td>8.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CMT</td>
      <td>2014-01-09 20:46:12</td>
      <td>2014-01-09 20:55:12</td>
      <td>1</td>
      <td>1.4</td>
      <td>-73.982392</td>
      <td>40.773382</td>
      <td>1</td>
      <td>N</td>
      <td>-73.960449</td>
      <td>40.763995</td>
      <td>CRD</td>
      <td>8.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.90</td>
      <td>0</td>
      <td>11.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CMT</td>
      <td>2014-01-09 20:44:47</td>
      <td>2014-01-09 20:59:46</td>
      <td>2</td>
      <td>2.3</td>
      <td>-73.988570</td>
      <td>40.739406</td>
      <td>1</td>
      <td>N</td>
      <td>-73.986626</td>
      <td>40.765217</td>
      <td>CRD</td>
      <td>11.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.50</td>
      <td>0</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CMT</td>
      <td>2014-01-09 20:44:57</td>
      <td>2014-01-09 20:51:40</td>
      <td>1</td>
      <td>1.7</td>
      <td>-73.960213</td>
      <td>40.770464</td>
      <td>1</td>
      <td>N</td>
      <td>-73.979863</td>
      <td>40.777050</td>
      <td>CRD</td>
      <td>7.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.70</td>
      <td>0</td>
      <td>10.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CMT</td>
      <td>2014-01-09 20:47:09</td>
      <td>2014-01-09 20:53:32</td>
      <td>1</td>
      <td>0.9</td>
      <td>-73.995371</td>
      <td>40.717248</td>
      <td>1</td>
      <td>N</td>
      <td>-73.984367</td>
      <td>40.720524</td>
      <td>CRD</td>
      <td>6.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.75</td>
      <td>0</td>
      <td>8.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Look at Max/Min/Mean of some columns
price_columns = [' trip_distance',' fare_amount', ' surcharge', ' tip_amount',' tolls_amount', ' total_amount']

for price in price_columns:
    
    num_zero = 0
    max_col = max(cab_data[price])
    print("The max",price,"is: ",max_col)
    min_col = min(cab_data[price])
    print("The min",price,"is: ",min_col)
    mean_col = np.mean(cab_data[price])
    print("The mean",price,"is: ",mean_col)
    num_nonzero = np.count_nonzero(cab_data[price])
    print("The number of non-zero data points is", num_nonzero)
    print("\n")
    

#Check which columns are useful to your analysis
test_columns = ['vendor_id',' mta_tax',' rate_code',' store_and_fwd_flag', ' payment_type']

data_columns = cab_data[test_columns].columns.copy()
data_test = cab_data[test_columns].copy()

column_testing = []
for row in data_test:
    column_testing.append(data_test[row].unique())
print("-"*100)    
print("Theses are the unique values in the data", column_testing)
print("-"*100)

#How many 0 tips were given
tip_zero = [tip == 0 for tip in cab_data[' tip_amount']]
print("The number of times no tip was given:" ,len(tip_zero))

cab_data = cab_data.drop(test_columns,1)
```

    The max  trip_distance is:  24.7
    The min  trip_distance is:  0.0
    The mean  trip_distance is:  3.0016
    The number of non-zero data points is 496
    
    
    The max  fare_amount is:  90.0
    The min  fare_amount is:  2.5
    The mean  fare_amount is:  12.66
    The number of non-zero data points is 500
    
    
    The max  surcharge is:  1.0
    The min  surcharge is:  0.0
    The mean  surcharge is:  0.69
    The number of non-zero data points is 483
    
    
    The max  tip_amount is:  32.0
    The min  tip_amount is:  0.0
    The mean  tip_amount is:  2.49288
    The number of non-zero data points is 485
    
    
    The max  tolls_amount is:  9.0
    The min  tolls_amount is:  0.0
    The mean  tolls_amount is:  0.29184
    The number of non-zero data points is 26
    
    
    The max  total_amount is:  118.8
    The min  total_amount is:  4.0
    The mean  total_amount is:  16.63172
    The number of non-zero data points is 500
    
    
    ----------------------------------------------------------------------------------------------------
    Theses are the unique values in the data [array(['CMT'], dtype=object), array([ 0.5,  0. ]), array([1, 2, 5]), array(['N', 'Y'], dtype=object), array(['CRD'], dtype=object)]
    ----------------------------------------------------------------------------------------------------
    The number of times no tip was given: 500



```python
print(cab_data.columns)

#clean_pickup.hist(figsize=(15,15))
plt.figure()
plt.title("Histogram of Trip Distances")
cab_data[' trip_distance'].hist()
np.dtype(cab_data[' passenger_count'])
plt.figure()
plt.title("Histogram of Number of Passangers")
cab_data[' passenger_count'].hist()
plt.figure()
cab_data[' tolls_amount'].hist()

plt.figure()
plt.scatter(cab_data[' tip_amount'],cab_data[' trip_distance']);
plt.title("Graph of tip vs trip distance")
plt.xlabel("Tip Amount")
plt.ylabel("Distance")
```

    Index([' pickup_datetime', ' dropoff_datetime', ' passenger_count',
           ' trip_distance', ' pickup_longitude', ' pickup_latitude',
           ' dropoff_longitude', ' dropoff_latitude', ' fare_amount', ' surcharge',
           ' tip_amount', ' tolls_amount', ' total_amount'],
          dtype='object')





    <matplotlib.text.Text at 0x112952b38>




![png](images/ProjectPart_3_2.png)



![png](images/ProjectPart_3_3.png)



![png](images/ProjectPart_3_4.png)



![png](images/ProjectPart_3_5.png)



```python
#Trying to find address using logitude and latitude
pickup_locations = cab_data[[' pickup_longitude',' pickup_latitude']].copy()
dropoff_locations = cab_data[[' dropoff_longitude',' dropoff_latitude']].copy()
```


```python
clean_pickup = pickup_locations[(pickup_locations[' pickup_latitude'].values != 0) & (pickup_locations[' pickup_longitude'].values != 0)]
clean_dropoff = dropoff_locations[(dropoff_locations[' dropoff_latitude'].values != 0)& (dropoff_locations[' dropoff_longitude'].values != 0)]

print(pickup_locations.columns)

#From this we can see that there are coordinates with (0,0)
num_nonzero_long = np.count_nonzero(clean_pickup[' pickup_longitude'])
num_nonzero_lat = np.count_nonzero(clean_pickup[' pickup_latitude'])
print(len(clean_pickup))
print(num_nonzero_long)
print(num_nonzero_long)

```

    Index([' pickup_longitude', ' pickup_latitude'], dtype='object')
    486
    486
    486



```python
#from sklearn.cluster import KMeans
#km = KMeans(3, init='k-means++')
#km.fit(clean_pickup)
#c = km.predict(clean_pickup)

print(np.count_nonzero(cab_data[' trip_distance']))
cab_data = cab_data[cab_data[' trip_distance'].values != 0]
print(len(cab_data))

cab_data.corr()
```

    496
    496





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>fare_amount</th>
      <th>surcharge</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>passenger_count</th>
      <td>1.000000</td>
      <td>0.007208</td>
      <td>-0.064719</td>
      <td>0.064941</td>
      <td>-0.064840</td>
      <td>0.064996</td>
      <td>0.033917</td>
      <td>-0.032998</td>
      <td>0.026429</td>
      <td>0.025874</td>
      <td>0.034033</td>
    </tr>
    <tr>
      <th>trip_distance</th>
      <td>0.007208</td>
      <td>1.000000</td>
      <td>-0.037750</td>
      <td>0.037985</td>
      <td>-0.038334</td>
      <td>0.037661</td>
      <td>0.959922</td>
      <td>-0.323674</td>
      <td>0.653436</td>
      <td>0.637439</td>
      <td>0.949864</td>
    </tr>
    <tr>
      <th>pickup_longitude</th>
      <td>-0.064719</td>
      <td>-0.037750</td>
      <td>1.000000</td>
      <td>-0.999988</td>
      <td>0.999993</td>
      <td>-0.999983</td>
      <td>-0.032065</td>
      <td>0.012679</td>
      <td>-0.023809</td>
      <td>-0.036761</td>
      <td>-0.033625</td>
    </tr>
    <tr>
      <th>pickup_latitude</th>
      <td>0.064941</td>
      <td>0.037985</td>
      <td>-0.999988</td>
      <td>1.000000</td>
      <td>-0.999987</td>
      <td>0.999989</td>
      <td>0.032253</td>
      <td>-0.012338</td>
      <td>0.023762</td>
      <td>0.037347</td>
      <td>0.033833</td>
    </tr>
    <tr>
      <th>dropoff_longitude</th>
      <td>-0.064840</td>
      <td>-0.038334</td>
      <td>0.999993</td>
      <td>-0.999987</td>
      <td>1.000000</td>
      <td>-0.999983</td>
      <td>-0.032728</td>
      <td>0.012864</td>
      <td>-0.024198</td>
      <td>-0.037505</td>
      <td>-0.034298</td>
    </tr>
    <tr>
      <th>dropoff_latitude</th>
      <td>0.064996</td>
      <td>0.037661</td>
      <td>-0.999983</td>
      <td>0.999989</td>
      <td>-0.999983</td>
      <td>1.000000</td>
      <td>0.031704</td>
      <td>-0.012559</td>
      <td>0.023287</td>
      <td>0.037436</td>
      <td>0.033307</td>
    </tr>
    <tr>
      <th>fare_amount</th>
      <td>0.033917</td>
      <td>0.959922</td>
      <td>-0.032065</td>
      <td>0.032253</td>
      <td>-0.032728</td>
      <td>0.031704</td>
      <td>1.000000</td>
      <td>-0.302448</td>
      <td>0.658367</td>
      <td>0.666293</td>
      <td>0.985827</td>
    </tr>
    <tr>
      <th>surcharge</th>
      <td>-0.032998</td>
      <td>-0.323674</td>
      <td>0.012679</td>
      <td>-0.012338</td>
      <td>0.012864</td>
      <td>-0.012559</td>
      <td>-0.302448</td>
      <td>1.000000</td>
      <td>-0.226890</td>
      <td>-0.212843</td>
      <td>-0.284071</td>
    </tr>
    <tr>
      <th>tip_amount</th>
      <td>0.026429</td>
      <td>0.653436</td>
      <td>-0.023809</td>
      <td>0.023762</td>
      <td>-0.024198</td>
      <td>0.023287</td>
      <td>0.658367</td>
      <td>-0.226890</td>
      <td>1.000000</td>
      <td>0.455844</td>
      <td>0.760879</td>
    </tr>
    <tr>
      <th>tolls_amount</th>
      <td>0.025874</td>
      <td>0.637439</td>
      <td>-0.036761</td>
      <td>0.037347</td>
      <td>-0.037505</td>
      <td>0.037436</td>
      <td>0.666293</td>
      <td>-0.212843</td>
      <td>0.455844</td>
      <td>1.000000</td>
      <td>0.715963</td>
    </tr>
    <tr>
      <th>total_amount</th>
      <td>0.034033</td>
      <td>0.949864</td>
      <td>-0.033625</td>
      <td>0.033833</td>
      <td>-0.034298</td>
      <td>0.033307</td>
      <td>0.985827</td>
      <td>-0.284071</td>
      <td>0.760879</td>
      <td>0.715963</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure()
plt.scatter(clean_pickup[' pickup_longitude'],clean_pickup[' pickup_latitude']);
fig = plt.figure(figsize=(120,100))
```


![png](images/ProjectPart_7_0.png)



    <matplotlib.figure.Figure at 0x11297dba8>



```python
m = Basemap(projection='merc', llcrnrlat=40.65, urcrnrlat=40.85, llcrnrlon=-74.25, urcrnrlon=-73.5, resolution ="i")
m.readshapefile("/Users/GCleverly/Documents/GA-DS-15/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY", "NYC")
```




    (266,
     5,
     [-78.9125492727273, 40.495691577906, 0.0, 0.0],
     [-73.6998272727273, 43.2675900246194, 0.0, 0.0],
     <matplotlib.collections.LineCollection at 0x112950898>)




![png](images/ProjectPart_8_1.png)



```python
shp = fiona.open("/Users/GCleverly/Documents/GA-DS-15/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shp")
bds = shp.bounds
shp.close()
extra = 0.01
ll = (bds[0], bds[1])
ur = (bds[2], bds[3])
coords = list(chain(ll, ur))
print(bds)
print(coords)
#w, h = coords[2] - coords[0], coords[3] - coords[1]
#print(h,w)
```

    (-78.9125492727273, 40.495691577906, -73.6998272727273, 43.2675900246194)
    [-78.9125492727273, 40.495691577906, -73.6998272727273, 43.2675900246194]



```python
m1 = Basemap(
    projection='tmerc',
    lon_0=-74,
    lat_0=40.75,
    #ellps = 'WGS84',
    llcrnrlon=-74.04, #coords[0], #- extra * w,
    llcrnrlat=40.7,#coords[1], #- extra + 0.01 * h,
    urcrnrlon=-73.9,#coords[2], #+ extra * w,
    urcrnrlat=40.83,#coords[3], #+ extra + 0.01 * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True);
m1.readshapefile("/Users/GCleverly/Documents/GA-DS-15/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY", "NYC")
```




    (266,
     5,
     [-78.9125492727273, 40.495691577906, 0.0, 0.0],
     [-73.6998272727273, 43.2675900246194, 0.0, 0.0],
     <matplotlib.collections.LineCollection at 0x10bf20e10>)




![png](images/ProjectPart_10_1.png)



```python
#print(m1.NYC_info)
list_city = [item for item in m1.NYC_info if item["CITY"] == "New York City-Manhattan"]
city_df = pd.DataFrame(list_city)

#print(m1.NYC_info)

df_map = pd.DataFrame({
    'poly': [Polygon(xy) for xy in m1.NYC],
    'city_name': [city['CITY'] for city in m1.NYC_info],
    'neighborhood':[hood['NAME'] for hood in m1.NYC_info]})
#df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
#df_map['area_km'] = df_map['area_m'] / 100000
#drop_cities = ["Albany","Buffalo","Syracuse"]
df_map = df_map[df_map['city_name'].str.contains("New York City")]
print(df_map[df_map["city_name"].str.contains("New York City-Manhattan")])

```

                       city_name         neighborhood  \
    1    New York City-Manhattan         West Village   
    4    New York City-Manhattan         East Village   
    13   New York City-Manhattan         Battery Park   
    16   New York City-Manhattan        Carnegie Hill   
    30   New York City-Manhattan             Gramercy   
    46   New York City-Manhattan                 Soho   
    71   New York City-Manhattan          Murray Hill   
    88   New York City-Manhattan         Little Italy   
    99   New York City-Manhattan         Central Park   
    112  New York City-Manhattan    Greenwich Village   
    114  New York City-Manhattan              Midtown   
    118  New York City-Manhattan  Morningside Heights   
    122  New York City-Manhattan               Harlem   
    136  New York City-Manhattan     Hamilton Heights   
    150  New York City-Manhattan              Tribeca   
    184  New York City-Manhattan    North Sutton Area   
    189  New York City-Manhattan      Upper East Side   
    197  New York City-Manhattan   Financial District   
    202  New York City-Manhattan               Inwood   
    210  New York City-Manhattan              Chelsea   
    213  New York City-Manhattan      Lower East Side   
    224  New York City-Manhattan            Chinatown   
    233  New York City-Manhattan   Washington Heights   
    240  New York City-Manhattan      Upper West Side   
    245  New York City-Manhattan              Clinton   
    251  New York City-Manhattan            Yorkville   
    285  New York City-Manhattan     Garment District   
    286  New York City-Manhattan          East Harlem   
    
                                                      poly  
    1    POLYGON ((2198.902218867855 3320.69318842749, ...  
    4    POLYGON ((4450.045152566122 3743.863652142874,...  
    13   POLYGON ((2330.667661432524 2169.241159487206,...  
    16   POLYGON ((7509.733513069918 9545.777416660043,...  
    30   POLYGON ((4670.231698318657 5401.853446721477,...  
    46   POLYGON ((2472.053909535906 2935.001177595791,...  
    71   POLYGON ((4670.231698318657 5401.853446721477,...  
    88   POLYGON ((3629.853361399193 2534.468128807741,...  
    99   POLYGON ((7396.579767509133 10906.56415895398,...  
    112  POLYGON ((2943.020680037649 3231.858144577078,...  
    114  POLYGON ((3659.066849076051 8192.606779367994,...  
    118  POLYGON ((5808.571207560421 11775.21328707697,...  
    122  POLYGON ((9033.608157467623 11961.15968968496,...  
    136  POLYGON ((7244.054711806729 12368.20891354683,...  
    150  POLYGON ((2273.727029173389 1741.76125949653, ...  
    184  POLYGON ((6373.346858399922 6758.833323888517,...  
    189  POLYGON ((8212.35608579433 8874.132999186542, ...  
    197  POLYGON ((2273.727029173389 1741.76125949653, ...  
    202  POLYGON ((10521.89764513508 19487.54836196547,...  
    210  POLYGON ((2328.76992160338 4838.975675337106, ...  
    213  POLYGON ((5605.393924041886 2140.591326109743,...  
    224  POLYGON ((3130.497918931239 1782.874651626219,...  
    233  POLYGON ((7571.382681259376 14966.36501148622,...  
    240  POLYGON ((6895.092670978767 11184.51671115431,...  
    245  POLYGON ((3585.766903835119 7432.39191665716, ...  
    251  POLYGON ((8227.150055877233 9287.543862532439,...  
    285  POLYGON ((3780.661878788218 5534.929076536154,...  
    286  POLYGON ((9062.076828796917 10341.71971010629,...  



```python
# Create Point objects in map coordinates from dataframe lon and lat values
map_points_pickup = pd.Series(
    [Point(m1(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(clean_pickup[' pickup_longitude'],clean_pickup[' pickup_latitude'])])
pickup_points = MultiPoint(list(map_points_pickup.values))

map_points_dropoff = pd.Series(
    [Point(m1(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(clean_dropoff[' dropoff_longitude'],clean_dropoff[' dropoff_latitude'])])
dropoff_points = MultiPoint(list(map_points_dropoff.values))


borough_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
# calculate points that fall within the NYC boundary
ny_points_pickup = filter(borough_polygon.contains, pickup_points)
ny_points_dropoff = filter(borough_polygon.contains, dropoff_points)
#l = list(ny_points_pickup)
#print(len(l))

x_points_pickup,x_points_dropoff = [],[]
y_points_pickup,y_points_dropoff = [],[]

for pts in ny_points_pickup:
    x_points_pickup.append(pts.x)
    y_points_pickup.append(pts.y)
    
for pt in ny_points_dropoff:
    x_points_dropoff.append(pt.x)
    y_points_dropoff.append(pt.y)

print(len(x_points_pickup))
print(len(y_points_pickup))
```

    462
    462



```python
hood_dict = {}
for hood in df_map['neighborhood']:
    if hood in hood_dict.keys():
        hood_dict[hood] +=1
    else:
        hood_dict[hood] = 1

b=max(hood_dict, key=lambda k: hood_dict[k])
print(b)
#print(hood_dict['Red Hook'])

df_map_east = df_map[df_map['neighborhood'] == "Tribeca"].copy()
print(df_map_east)
```

    Red Hook
                       city_name neighborhood  \
    150  New York City-Manhattan      Tribeca   
    
                                                      poly  
    150  POLYGON ((2273.727029173389 1741.76125949653, ...  



```python
#find the points for only eastchester

# Create Point objects for the region we want to examine
#list_of_locations = []

#map_points_pickup = pd.Series(
 #   [Point(m1(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(clean_pickup[' pickup_longitude'],clean_pickup[' pickup_latitude'])])
#pickup_points = MultiPoint(list(map_points_pickup.values))

#map_points_dropoff = pd.Series(
 #   [Point(m1(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(clean_dropoff[' dropoff_longitude'],clean_dropoff[' dropoff_latitude'])])
#dropoff_points = MultiPoint(list(map_points_dropoff.values))
uptown_areas = ["Upper West Side","Upper East Side","Harlem","Morningside Heights"]
df_map_east = df_map[df_map['neighborhood'][uptown_areas]].copy()

east_polygon = prep(MultiPolygon(list(df_map_east['poly'].values)))

east_points_pickup = filter(east_polygon.contains, pickup_points)
east_points_dropoff = filter(borough_polygon.contains, dropoff_points)

west_points_pickup = filter(borough_polygon.contains, pickup_points)
west_points_dropoff = filter(east_polygon.contains, dropoff_points)

x_east_pickup,x_east_dropoff = [],[]
y_east_pickup,y_east_dropoff = [],[]

for pt in east_points_pickup:
    x_east_pickup.append(pt.x)
    y_east_pickup.append(pt.y)
    
for pt in east_points_dropoff:
    x_east_dropoff.append(pt.x)
    y_east_dropoff.append(pt.y)
    
x_west_pickup,x_west_dropoff = [],[]
y_west_pickup,y_west_dropoff = [],[]

for pt in west_points_pickup:
    x_west_pickup.append(pt.x)
    y_west_pickup.append(pt.y)
    
for pt in west_points_dropoff:
    x_west_dropoff.append(pt.x)
    y_west_dropoff.append(pt.y)

#print(len(x_east_dropoff),len(x_east_pickup),len(y_east_dropoff),len(y_east_pickup))

lengths_plot = [len(x_east_dropoff),len(x_east_pickup),len(y_east_dropoff),len(y_east_pickup)]
max_length_plot = min(lengths_plot)

lengths_plot_b = [len(x_west_dropoff),len(x_west_pickup),len(y_west_dropoff),len(y_west_pickup)]
max_length_plot_b = min(lengths_plot_b)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-118-7b5b06f59db3> in <module>()
         12 #dropoff_points = MultiPoint(list(map_points_dropoff.values))
         13 uptown_areas = ["Upper West Side","Upper East Side","Harlem","Morningside Heights"]
    ---> 14 df_map_east = df_map[df_map['neighborhood'][uptown_areas]].copy()
         15 
         16 east_polygon = prep(MultiPolygon(list(df_map_east['poly'].values)))


    //anaconda/lib/python3.5/site-packages/pandas/core/frame.py in __getitem__(self, key)
       1906         if isinstance(key, (Series, np.ndarray, Index, list)):
       1907             # either boolean or fancy integer index
    -> 1908             return self._getitem_array(key)
       1909         elif isinstance(key, DataFrame):
       1910             return self._getitem_frame(key)


    //anaconda/lib/python3.5/site-packages/pandas/core/frame.py in _getitem_array(self, key)
       1933     def _getitem_array(self, key):
       1934         # also raises Exception if object array with NA values
    -> 1935         if com.is_bool_indexer(key):
       1936             # warning here just in case -- previously __setitem__ was
       1937             # reindexing but __getitem__ was not; it seems more reasonable to


    //anaconda/lib/python3.5/site-packages/pandas/core/common.py in is_bool_indexer(key)
       2073             if not lib.is_bool_array(key):
       2074                 if isnull(key).any():
    -> 2075                     raise ValueError('cannot index with vector containing '
       2076                                      'NA / NaN values')
       2077                 return False


    ValueError: cannot index with vector containing NA / NaN values



```python
# draw ward patches from polygons
#df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
#    x,
#    fc='#555555',
#    ec='#787878', lw=.25, alpha=.9,#label="New York",
#    zorder=4))

s = []
for x,y in zip(df_map['poly'],df_map['neighborhood']):
    s.append(PolygonPatch(x,fc='#555555',ec='#787878',lw=0.25,alpha=0.9,zorder=4,label=y))
df_map['patches'] = s 

plt.clf()
fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

ax.scatter(
    x_points_pickup, y_points_pickup,
    5, marker='o', lw=.25,
    facecolor='#33ccff', edgecolor='w',
    alpha=0.9, antialiased=True,
    label='Cab Pickups', zorder=3)
ax.scatter(
    x_points_dropoff, y_points_dropoff,
    5, marker='o', lw=.25,
    facecolor='#8ec256', edgecolor='w',
    alpha=0.9, antialiased=True,
    label='Cab Pickups', zorder=3)

#Plotting trips from West Village
m1.plot((x_east_pickup[0:max_length_plot],x_east_dropoff[0:max_length_plot]),(y_east_pickup[0:max_length_plot],y_east_dropoff[0:max_length_plot]), linewidth=1,c="b")
m1.plot((x_west_pickup[0:max_length_plot_b],x_west_dropoff[0:max_length_plot_b]),(y_west_pickup[0:max_length_plot_b],y_west_dropoff[0:max_length_plot_b]), linewidth=1,c="c")
#m1.drawgreatcircle(x_points_pickup[0],y_points_pickup[0],x_points_dropoff[0],y_points_dropoff[0], linewidth=1)
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))


#The following is preliminary work to get neighborhood labels on the plot
#ax.text(4, 20, 'boxed italics text in data coords', style='italic')
#for x in df_map['neighborhood']:
#    ax.annotate(x,xy=(10,300))
plt.show()
```


    <matplotlib.figure.Figure at 0x10f015780>



![png](images/ProjectPart_15_1.png)



```python

```


```python
#Now I want to plot the time histogram
time_df = cab_data[[' pickup_datetime',' dropoff_datetime']].copy()
time_df.columns = ['pickup_datetime','dropoff_datetime']

time_df['pickup_datetime'] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in time_df['pickup_datetime']]
time_df['dropoff_datetime'] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in time_df['dropoff_datetime']]

time_df.head()

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-01-09 20:45:25</td>
      <td>2014-01-09 20:52:31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-01-09 20:46:12</td>
      <td>2014-01-09 20:55:12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-01-09 20:44:47</td>
      <td>2014-01-09 20:59:46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-01-09 20:44:57</td>
      <td>2014-01-09 20:51:40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-01-09 20:47:09</td>
      <td>2014-01-09 20:53:32</td>
    </tr>
  </tbody>
</table>
</div>




```python
def chart_hours(occurance_list):
    hour_list = [t.hour for t in occurance_list]
    numbers=[x for x in range(0,24)]
    labels=map(lambda x: str(x), numbers)
    plt.xticks(numbers, labels)
    plt.xlim(0,24)
    plt.hist(hour_list)
    plt.show()
    
def chart_days(occurance_list):
    days_list = [t.day for t in occurance_list]
    numbers=[x for x in range(0,31)]
    labels=map(lambda x: str(x), numbers)
    plt.xticks(numbers, labels)
    plt.xlim(0,31)
    plt.hist(days_list)
    plt.show()

chart_hours(time_df['pickup_datetime'])
chart_hours(time_df['pickup_datetime'])

chart_days(time_df['pickup_datetime'])

```


![png](images/ProjectPart_18_0.png)



![png](images/ProjectPart_18_1.png)



![png](images/ProjectPart_18_2.png)



```python
#Need functionality to change neighborhood via a slider
#Need better presentation with the trip path
```


```python

```


```python

```


```python

```
