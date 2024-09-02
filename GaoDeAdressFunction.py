import requests
import json

key = ''#高德
t_key='	'#腾讯
def geo(address:str)->dict:#获得经纬度
    parameters={
        'key':key,
        'citylimit':False,
        'address':address
    }
    r=requests.get("https://restapi.amap.com/v3/geocode/geo?parameters",params=parameters)
    data=r.json()['geocodes'][0]['location']
    return data

def regeo(location:str)->dict:
    parameters={
        'key':key,
        'location':location,
        'extensions':all
    }
    r=requests.get("https://restapi.amap.com/v3/geocode/regeo?parameters",params=parameters)
    data=r.json()['regeocode']['addressComponent']['building']['name']
    return data

def walking(origin:str,destination:str)->dict:
    parameters={
        'key':key,
        'origin':origin,
        'destination':destination,
        'output':'json'
    }
    r=requests.get("https://restapi.amap.com/v3/direction/walking?parameters",params=parameters)
    data=r.json()['route']['paths'][0]['steps']
    address1=regeo(origin)
    address2=regeo(destination)
    print(f'从{address1}到{address2}的路线为')
    for i in range (0,len(data)):
        print(data[int(i)]['instruction'])

def get_location_by_ip():
    parameters={
        'key':t_key,
        'output': 'json'
    }
    r = requests.get("https://apis.map.qq.com/ws/location/v1/ip", params=parameters)
    data = r.json()
    return data
if __name__ == '__main__':
    print(get_location_by_ip())
