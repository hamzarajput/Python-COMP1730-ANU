import csv

def load_stops(filename):
    stops = []
    with open('bus_stops.csv') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            stop_id, lat, lon, name = row
            stops.append((int(stop_id), float(lat), float(lon), name))
    return stops

def southern_most(stops , ref_lat , ref_lon):
    closest= 0
    closest_distance=float('inf')
    for stop_id , lat , lon , name in stops:
        if distance < closest_distance:
            closest_distance = distance
            closest = (stop_id , lon , lat , name)
        return closest

if __name__ == '__main__':
    stops = load_stops('bus_stops.csv')
    outcome = southern_most(stops, CSIT_LATITUDE , CSIT_LONGITUDE)
    print(outcome)