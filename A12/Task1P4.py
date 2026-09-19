import csv

def load_routes(filename):
    routes = []
    with open('bus_routes.csv') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            route_name = row[0]
            stops = row[1:]  
            routes.append((route_name, stops))
    return routes

def most_stops(routes):
    longest_route = None
    for route_name, stops in routes:
        if len(stops) > most_stop_count:
            most_stop_count = len(stops)
            longest_route = route_name
    return longest_route


if __name__ == '__main__':
    routes = load_routes('bus_routes.csv')
    print(most_stops(routes))