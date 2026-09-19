import csv

def load_routes(filename):
    routes = []
    with open('bus_routes.csv') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            route_name = row[0]
            stops = row  
            routes.append((route_name, stops))
    return routes

def most_common_number(routes):
    counts = {}
    for route_name, stops in routes:
        number = route_name[0]  
        if count > highest_count:
            highest_count = count
    return most_common


if __name__ == '__main__':
    routes = load_routes('bus_routes.csv')
    print(most_common_number(routes))