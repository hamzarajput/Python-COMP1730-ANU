import csv

def load_times(filename):
    times = []
    with open("bus_routes.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            route_name = row[0]
            stop_id = int(row[1])
            times[(route_name, stop_id)] = row[2:]
    return times

def most_common_number(routes):
    stop_counts = Counter(stop_id for(route_name , stop_id) in times.key)
    most_common_stop_id = stop_counts.most_common[0]
    return most_common_stop_id , count

if __name__ == '__main__':
    common = load_times('bus_routes.csv') 
    popular = most_common_number(times)
    print(popular)