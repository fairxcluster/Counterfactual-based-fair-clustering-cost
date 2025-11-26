import configparser
import sys

from fair_clustering_v2 import fair_clustering
from util.configutil import read_list

config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
config_str = "adult" if len(sys.argv) == 1 else sys.argv[1]

print("Using config_str = {}".format(config_str))

# Read variables
data_dir = config[config_str].get("data_dir")
print(data_dir)
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
print(num_clusters)
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
violating = config["DEFAULT"].getboolean("violating")
violation = config["DEFAULT"].getfloat("violation")

seeds = list(range(10))  # Seeds from 0 to 9
for n_clusters in num_clusters:
    for seed in seeds:
        print(f"Running for k={n_clusters}, seed={seed}")
        fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, deltas, max_points, violating, violation, seed=seed)