import argparse
from scheduler import Scheduler

b_default = "./configs/split_inference.json"
s_default = "./configs/system_config.json"

parser = argparse.ArgumentParser(description='Run SIMBA benchmark')
parser.add_argument('-b', nargs='?', default=b_default, type=open,
                    help='filepath for benchmark config, default: {}'.format(b_default))
parser.add_argument('-s', nargs='?', default=s_default, type=open,
                    help='filepath for system config, default: {}'.format(s_default))
print("Here before")

args = parser.parse_args()
print("Here before 2")

scheduler = Scheduler(args)
print("Here")
scheduler.run_job()

print("Here out")
