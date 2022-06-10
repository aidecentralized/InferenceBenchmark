import argparse
from scheduler import Scheduler

b_default = "./configs/decoder_attack.json"
s_default = "./configs/system_config.json"

parser = argparse.ArgumentParser(description='Run SIMBA benchmark')
parser.add_argument('-b', nargs='?', default=b_default, type=open,
                    help='filepath for benchmark config, default: {}'.format(b_default))
parser.add_argument('-s', nargs='?', default=s_default, type=open,
                    help='filepath for system config, default: {}'.format(s_default))

args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args)
scheduler.initialize()
scheduler.run_job()
