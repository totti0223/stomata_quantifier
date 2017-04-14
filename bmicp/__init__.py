from . import utils
import sys, os, time, statistics

# https://qiita.com/terms
# section 9-1. of qiita.
# program codes are free for academic use unless licence is defined by respective authors mentioned in the respective section.

def bm_icp(config_path, dir_path):
    utils.import_config(config_path)
    item_list = utils.check_type_of_input(dir_path)

    print ("analyzing.....", '\n' * 1)

    time_container = []
    all_start = time.time()

    for item in item_list:
        print (item)
        start = time.time()
        utils.analyze(item)  # core module
        end = time.time()
        time_container.append(end - start)

    all_end = time.time()

    print ("Finished. csv files and annotated images are generated in the input directory. \n")
    if len(time_container) > 1:
        print ("mean time processing:" , statistics.mean(time_container))
        print ("stdev time processing:" , statistics.stdev(time_container))
    print ("total time:", all_end - all_start)

def cui(dir_path, config_path=os.path.join(os.path.dirname(__file__), "config.ini")):
    '''
    bio-module, stomata quantifier.
    ver. 1.0, 2017/4/13
    yosuke toda
    tyosuke@aquaseerser.com

    Input parameter
        dir_path: path of file or directory that contains image files
        config_path (optional): if not defined, will use preset config.ini
    '''
    if config_path == os.path.join(os.path.dirname(__file__), "config.ini"):
        print ("using preset config.ini")

    bm_icp(config_path, dir_path)
