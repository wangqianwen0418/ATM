from atm.database import Database, db_session
from atm.utilities import load_metrics
from atm.config import (add_arguments_aws_s3, add_arguments_logging,
                        add_arguments_sql, load_config, initialize_logging)

import argparse

parser = argparse.ArgumentParser(description='Add more classifiers to database')
add_arguments_sql(parser)
add_arguments_aws_s3(parser)
add_arguments_logging(parser)

# add worker-specific arguments
parser.add_argument('--cloud-mode', action='store_true', default=False,
                    help='Whether to run this worker in cloud mode')
parser.add_argument('--dataruns', help='Only train on dataruns with these ids',
                    nargs='+')
parser.add_argument('--time', help='Number of seconds to run worker', type=int)
parser.add_argument('--choose-randomly', action='store_true',
                    help='Choose dataruns to work on randomly (default = sequential order)')
parser.add_argument('--no-save', dest='save_files', default=True,
                    action='store_const', const=False,
                    help="don't save models and metrics at all")

# parse arguments and load configuration
args = parser.parse_args()

sql_config, _, aws_config, log_config = load_config(**vars(args))

db = Database(**vars(sql_config))


with db_session(db): # keep a database session open to access the dataruns
    ## get all the classifier in the dataset
    classifiers = db.get_classifiers()
    ## or
    ## get one classifier by the classifier ID
    # classifier = db.get_classifier(classifier_id)
    print("total {} classifiers".format( len(classifiers) ))
    for classifier in classifiers:
        metrics = load_metrics(classifier, metric_dir="./metrics")
        print(metrics)