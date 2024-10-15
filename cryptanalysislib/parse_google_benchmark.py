#!/usr/bin/env python3
import pathlib 
import logging 
import json 
from typing import List

def read_google_benchmark_data(file: str) -> List:
    """parse a csv/json output of a google benchmark run """
    extension = pathlib.Path(file).suffix
    data = []
    with open(file) as fp:
        try:
            if extension == ".csv":
                # data = pd.read_csv(args.file, usecols=["name", args.metric])
                print("error not impl") 
            elif extension == ".json":
                json_data = json.load(fp)
                data = json_data["benchmarks"]
            else:
                logging.error("Unsupported file extension '{}'".format(extension))
                exit(1)
        except ValueError:
            logging.error('Could not parse the benchmark data.')
            return data
    # read atleast a little bit
    assert len(data) > 0

    # publish the two needed fields `label`, `size`
    for d in data:
        d["label"] = d["name"].split("/")[0]
        d["size"] = d["name"].split("/")[1]
    return data
