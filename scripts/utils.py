# -*- encoding: utf-8 -*-

## Imports
import gc
import os
import numpy as np
import pandas as pd
import random
import polars as pl
import yaml


def load_config(path):
    """YAMLファイルを読み込む関数"""
    
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config