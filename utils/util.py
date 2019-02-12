#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
