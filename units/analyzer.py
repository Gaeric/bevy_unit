#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: analyzer.py
# Datetime: Mon Jul 21 17:22:00 2025
# Software: Emacs
# Author:   Gaeric

import re

vec3_re = re.compile(r'Vec3\((.*)\,(.*)\,(.*)\)\, rotation:.*')

positions = []

with open('useful.log') as fp:
    for line in fp.readlines():
        match = vec3_re.match(line)
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        positions.append((x, y, z))

groups = len(positions) - 1

for i in range(groups):
    x0, y0, z0 = positions[i]
    x1, y1, z1 = positions[i + 1]

    print(f'z length is {z1 - z0}')
