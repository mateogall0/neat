#!/usr/bin/env python3
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')

entire_x, entire_y = mnist['data'].values, mnist['target'].values

x, val_x, y, val_y = train_test_split(entire_x,
                                      entire_y,
                                      test_size=0.1667,
                                      random_state=42,)
