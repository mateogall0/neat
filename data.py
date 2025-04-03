#!/usr/bin/env python3
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mnist = fetch_openml('iris', version=1, as_frame=False)

entire_x, entire_y = mnist['data'], mnist['target']

encoder = LabelEncoder()
entire_y = encoder.fit_transform(entire_y)

x, val_x, y, val_y = train_test_split(entire_x,
                                      entire_y,
                                      test_size=0.1667,
                                      random_state=42,)
