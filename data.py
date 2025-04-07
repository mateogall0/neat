#!/usr/bin/env python3
from sklearn.datasets import fetch_openml, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = load_wine()
# dataset = fetch_openml('wine', version=1, as_frame=True)

entire_x, entire_y = dataset['data'], dataset['target']

encoder = LabelEncoder()
entire_y = encoder.fit_transform(entire_y)

x, val_x, y, val_y = train_test_split(entire_x,
                                      entire_y,
                                      test_size=0.1667,
                                      random_state=42,
                                      shuffle=True,)
