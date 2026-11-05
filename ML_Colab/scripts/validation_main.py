#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:08:59 2025

@author: santi
"""

import sys
sys.path.append("../code/py")

from validate_lightgbm import (
    load_model, 
    load_validation_data, 
    verify_features, 
    prepare_data, 
    evaluate_model,
    EXPECTED_FEATURES
)

model = load_model("../py_project/results/models/financial_scaled/direction_next_5/LightGBM.joblib")
df = load_validation_data("../output/tables/validation_scaled.csv")
verify_features(df, EXPECTED_FEATURES)
X, y, dates = prepare_data(df, target_col="direction_next_5")
results = evaluate_model(model, X, y, dates)