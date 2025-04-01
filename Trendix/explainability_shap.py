
import shap
import numpy as np
import pandas as pd
import torch

class SHAPExplainer:
    def __init__(self, model, background_data, model_type="pytorch"):
        self.model = model.eval() if model_type == "pytorch" else model
        self.model_type = model_type

        if model_type == "pytorch":
            self.explainer = shap.DeepExplainer(self.model, background_data)
        else:
            self.explainer = shap.Explainer(model, background_data)

    def explain(self, input_data):
        shap_values = self.explainer.shap_values(input_data)
        return shap_values

    def summarize(self, shap_values, input_data, feature_names=None):
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        summary_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(mean_abs_shap)),
            'importance': mean_abs_shap
        }).sort_values(by='importance', ascending=False)
        return summary_df
