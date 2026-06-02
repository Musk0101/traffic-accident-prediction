import shap
import torch

def explain(model, sample):
    explainer = shap.DeepExplainer(model, sample)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values)
