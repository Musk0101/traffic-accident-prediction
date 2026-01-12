package com.ai.trafficbackend.dto;

import java.util.Map;

public class ExplainResponse {

    private double prediction;
    private Map<String, Double> shap_values;

    public double getPrediction() { return prediction; }
    public void setPrediction(double prediction) { this.prediction = prediction; }

    public Map<String, Double> getShap_values() { return shap_values; }
    public void setShap_values(Map<String, Double> shap_values) {
        this.shap_values = shap_values;
    }
}

