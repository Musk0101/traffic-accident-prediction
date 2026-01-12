package com.ai.trafficbackend.dto;

public class ExplainRequest {

    private double temperature;
    private double humidity;
    private double wind_speed;
    private double visibility;
    private double pressure;

    // Getters & Setters
    public double getTemperature() { return temperature; }
    public void setTemperature(double temperature) { this.temperature = temperature; }

    public double getHumidity() { return humidity; }
    public void setHumidity(double humidity) { this.humidity = humidity; }

    public double getWind_speed() { return wind_speed; }
    public void setWind_speed(double wind_speed) { this.wind_speed = wind_speed; }

    public double getVisibility() { return visibility; }
    public void setVisibility(double visibility) { this.visibility = visibility; }

    public double getPressure() { return pressure; }
    public void setPressure(double pressure) { this.pressure = pressure; }
}
