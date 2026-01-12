package com.ai.trafficbackend.controller;

import java.util.HashMap;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
@RestController
@RequestMapping("/api/traffic")
@CrossOrigin(origins = "http://localhost:5173")
public class TrafficController {

    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody TrafficRequest request) {
        Map<String, Object> response = new HashMap<>();
        response.put("prediction", "LOW");
        response.put("confidence", 0.82);
        return ResponseEntity.ok(response);
    }

    public static class TrafficRequest {
        // Add request fields as needed, example:
        // public String location;
        // public int hour;
        // public TrafficRequest() {}
        // Getters/setters can be added if required by deserialization
    }
}

    


