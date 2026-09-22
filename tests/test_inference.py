from satellite_ml.inference import load_model, predict_one


def test_predict_one_returns_binary_prediction():
    model = load_model()

    telemetry = {
        "battery_voltage": 28.1,
        "battery_current": 1.7,
        "battery_temperature": 24.8,
        "solar_panel_current": 4.9,
        "bus_voltage": 28.0,
        "attitude_error": 0.02,
        "eclipse": 0,
    }

    prediction = predict_one(
        telemetry,
        model=model,
    )

    assert prediction in {0, 1}
