from pydantic import BaseModel, ConfigDict, Field


class TelemetryRequest(BaseModel):
    """Telemetry sample received by the prediction API."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "battery_voltage": 28.1,
                "battery_current": 1.7,
                "battery_temperature": 24.8,
                "solar_panel_current": 4.9,
                "bus_voltage": 28.0,
                "attitude_error": 0.02,
                "eclipse": 0,
            }
        },
    )

    battery_voltage: float
    battery_current: float
    battery_temperature: float
    solar_panel_current: float
    bus_voltage: float
    attitude_error: float
    eclipse: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response returned after model inference."""

    anomaly: bool
    prediction: int
    model: str
    version: str
    source_alias: str | None = None


class HealthResponse(BaseModel):
    """Health information exposed by the API."""

    status: str
    model_loaded: bool
