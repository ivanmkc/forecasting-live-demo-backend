from typing import List

from models.forecast import Forecast


class ForecastsService:
    _forecasts: List[Forecast] = []

    def append_forecast(self, forecast: Forecast):
        self._forecasts.append(forecast)
