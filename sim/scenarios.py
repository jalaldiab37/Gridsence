"""
Scenario Manager
Predefined scenarios and custom scenario creation for grid simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class ScenarioType(Enum):
    """Types of simulation scenarios."""
    NORMAL = "normal"
    HEAT_WAVE = "heat_wave"
    COLD_SNAP = "cold_snap"
    INDUSTRIAL_SURGE = "industrial_surge"
    RENEWABLE_VARIABILITY = "renewable_variability"
    DEMAND_GROWTH = "demand_growth"
    EQUIPMENT_FAILURE = "equipment_failure"
    CASCADING_FAILURE = "cascading_failure"
    CUSTOM = "custom"


@dataclass
class OutageScenario:
    """Definition of an outage scenario."""
    name: str
    scenario_type: ScenarioType
    description: str
    duration_hours: int = 168  # Default 1 week
    
    # Load modifiers
    demand_growth_pct: float = 0.0
    peak_amplification: float = 1.0
    
    # Weather conditions
    heat_wave: bool = False
    cold_snap: bool = False
    temperature_delta: float = 0.0
    
    # Industrial factors
    industrial_spike: bool = False
    industrial_multiplier: float = 1.0
    
    # Renewable factors
    renewable_fraction: float = 0.2
    solar_variability: float = 0.1
    wind_variability: float = 0.2
    
    # Equipment factors
    generator_outage_mw: float = 0.0
    transmission_constraint_pct: float = 0.0
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for simulation."""
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'description': self.description,
            'duration_hours': self.duration_hours,
            'demand_growth_pct': self.demand_growth_pct,
            'peak_amplification': self.peak_amplification,
            'heat_wave': self.heat_wave,
            'cold_snap': self.cold_snap,
            'temperature_delta': self.temperature_delta,
            'industrial_spike': self.industrial_spike,
            'industrial_multiplier': self.industrial_multiplier,
            'renewable_fraction': self.renewable_fraction,
            'solar_variability': self.solar_variability,
            'wind_variability': self.wind_variability,
            'generator_outage_mw': self.generator_outage_mw,
            'transmission_constraint_pct': self.transmission_constraint_pct,
            **self.extra_params
        }


class ScenarioManager:
    """
    Manage and create simulation scenarios.
    Provides predefined scenarios and custom scenario builder.
    """
    
    # Predefined scenarios
    PREDEFINED_SCENARIOS = {
        'normal_week': OutageScenario(
            name="Normal Week",
            scenario_type=ScenarioType.NORMAL,
            description="Typical week with standard load patterns",
            duration_hours=168,
            renewable_fraction=0.2
        ),
        
        'summer_heat_wave': OutageScenario(
            name="Summer Heat Wave",
            scenario_type=ScenarioType.HEAT_WAVE,
            description="Extreme heat event lasting 5 days with high AC demand",
            duration_hours=120,
            heat_wave=True,
            temperature_delta=15.0,
            peak_amplification=1.35,
            renewable_fraction=0.25  # More solar available
        ),
        
        'winter_cold_snap': OutageScenario(
            name="Winter Cold Snap",
            scenario_type=ScenarioType.COLD_SNAP,
            description="Extreme cold event with high heating demand",
            duration_hours=96,
            cold_snap=True,
            temperature_delta=-20.0,
            peak_amplification=1.40,
            renewable_fraction=0.15  # Less solar, some wind
        ),
        
        'industrial_surge': OutageScenario(
            name="Industrial Surge",
            scenario_type=ScenarioType.INDUSTRIAL_SURGE,
            description="Major industrial expansion causing sustained demand increase",
            duration_hours=168,
            industrial_spike=True,
            industrial_multiplier=1.25,
            demand_growth_pct=8.0
        ),
        
        'high_renewable': OutageScenario(
            name="High Renewable Variability",
            scenario_type=ScenarioType.RENEWABLE_VARIABILITY,
            description="Grid with 50% renewable penetration and variable output",
            duration_hours=168,
            renewable_fraction=0.50,
            solar_variability=0.3,
            wind_variability=0.4
        ),
        
        'demand_growth': OutageScenario(
            name="Rapid Demand Growth",
            scenario_type=ScenarioType.DEMAND_GROWTH,
            description="Simulates 3 years of demand growth in short period",
            duration_hours=168,
            demand_growth_pct=15.0
        ),
        
        'generator_trip': OutageScenario(
            name="Large Generator Trip",
            scenario_type=ScenarioType.EQUIPMENT_FAILURE,
            description="1000 MW generator trips offline unexpectedly",
            duration_hours=48,
            generator_outage_mw=1000.0
        ),
        
        'transmission_constraint': OutageScenario(
            name="Transmission Bottleneck",
            scenario_type=ScenarioType.EQUIPMENT_FAILURE,
            description="Major transmission line out reduces import capacity by 20%",
            duration_hours=72,
            transmission_constraint_pct=20.0
        ),
        
        'worst_case': OutageScenario(
            name="Worst Case Scenario",
            scenario_type=ScenarioType.CASCADING_FAILURE,
            description="Heat wave + generator trip + high demand - stress test",
            duration_hours=72,
            heat_wave=True,
            peak_amplification=1.4,
            generator_outage_mw=800.0,
            demand_growth_pct=10.0,
            temperature_delta=12.0
        )
    }
    
    def __init__(self):
        self.scenarios: Dict[str, OutageScenario] = dict(self.PREDEFINED_SCENARIOS)
        self.active_scenario: Optional[OutageScenario] = None
    
    def get_scenario(self, name: str) -> OutageScenario:
        """Get a scenario by name."""
        if name not in self.scenarios:
            available = list(self.scenarios.keys())
            raise ValueError(f"Unknown scenario: {name}. Available: {available}")
        return self.scenarios[name]
    
    def list_scenarios(self) -> List[Dict[str, str]]:
        """List all available scenarios with descriptions."""
        return [
            {
                'name': name,
                'type': scenario.scenario_type.value,
                'description': scenario.description,
                'duration_hours': scenario.duration_hours
            }
            for name, scenario in self.scenarios.items()
        ]
    
    def create_custom_scenario(
        self,
        name: str,
        description: str = "Custom scenario",
        **kwargs
    ) -> OutageScenario:
        """
        Create a custom scenario with specified parameters.
        
        Args:
            name: Unique scenario name
            description: Scenario description
            **kwargs: Scenario parameters
        
        Returns:
            New OutageScenario instance
        """
        scenario = OutageScenario(
            name=name,
            scenario_type=ScenarioType.CUSTOM,
            description=description,
            duration_hours=kwargs.get('duration_hours', 168),
            demand_growth_pct=kwargs.get('demand_growth_pct', 0.0),
            peak_amplification=kwargs.get('peak_amplification', 1.0),
            heat_wave=kwargs.get('heat_wave', False),
            cold_snap=kwargs.get('cold_snap', False),
            temperature_delta=kwargs.get('temperature_delta', 0.0),
            industrial_spike=kwargs.get('industrial_spike', False),
            industrial_multiplier=kwargs.get('industrial_multiplier', 1.0),
            renewable_fraction=kwargs.get('renewable_fraction', 0.2),
            solar_variability=kwargs.get('solar_variability', 0.1),
            wind_variability=kwargs.get('wind_variability', 0.2),
            generator_outage_mw=kwargs.get('generator_outage_mw', 0.0),
            transmission_constraint_pct=kwargs.get('transmission_constraint_pct', 0.0)
        )
        
        self.scenarios[name] = scenario
        return scenario
    
    def set_active_scenario(self, name: str):
        """Set the active scenario for simulation."""
        self.active_scenario = self.get_scenario(name)
    
    def modify_load_profile(
        self,
        base_load: np.ndarray,
        scenario: OutageScenario,
        start_hour: int = 0
    ) -> np.ndarray:
        """
        Modify a base load profile according to scenario parameters.
        
        Args:
            base_load: Original load forecast
            scenario: Scenario to apply
            start_hour: Starting hour of day
        
        Returns:
            Modified load profile
        """
        modified = base_load.copy()
        hours = len(base_load)
        
        # Apply demand growth
        if scenario.demand_growth_pct > 0:
            growth_factor = 1 + (scenario.demand_growth_pct / 100)
            modified *= growth_factor
        
        # Apply peak amplification (during peak hours 9 AM - 9 PM)
        if scenario.peak_amplification != 1.0:
            for i in range(hours):
                hour = (start_hour + i) % 24
                if 9 <= hour <= 21:
                    modified[i] *= scenario.peak_amplification
        
        # Heat wave effect (high AC load during afternoon)
        if scenario.heat_wave:
            for i in range(hours):
                hour = (start_hour + i) % 24
                if 10 <= hour <= 20:
                    heat_factor = 1.2 + 0.15 * np.sin((hour - 10) * np.pi / 10)
                    modified[i] *= heat_factor
        
        # Cold snap effect (high heating load morning and evening)
        if scenario.cold_snap:
            for i in range(hours):
                hour = (start_hour + i) % 24
                if 5 <= hour <= 9 or 17 <= hour <= 22:
                    cold_factor = 1.3
                else:
                    cold_factor = 1.15
                modified[i] *= cold_factor
        
        # Industrial spike (business hours on weekdays)
        if scenario.industrial_spike:
            for i in range(hours):
                hour = (start_hour + i) % 24
                day = (i // 24) % 7
                if day < 5 and 8 <= hour <= 18:  # Weekday business hours
                    modified[i] *= scenario.industrial_multiplier
        
        return modified
    
    def generate_renewable_profile(
        self,
        hours: int,
        scenario: OutageScenario,
        capacity_mw: float,
        start_hour: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate solar and wind generation profiles.
        
        Returns:
            Tuple of (solar_mw, wind_mw) arrays
        """
        np.random.seed(42)  # Reproducibility
        
        solar_capacity = capacity_mw * scenario.renewable_fraction * 0.6
        wind_capacity = capacity_mw * scenario.renewable_fraction * 0.4
        
        solar = np.zeros(hours)
        wind = np.zeros(hours)
        
        for i in range(hours):
            hour = (start_hour + i) % 24
            
            # Solar: bell curve during daylight
            if 6 <= hour <= 18:
                solar_base = np.sin((hour - 6) * np.pi / 12)
                solar_noise = np.random.normal(0, scenario.solar_variability)
                solar[i] = solar_capacity * max(0, solar_base + solar_noise)
            
            # Wind: more variable, typically higher at night
            wind_base = 0.4 + 0.2 * np.sin((hour + 6) * np.pi / 12)
            wind_noise = np.random.normal(0, scenario.wind_variability)
            wind[i] = wind_capacity * max(0, wind_base + wind_noise)
        
        return solar, wind
    
    def compare_scenarios(
        self,
        scenario_names: List[str],
        base_load: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compare multiple scenarios by applying them to same base load.
        
        Returns:
            Dictionary mapping scenario names to modified loads
        """
        results = {'base': base_load}
        
        for name in scenario_names:
            scenario = self.get_scenario(name)
            results[name] = self.modify_load_profile(base_load, scenario)
        
        return results


# Convenience function for creating scenarios
def create_scenario(name: str, **kwargs) -> OutageScenario:
    """Quick helper to create a custom scenario."""
    manager = ScenarioManager()
    return manager.create_custom_scenario(name, **kwargs)

