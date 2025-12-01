"""
Grid Simulation Engine
Simulates grid stability, outage risks, and load shedding strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class RiskLevel(Enum):
    """Grid risk level classification."""
    GREEN = "green"      # Normal operation
    YELLOW = "yellow"    # Elevated risk, monitoring required
    ORANGE = "orange"    # High risk, prepare mitigation
    RED = "red"          # Critical, immediate action needed


@dataclass
class GridState:
    """Current state of the power grid."""
    timestamp: datetime
    current_load_mw: float
    predicted_load_mw: float
    capacity_mw: float
    reserve_margin: float
    risk_level: RiskLevel
    active_generators: int
    renewable_contribution: float = 0.0
    temperature_c: float = 20.0
    
    @property
    def load_factor(self) -> float:
        """Current load as percentage of capacity."""
        return (self.current_load_mw / self.capacity_mw) * 100
    
    @property
    def available_reserve(self) -> float:
        """Available reserve capacity in MW."""
        return self.capacity_mw - self.current_load_mw


@dataclass
class SimulationResult:
    """Result of a grid simulation run."""
    scenario_name: str
    duration_hours: int
    timestamps: List[datetime]
    loads: List[float]
    risk_levels: List[RiskLevel]
    outage_events: List[Dict[str, Any]]
    load_shedding_events: List[Dict[str, Any]]
    peak_load: float
    min_reserve_margin: float
    total_unserved_energy_mwh: float
    mitigation_actions: List[str]


class GridSimulator:
    """
    Main grid simulation engine.
    Models load impacts on grid stability and simulates various scenarios.
    """
    
    # Default grid parameters
    DEFAULT_CAPACITY_MW = 10000.0
    DEFAULT_RESERVE_MARGIN = 0.15  # 15% reserve requirement
    BREAKER_RESPONSE_DELAY_MS = 50  # Circuit breaker response time
    
    def __init__(
        self,
        capacity_mw: float = DEFAULT_CAPACITY_MW,
        reserve_margin: float = DEFAULT_RESERVE_MARGIN,
        n_generators: int = 20
    ):
        self.capacity_mw = capacity_mw
        self.reserve_margin = reserve_margin
        self.n_generators = n_generators
        
        # Risk thresholds (as percentage of capacity)
        self.risk_thresholds = {
            RiskLevel.GREEN: 0.70,   # < 70% load
            RiskLevel.YELLOW: 0.80,  # 70-80% load
            RiskLevel.ORANGE: 0.90,  # 80-90% load
            RiskLevel.RED: 0.95      # > 90% load
        }
        
        # Simulation state
        self.current_state: Optional[GridState] = None
        self.history: List[GridState] = []
        self.outage_log: List[Dict[str, Any]] = []
    
    def assess_risk(self, load_mw: float) -> RiskLevel:
        """
        Assess grid risk level based on current load.
        
        Args:
            load_mw: Current or predicted load in MW
        
        Returns:
            RiskLevel enum value
        """
        load_ratio = load_mw / self.capacity_mw
        
        if load_ratio >= self.risk_thresholds[RiskLevel.RED]:
            return RiskLevel.RED
        elif load_ratio >= self.risk_thresholds[RiskLevel.ORANGE]:
            return RiskLevel.ORANGE
        elif load_ratio >= self.risk_thresholds[RiskLevel.YELLOW]:
            return RiskLevel.YELLOW
        else:
            return RiskLevel.GREEN
    
    def calculate_reserve_margin(self, load_mw: float) -> float:
        """Calculate current reserve margin."""
        return (self.capacity_mw - load_mw) / self.capacity_mw
    
    def simulate_step(
        self,
        timestamp: datetime,
        load_mw: float,
        predicted_load_mw: float,
        temperature_c: float = 20.0,
        renewable_mw: float = 0.0
    ) -> GridState:
        """
        Simulate a single time step.
        
        Args:
            timestamp: Current timestamp
            load_mw: Current load in MW
            predicted_load_mw: Forecasted load
            temperature_c: Ambient temperature
            renewable_mw: Renewable generation
        
        Returns:
            Updated GridState
        """
        # Adjust capacity for temperature (derating at high temps)
        temp_factor = 1.0
        if temperature_c > 35:
            temp_factor = 0.95 - 0.01 * (temperature_c - 35)
        effective_capacity = self.capacity_mw * temp_factor
        
        # Calculate metrics
        risk_level = self.assess_risk(load_mw)
        reserve = self.calculate_reserve_margin(load_mw)
        
        # Determine active generators
        generator_capacity = effective_capacity / self.n_generators
        active_generators = min(
            self.n_generators,
            int(np.ceil(load_mw / generator_capacity)) + 2  # +2 for reserve
        )
        
        state = GridState(
            timestamp=timestamp,
            current_load_mw=load_mw,
            predicted_load_mw=predicted_load_mw,
            capacity_mw=effective_capacity,
            reserve_margin=reserve,
            risk_level=risk_level,
            active_generators=active_generators,
            renewable_contribution=renewable_mw / load_mw if load_mw > 0 else 0,
            temperature_c=temperature_c
        )
        
        self.current_state = state
        self.history.append(state)
        
        return state
    
    def simulate_scenario(
        self,
        load_forecast: np.ndarray,
        start_time: datetime,
        scenario_params: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """
        Run a full simulation scenario.
        
        Args:
            load_forecast: Array of forecasted loads
            start_time: Simulation start time
            scenario_params: Additional scenario parameters
        
        Returns:
            SimulationResult with detailed outcomes
        """
        params = scenario_params or {}
        
        # Apply scenario modifiers
        demand_growth = params.get('demand_growth_pct', 0) / 100
        heat_wave = params.get('heat_wave', False)
        industrial_spike = params.get('industrial_spike', False)
        renewable_fraction = params.get('renewable_fraction', 0.2)
        
        # Initialize results
        timestamps = []
        loads = []
        risk_levels = []
        outage_events = []
        shedding_events = []
        unserved_energy = 0.0
        
        for i, base_load in enumerate(load_forecast):
            timestamp = start_time + timedelta(hours=i)
            
            # Apply modifiers
            load = base_load * (1 + demand_growth)
            
            if heat_wave and 10 <= timestamp.hour <= 18:
                load *= 1.25  # 25% increase during heat wave peak hours
            
            if industrial_spike and 8 <= timestamp.hour <= 17 and timestamp.weekday() < 5:
                load *= 1.1  # 10% industrial increase
            
            # Temperature from heat wave or default
            temp = 40 if heat_wave else 20 + 10 * np.sin((timestamp.hour - 6) * np.pi / 12)
            
            # Renewable generation (simplified solar curve)
            renewable_mw = 0
            if renewable_fraction > 0:
                solar_factor = max(0, np.sin((timestamp.hour - 6) * np.pi / 12))
                renewable_mw = self.capacity_mw * renewable_fraction * solar_factor * 0.5
            
            # Simulate step
            state = self.simulate_step(timestamp, load, base_load, temp, renewable_mw)
            
            timestamps.append(timestamp)
            loads.append(load)
            risk_levels.append(state.risk_level)
            
            # Check for outage conditions
            if state.risk_level == RiskLevel.RED:
                if load > self.capacity_mw:
                    # Blackout event
                    shed_amount = load - self.capacity_mw * 0.9
                    outage_events.append({
                        'timestamp': timestamp,
                        'type': 'blackout_risk',
                        'load_mw': load,
                        'excess_mw': load - self.capacity_mw
                    })
                    
                    # Load shedding response
                    shedding_events.append({
                        'timestamp': timestamp,
                        'shed_mw': shed_amount,
                        'duration_hours': 1,
                        'affected_zones': self._calculate_affected_zones(shed_amount)
                    })
                    
                    unserved_energy += shed_amount
        
        # Compile results
        result = SimulationResult(
            scenario_name=params.get('name', 'Default Scenario'),
            duration_hours=len(load_forecast),
            timestamps=timestamps,
            loads=loads,
            risk_levels=risk_levels,
            outage_events=outage_events,
            load_shedding_events=shedding_events,
            peak_load=max(loads),
            min_reserve_margin=min(s.reserve_margin for s in self.history[-len(loads):]),
            total_unserved_energy_mwh=unserved_energy,
            mitigation_actions=self._generate_mitigation_actions(risk_levels, params)
        )
        
        return result
    
    def _calculate_affected_zones(self, shed_mw: float) -> List[str]:
        """Calculate which zones would be affected by load shedding."""
        zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
        n_zones = min(len(zones), int(np.ceil(shed_mw / 200)))
        return zones[:n_zones]
    
    def _generate_mitigation_actions(
        self,
        risk_levels: List[RiskLevel],
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended mitigation actions based on risk analysis."""
        actions = []
        
        # Count risk level occurrences
        red_count = sum(1 for r in risk_levels if r == RiskLevel.RED)
        orange_count = sum(1 for r in risk_levels if r == RiskLevel.ORANGE)
        
        if red_count > 0:
            actions.extend([
                "URGENT: Activate demand response programs immediately",
                "Prepare rotating outage schedules",
                "Request emergency power imports from neighboring grids",
                "Dispatch all available peaking units"
            ])
        
        if orange_count > len(risk_levels) * 0.1:
            actions.extend([
                "Issue conservation appeal to customers",
                "Pre-position mobile generation units",
                "Increase monitoring frequency to 5-minute intervals"
            ])
        
        if params.get('heat_wave'):
            actions.append("Coordinate with hospitals and critical facilities for backup power")
        
        if params.get('renewable_fraction', 0) > 0.3:
            actions.append("Ensure battery storage systems are fully charged before peak")
        
        if not actions:
            actions.append("Continue normal operations with standard monitoring")
        
        return actions
    
    def simulate_breaker_response(
        self,
        fault_location: str,
        fault_current_ka: float
    ) -> Dict[str, Any]:
        """
        Simulate circuit breaker response to a fault.
        
        Args:
            fault_location: Location identifier
            fault_current_ka: Fault current in kA
        
        Returns:
            Breaker response details
        """
        # Simplified breaker response model
        base_response_ms = self.BREAKER_RESPONSE_DELAY_MS
        
        # Higher fault currents trigger faster response
        if fault_current_ka > 50:
            response_time = base_response_ms * 0.5
        elif fault_current_ka > 20:
            response_time = base_response_ms * 0.75
        else:
            response_time = base_response_ms
        
        # Cascading failure risk
        cascade_risk = min(0.9, fault_current_ka / 100)
        
        return {
            'location': fault_location,
            'fault_current_ka': fault_current_ka,
            'response_time_ms': response_time,
            'breaker_action': 'trip',
            'cascade_risk': cascade_risk,
            'isolation_successful': cascade_risk < 0.5,
            'affected_load_mw': fault_current_ka * 11  # Simplified calculation
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk levels from simulation history."""
        if not self.history:
            return {}
        
        risk_counts = {level: 0 for level in RiskLevel}
        for state in self.history:
            risk_counts[state.risk_level] += 1
        
        total = len(self.history)
        
        return {
            'total_periods': total,
            'risk_distribution': {
                level.value: count / total * 100 
                for level, count in risk_counts.items()
            },
            'peak_load_mw': max(s.current_load_mw for s in self.history),
            'min_reserve_margin': min(s.reserve_margin for s in self.history),
            'periods_at_risk': sum(1 for s in self.history 
                                   if s.risk_level in [RiskLevel.ORANGE, RiskLevel.RED])
        }
    
    def reset(self):
        """Reset simulator state."""
        self.current_state = None
        self.history = []
        self.outage_log = []


