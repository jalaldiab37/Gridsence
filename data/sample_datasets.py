"""
Sample Dataset Information
Details about available public datasets for electrical load forecasting.
"""

DATASETS = {
    "opsd_europe": {
        "name": "Open Power System Data (OPSD)",
        "url": "https://data.open-power-system-data.org/",
        "description": "European electricity load, generation, and prices",
        "countries": ["DE", "GB", "FR", "ES", "IT", "AT", "BE", "NL"],
        "resolution": "15-minute and hourly",
        "years": "2006-2020+",
        "features": [
            "Total load",
            "Generation by type",
            "Cross-border flows",
            "Day-ahead prices"
        ],
        "download_cmd": "wget https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
    },
    
    "ercot_texas": {
        "name": "ERCOT (Electric Reliability Council of Texas)",
        "url": "https://www.ercot.com/gridinfo/load/load_hist",
        "description": "Texas grid load and generation data",
        "regions": ["COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST"],
        "resolution": "Hourly",
        "years": "2000-present",
        "features": [
            "Actual system load",
            "Load forecast",
            "Wind/solar generation",
            "Generation by fuel type"
        ]
    },
    
    "ieso_ontario": {
        "name": "IESO (Independent Electricity System Operator)",
        "url": "https://www.ieso.ca/power-data",
        "description": "Ontario, Canada electricity demand",
        "resolution": "Hourly",
        "years": "2002-present",
        "features": [
            "Market demand",
            "Ontario demand",
            "Hourly generation",
            "Intertie flows"
        ]
    },
    
    "eia_usa": {
        "name": "U.S. Energy Information Administration (EIA)",
        "url": "https://www.eia.gov/electricity/data.php",
        "description": "Comprehensive US electricity statistics",
        "resolution": "Monthly, some hourly",
        "coverage": "All US states and regions",
        "features": [
            "Sales and revenue",
            "Generation by state",
            "Fuel consumption",
            "Retail prices"
        ]
    },
    
    "kaggle_household": {
        "name": "Individual Household Electric Power Consumption",
        "url": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
        "description": "Single household consumption data (UCI ML Repository)",
        "resolution": "1-minute",
        "duration": "4 years",
        "features": [
            "Global active power",
            "Global reactive power",
            "Voltage",
            "Global intensity",
            "Sub-metering (kitchen, laundry, water heater/AC)"
        ]
    },
    
    "kaggle_american_electric": {
        "name": "Hourly Energy Consumption",
        "url": "https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption",
        "description": "PJM hourly energy consumption data",
        "resolution": "Hourly",
        "regions": ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI", "PJME", "PJMW"],
        "years": "2002-2018",
        "features": [
            "Megawatt consumption by region"
        ]
    },
    
    "entso_e_europe": {
        "name": "ENTSO-E Transparency Platform",
        "url": "https://transparency.entsoe.eu/",
        "description": "European Network of Transmission System Operators",
        "coverage": "36 European countries",
        "resolution": "15-minute to hourly",
        "features": [
            "Total load",
            "Day-ahead forecasts",
            "Generation per type",
            "Cross-border physical flows",
            "Installed generation capacity"
        ]
    }
}


def list_datasets():
    """Print available datasets."""
    print("=" * 60)
    print("AVAILABLE ELECTRICAL LOAD DATASETS")
    print("=" * 60)
    
    for key, info in DATASETS.items():
        print(f"\n📊 {info['name']}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")
        if 'resolution' in info:
            print(f"   Resolution: {info['resolution']}")
        print()


def get_dataset_info(name: str) -> dict:
    """Get information about a specific dataset."""
    return DATASETS.get(name, {})


if __name__ == "__main__":
    list_datasets()


