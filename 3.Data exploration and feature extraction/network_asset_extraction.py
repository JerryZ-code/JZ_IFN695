import numpy as np

def extract_asset_type(asset_name):
    asset_name = asset_name.lower()  # make matching easier

    if 'transformer' in asset_name:
        return 'Transformer'
    elif 'line' in asset_name:
        return 'Line'
    elif 'cable' in asset_name:
        return 'Cable'
    elif 'switch' in asset_name:
        return 'Switch'
    elif 'reactor' in asset_name:
        return 'Reactor'
    elif 'capacitor' in asset_name:
        return 'Capacitor'
    elif 'bus' in asset_name:
        return 'Bus'
    elif 'breaker' in asset_name:
        return 'Breaker'
    elif 'isolator' in asset_name:
        return 'Isolator'
    elif 'bay' in asset_name:
        return 'Bay'
    elif '!!! new hio' in asset_name:
        return np.nan  # ignore HIO notifications
    else:
        return 'Other'  # fallback for unknown types