from pint import UnitRegistry

# Create a unit registry
ureg = UnitRegistry()

def convert_unit(unit):
    try:
        # Parse the unit
        parsed_unit = ureg.parse_expression(unit)
        # Return the full unit name
        return parsed_unit.units
    except Exception as e:
        print(f"Error converting unit '{unit}': {e}")
        return unit

# Example usage
print(convert_unit('ft'))  # Output: foot
print(convert_unit('cm'))  # Output: centimeter
