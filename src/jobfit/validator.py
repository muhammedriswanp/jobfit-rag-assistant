import re

def validate_json(output, expected_schema):
    for key, expected_type in expected_schema.items():
        if key not in output:
            return False
        if expected_type and not isinstance(output[key], expected_type):
            return False
    return True
