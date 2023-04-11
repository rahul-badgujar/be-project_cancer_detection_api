def parse_bool(value, default_to: bool) -> bool:
    if type(value) == type(True):
        return value
    value = str(value).lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise Exception(f"invalid value for type boolean: {value}")
