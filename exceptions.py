"""
exceptions.py: Different exceptions created for poextiment.py
"""


class JsonKeyNotFound(Exception):
    def __init__(self, keyname):
        super().__init__('ERROR: Provided key (' + str(keyname) + ') for json file not found')

class InvalidFiletype(Exception):
    def __init__(self, filetype):
        error = 'ERROR: provided filetype \"' + str(filetype) + '\", when only \'txt\' and \'json\' are supported. If a different filetype, you may provide your own parser.'
        super().__init__(error)

class SelfParseNotString(Exception):
    def __init__(self):
        super().__init__('ERROR: Result of user-inputted parser was not a string. Must be a string to undergo processing.')