"""Custom exceptions for PyFVCOM2."""

__all__ = [
    "PyFVCOM2RuntimeError",
    "PyFVCOM2AttributeError",
    "PyFVCOM2ValueError",
    "PyFVCOM2TypeError",
    "PyFVCOM2FileNotFoundError",
]


class PyFVCOM2Exception(Exception):
    pass


class PyFVCOM2RuntimeError(PyFVCOM2Exception):
    pass


class PyFVCOM2AttributeError(PyFVCOM2Exception):
    pass


class PyFVCOM2ValueError(PyFVCOM2Exception):
    pass


class PyFVCOM2TypeError(PyFVCOM2Exception):
    pass


class PyFVCOM2FileNotFoundError(PyFVCOM2Exception):
    pass
