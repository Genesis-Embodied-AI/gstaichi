from .enums import Layout


class NdarrayTypeMetadata:
    def __init__(self, element_type, shape=None, needs_grad=False):
        self.element_type = element_type
        self.shape = shape
        self.layout = Layout.AOS
        self.needs_grad = needs_grad
