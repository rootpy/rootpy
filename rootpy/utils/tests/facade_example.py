from rootpy.utils.module_facade import Facade


def module_level_function(what):
    return what

module_level_constant = "MODULE_LEVEL_CONSTANT"


@Facade(__name__, expose_internal=False, submodule=True)
class internal(object):
    @property
    def hello(self):
        return "hello"


@Facade(__name__, expose_internal=True)
class ExampleModuleFacade(object):
    class_level = "class_level"

    @property
    def hello(self):
        return "hello"

    def __getattr__(self, key):
        if key == "something":
            return "something"

    def __getitem__(self, key):
        return key

    def __dir__(self):
        return ["something"]

    def attach_thing(self, param):
        return param
