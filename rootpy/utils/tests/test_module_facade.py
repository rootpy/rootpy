import rootpy.utils.tests.facade_example as F

def test_module_facade():
    assert F.hello == "hello"
    assert F.something == "something"
    assert F.attach_thing("thing") == "thing"
    assert F.class_level == "class_level"
    assert "something" in dir(F)
    assert F["item"] == "item"
    assert F.module_level_constant == "MODULE_LEVEL_CONSTANT"
    assert "module_level_constant" in dir(F)
    assert F.module_level_function("a") == "a"
    assert "module_level_function" in dir(F)

def test_internal_facade():
    from rootpy.utils.tests.facade_example.internal import hello
    assert hello == "hello"
    assert F.internal.hello == "hello"
    assert dir(hello)

