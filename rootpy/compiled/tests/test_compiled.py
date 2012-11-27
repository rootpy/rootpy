import rootpy.compiled as C

C.register_file("test.cxx", ["AnswerToLtUaE", "RootpyTestCompiled"])

def test_compiled():
    assert C.AnswerToLtUaE() == 42
    assert C.RootpyTestCompiled().blah() == 84

