# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import rootpy.compiled as C

C.register_file("test_compiled.cxx",
                ["AnswerToLtUaE", "RootpyTestCompiled"])

C.register_code("""

    #include <string>
    std::string _rootpy_test() { return "Hello, world"; }

""", "_rootpy_test".split())


def test_compiled():
    assert C.AnswerToLtUaE() == 42
    assert C.RootpyTestCompiled().blah() == 84
    assert C._rootpy_test() == "Hello, world"
