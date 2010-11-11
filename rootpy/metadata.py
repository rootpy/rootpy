"""
Load metadata about datasets from xml
"""
from xml.dom import minidom, Node

def load_variables(string):
    """
    Retrieve the variable information
    """
    try:
        # attempt to read as filename
        doc = minidom.parse(string)
    except IOError:
        # the string is the xml?
        doc = minidom.parseString(string)
    variables_node = doc.getElementsByTagName("variables")[0]
    variable_nodes = variables_node.getElementsByTagName("variable")
    variables = {}
    for node in variable_nodes:
        var_dict = {}
        var_dict["type"] = str(node.attributes["type"].value)
        extra_attr_nodes = node.childNodes
        for ext_node in extra_attr_nodes:
            if ext_node.nodeType == Node.ELEMENT_NODE:
                var_dict[str(ext_node.nodeName)] = \
                    str(ext_node.attributes["value"].value)
        variables[str(node.attributes["name"].value)] = var_dict
    return variables
