from xml.dom import minidom, Node

def load_variables(filename):

    file = open(filename,'r')
    doc = minidom.parse(file)
    file.close()
    variablesNode = doc.getElementsByTagName("variables")[0]
    variableNodes = variablesNode.getElementsByTagName("variable")
    map = {}
    for node in variableNodes:
        varDict = {}
        varDict["type"] = str(node.attributes["type"].value)
        extraAttrNodes = node.childNodes
        for extNode in extraAttrNodes:
            if extNode.nodeType == Node.ELEMENT_NODE:
                varDict[str(extNode.nodeName)] = str(extNode.attributes["value"].value)
        map[str(node.attributes["name"].value)] = varDict
    return map
