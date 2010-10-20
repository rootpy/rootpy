    
def parse(string,objCuts=None):
    
    maps = []
    tuple = string.split(":")
    samples = tuple[0]
    jetType = "all"
    suffix = "test"
    weight = 0.
    format = "E0"
    legend = "P"
    marker = ""
    markercolour = ""
    fillcolour=""
    linecolour=""
    linestyle = ""
    label = ""
    cuts = ""
    fill = 0
    for parameter in tuple[1:]:
        if parameter.startswith("jet"):
            jetType = parameter.split('=')[1]
        elif parameter.startswith("suffix"):
            suffix = parameter.split('=')[1]
        elif parameter.startswith("weight"):
            weight = float(parameter.split('=')[1])
        elif parameter.startswith("format"):
            format = parameter.split('=')[1].replace("-"," ")
        elif parameter.startswith("legend"):
            legend = parameter.split('=')[1].replace("-"," ")
        elif parameter.startswith("fillcolour"):
            fillcolour = parameter.split('=')[1]
        elif parameter.startswith("linecolour"):
            linecolour = parameter.split('=')[1]
        elif parameter.startswith("line"):
            linestyle = parameter.split('=')[1]
        elif parameter.startswith("markercolour"):
            markercolour = parameter.split('=')[1]
        elif parameter.startswith("marker"):
            marker = parameter.split('=')[1]
        elif parameter.startswith("label"):
            if label != "":
                label += " "
            label += "=".join(parameter.split('=')[1:])
        elif parameter.startswith("cuts"):
            cuts = parameter[5:]
        elif parameter.startswith("fill"):
            fill = int(parameter.split('=')[1])
    for name in samples.split("+"):
        maps.append({"name":name,"weight":weight,"jetType":jetType,"suffix":suffix,"format":format,"legend":legend,"marker":marker,"fillcolour":fillcolour,"linecolour":linecolour,"linestyle":linestyle,"fill":fill,"label":label,"cuts":cuts,"objCuts":objCuts})
    return maps

def getLabel(sample):
    
    label = sample["name"]
    if sample["jetType"] != "all":
        label += " j="+sample["jetType"]
    return label
        
        
