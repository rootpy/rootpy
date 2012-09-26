from rootpy.tree import Tree
import rootpy.io

f = rootpy.io.open('ttree.root')
tree = f.get('ParTree_Postselect', ignore_unsupported=True)
tree.csv()

for event in tree:
    print event
