import rootpy.io
from rootpy.tree import Tree

f = rootpy.io.open('ttree.root')
tree = f.get('ParTree_Postselect', ignore_unsupported=True)
tree.csv()#stream=open('test.csv', 'w'))

for event in tree:
    print event
