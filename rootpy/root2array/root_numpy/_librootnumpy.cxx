#include <Python.h>
#include <string>
#include <iostream>
#include <TTree.h>
#include <TFile.h>
#include <TChain.h>
#include <TLeaf.h>
#include <map>
#include <numpy/arrayobject.h>
#include <cassert>
#include <set>
#include <iomanip>
#include <fstream>
#include <TreeStructure.h>
#define RNDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << s << std::endl;

#define HAVE_COBJ ( (PY_VERSION_HEX <  0x03020000) )
#define HAVE_CAPSULE ( ((PY_VERSION_HEX >=  0x02070000) && (PY_VERSION_HEX <  0x03000000)) || (PY_VERSION_HEX >=  0x03010000) )
//return all branch name from tree
std::vector<std::string> get_branchnames(TTree& tree){
    TObjArray* branches = (TObjArray*)tree.GetListOfBranches();
    std::vector<std::string> ret;
    for(int i=0;i<branches->GetEntries();++i){
        TBranch* thisBranch = dynamic_cast<TBranch*>(branches->At(i));
        const char* tmp = thisBranch->GetName();
        std::string str(tmp);
        ret.push_back(tmp);
    }
    return ret;
}

//vector unique with order preservation(just get the first one) O(nlogn)
std::vector<std::string> vector_unique(const std::vector<std::string>& org){
    using namespace std;
    set<string> myset;
    myset.insert(org.begin(),org.end());
    vector<string> ret;
    for(int i=0;i<org.size();i++){
        set<string>::iterator it = myset.find(org[i]);
        if(it!=myset.end()){
            myset.erase(it);
            ret.push_back(org[i]);
        }
    }
    return ret;
}

//convert list of string to vector of string
//if los is just a string vos will be filled with that string
//if los is null or PyNone it do nothing to vos and return OK;
int los2vos(PyObject* los, std::vector<std::string>& vos){
    int ret=1;
    if(los==NULL){
        //do nothing
    }
    else if(los==Py_None){
        //do nothing
    }
    else if(PyString_Check(los)){//passing string put that in to vector
        char* tmp = PyString_AsString(los);
        vos.push_back(tmp);
    }else if(PyList_Check(los)){//an actual list of string
        int len = PyList_Size(los);
        for(int i=0;i<len;i++){
            PyObject* s = PyList_GetItem(los,i);
            if(!s){return NULL;}
            char* tmp = PyString_AsString(s);
            if(!tmp){return NULL;}
            std::string str(tmp);
            vos.push_back(tmp);
        }
    }else{
        ret=NULL;
    }
    return ret;
}

bool file_exists(std::string fname){
    std::ifstream my_file(fname.c_str());
    return my_file.good();
}

bool has_wildcard(std::string fname){
    return fname.find("*") != std::string::npos;
}
/**
* loadTTree from PyObject if fnames is list of string then it load every pattern in fnames
* if fnames is just a string then it load just one
* caller is responsible to delete the return value
* null is return if fnames is not a valid python object(either list of string or string)
*/
TTree* loadTree(PyObject* fnames, const char* treename){
    using namespace std;
    vector<string> vs;
    if(!los2vos(fnames,vs)){return NULL;}
    TChain* chain = new TChain(treename);
    int total= 0;
    for(int i=0;i<vs.size();++i){
        //check if it's not pattern(chain->Add always return 1 (no idea what's the rationale))
        //if fname doesn't contain wildcard
        if(!has_wildcard(vs[i]) && !file_exists(vs[i])){//no wildcard and file doesn't exists
            PyErr_SetString(PyExc_IOError,("File "+vs[i]+" not found or not readable").c_str());
            delete chain;
            return NULL;
        }
        int fileadded = chain->Add(vs[i].c_str());
        if(fileadded==0){
            std::cerr << "Warning: pattern " << vs[i] << " does not match any file" << std::endl;
        }
        total+=fileadded;
    }
    if(total==0){
        delete chain;
        PyErr_SetString(PyExc_IOError,"None of the pattern match any file. May be a typo?");
        return NULL;
    }
    //check if the tree exists
    if(chain->GetListOfBranches()==0){
        delete chain;
        string msg = "";
        msg = msg + "Tree " + treename + " not found. Try list_trees(fname).";
        PyErr_SetString(PyExc_IOError,msg.c_str());
        return NULL;
    }
    return dynamic_cast<TTree*>(chain);
}

PyObject* root2array_helper(TTree& tree, PyObject* branches_){
    using namespace std;
    vector<string> branches;
    if(!los2vos(branches_,branches)){return NULL;}
    if(branches.size()==0){branches=branch_names(&tree);}

    TreeStructure t(&tree,branches);
    PyObject* array = NULL;
    if(t.good){
        array = t.build_array();
    }else{
        return NULL;
    }
    return array;
}

PyObject* root2array(PyObject *self, PyObject *args, PyObject* keywords){
    using namespace std;
    PyObject* fnames=NULL;
    char* treename_;
    PyObject* branches_=NULL;
    PyObject* array=NULL;
    static const char* keywordslist[] = {"fname","treename","branches",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,keywords,"Os|O",const_cast<char **>(keywordslist),&fnames,&treename_,&branches_)){
        return NULL;
    }

    TTree* chain = loadTree(fnames,treename_);
    if(!chain){return NULL;}

    //int numEntries = chain->GetEntries();

    array = root2array_helper(*chain,branches_);
    delete chain;
    return array;
}

//calling these like this: root2array_from_*
// if 'AsCapsule' in dir(ROOT) call the capsule one (doesn't exist yet as of this writing)
// otherwise call the cobj one
// capsule one is provided

//these people have cobject
#if HAVE_COBJ
PyObject* root2array_from_cobj(PyObject *self, PyObject *args, PyObject* keywords){
    using namespace std;
    TTree* tree=NULL;
    PyObject* tree_=NULL;
    PyObject* branches_=NULL;
    PyObject* array=NULL;
    static const char* keywordslist[] = {"tree","branches",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,keywords,"O|O",const_cast<char **>(keywordslist),&tree_,&branches_)){
        return NULL;
    }

    if(!PyCObject_Check(tree_)){return NULL;}
    //this is not safe so be sure to know what you are doing type check in python first
    //this is a c++ limitation because void* have no vtable so dynamic cast doesn't work
    TTree* chain = static_cast<TTree*>(PyCObject_AsVoidPtr(tree_));

    if(!chain){
        PyErr_SetString(PyExc_TypeError,"Unable to convert tree to TTree*");
        return NULL;
    }

    //int numEntries = chain->GetEntries();

    return root2array_helper(*chain,branches_);
}
#endif

//and these people have capsule
#if HAVE_CAPSULE
PyObject* root2array_from_capsule(PyObject *self, PyObject *args, PyObject* keywords){
    using namespace std;
    TTree* tree=NULL;
    PyObject* tree_=NULL;
    PyObject* branches_=NULL;
    PyObject* array=NULL;
    static const char* keywordslist[] = {"tree","branches",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,keywords,"O|O",const_cast<char **>(keywordslist),&tree_,&branches_)){
        return NULL;
    }

    if(!PyCapsule_CheckExact(tree_)){return NULL;}
    //this is not safe so be sure to know what you are doing type check in python first
    //this is a c++ limitation because void* have no vtable so dynamic cast doesn't work
    TTree* chain = static_cast<TTree*>(PyCapsule_GetPointer(tree_,NULL));
    if(!chain){
        PyErr_SetString(PyExc_TypeError,"Unable to convert tree to TTree*");
        return NULL;
    }

    //int numEntries = chain->GetEntries();

    return root2array_helper(*chain,branches_);
}
#endif

PyObject* test(PyObject *self, PyObject *args){
    using namespace std;
    cout << sizeof(Int_t) << endl;
    cout << sizeof(Bool_t) << endl;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* list_trees(PyObject* self, PyObject* arg){
    char* cfname;
    if(!PyArg_ParseTuple(arg,"s",&cfname)){
        return NULL;
    }

    TFile f(cfname);
    if(f.IsZombie()){
        std::string msg;
        msg += "Unable to open root file ";
        msg += cfname;
        PyErr_SetString(PyExc_IOError,msg.c_str());
        return NULL;
    }

    TList* list = f.GetListOfKeys();
    TIter next(list);
    PyObject* ret = PyList_New(0);
    while(TObject* key = next()){
        TObject* obj = f.Get(key->GetName());
        if(strncmp(obj->ClassName(),"TTree",10)==0){
            PyObject* tmp = PyString_FromString(key->GetName());
            PyList_Append(ret,tmp);
        }
    }

    return ret;
}

PyObject* list_branches(PyObject* self, PyObject* arg){
    char* cfname;
    char* ctname;
    if(!PyArg_ParseTuple(arg,"ss",&cfname,&ctname)){
        return NULL;
    }

    TFile f(cfname);
    if(f.IsZombie()){
        std::string msg;
        msg += "Unable to open file: ";
        msg += cfname;
        PyErr_SetString(PyExc_IOError,msg.c_str());
        return NULL;
    }

    TTree* tree = dynamic_cast<TTree*>(f.Get(ctname));
    if(tree==0){
        std::string msg;
        msg += "Unable to open tree: ";
        msg += ctname;
        msg +="from ";
        msg +=cfname;
        PyErr_SetString(PyExc_IOError,msg.c_str());
        return NULL;
    }
    PyObject* ret = PyList_New(0);
    TObjArray* lob = tree->GetListOfBranches();
    for(int i=0;i<lob->GetEntries();++i){
        TObject* obj = lob->At(i);
        TBranch* b = dynamic_cast<TBranch*>(obj);
        PyObject* tmp = PyString_FromString(b->GetName());
        PyList_Append(ret,tmp);
    }
    return ret;
}

static PyMethodDef methods[] = {
    {"test",test,METH_VARARGS,""},
    {"root2array",  (PyCFunction)root2array, METH_VARARGS|METH_KEYWORDS,
    "root2array(fnames,treename,branches=None)\n"
    "convert tree treename in root files specified in fnames to numpy structured array\n"
    "------------------\n"
    "return numpy array\n"
    "fnames: list of string or string. Root file name patterns. Anything that works with TChain.Add is accepted\n"
    "treename: name of tree to convert to numpy array\n"
    "branches(optional): list of string for branch name to be extracted from tree.\n"
    "\tIf branches is not specified or is none or is empty, all from the first treebranches are extracted\n"
    "\tIf branches contains duplicate branches, only the first one is used.\n"
    "\n"
    "Caveat: This should not matter for most use cases. But, due to the way TChain works, if the trees specified in the input files have different\n"
    "structures, only the branch in the first tree will be automatically extracted. You can work around this by either reordering the input file or\n"
    "specify the branches manually.\n"
    "------------------\n"
    "Ex:\n"
    "root2array('a.root','mytree')#read all branches from tree named mytree from a.root\n\n"
    "root2array('a*.root','mytree')#read all branches from tree named mytree from a*.root\n\n"
    "root2array(['a*.root','b*.root'],'mytree')#read all branches from tree named mytree from a*.root and b*.root\n\n"
    "root2array('a.root','mytree','x')#read branch x from tree named mytree from a.root(useful if memory usage matters)\n\n"
    "root2array('a.root','mytree',['x','y'])#read branch x and y from tree named mytree from a.root\n"
    },
    #if HAVE_COBJ
    {"root2array_from_cobj",  (PyCFunction)root2array_from_cobj, METH_VARARGS|METH_KEYWORDS,
    "root2array_from_cobj(PyCObject tree, branches=None)\n"
    "convert TTree in form of PyCObject to structured array. branches accept many form of arguments. See root2array for details \n"
    },
    #endif
    #if HAVE_CAPSULE
    {"root2array_from_capsule",  (PyCFunction)root2array_from_capsule, METH_VARARGS|METH_KEYWORDS,
    "root2array_from_capsule(PyCObject tree, branches=None)\n"
    "convert TTree in form of PyCapsule to structured array. branches accept many form of arguments. See root2array for details \n"
    },
    #endif
    {"list_branches",  (PyCFunction)list_branches, METH_VARARGS,""},
    {"list_trees",  (PyCFunction)list_trees, METH_VARARGS,""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

void cleanup(){
    //do nothing
}

PyMODINIT_FUNC
init_librootnumpy(void)
{
    import_array();
    init_roottypemap();
    (void) Py_InitModule("_librootnumpy", methods);
    //import_array();
}
