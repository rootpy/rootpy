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

#define RNDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << s << std::endl;

#define HAVE_COBJ ( (PY_VERSION_HEX <  0x03020000) )
#define HAVE_CAPSULE ( ((PY_VERSION_HEX >=  0x02070000) && (PY_VERSION_HEX <  0x03000000)) || (PY_VERSION_HEX >=  0x03010000) )
struct TypeInfo{
    PyObject* nptype;
    int size;//in bytes
    TypeInfo(const TypeInfo& t):nptype(t.nptype),size(t.size){Py_INCREF(nptype);}
    TypeInfo(const char* nptype, int size):nptype(PyString_FromString(nptype)),size(size){};
    ~TypeInfo(){Py_DECREF(nptype);}
};

static std::map<std::string, TypeInfo> root_typemap;
//map roottype string to TypeInfo Object
void init_roottypemap(){
    using std::make_pair;
    //TODO: correct this one so it doesn't depend on system
    // from TTree doc
    // - C : a character string terminated by the 0 character
    // - B : an 8 bit signed integer (Char_t)
    // - b : an 8 bit unsigned integer (UChar_t)
    // - S : a 16 bit signed integer (Short_t)
    // - s : a 16 bit unsigned integer (UShort_t)
    // - I : a 32 bit signed integer (Int_t)
    // - i : a 32 bit unsigned integer (UInt_t)
    // - F : a 32 bit floating point (Float_t)
    // - D : a 64 bit floating point (Double_t)
    // - L : a 64 bit signed integer (Long64_t)
    // - l : a 64 bit unsigned integer (ULong64_t)
    // - O : [the letter 'o', not a zero] a boolean (Bool_t)
    // from numericdtype.py
    // # b -> boolean
    // # u -> unsigned integer
    // # i -> signed integer
    // # f -> floating point
    // # c -> complex
    // # M -> datetime
    // # m -> timedelta
    // # S -> string
    // # U -> Unicode string
    // # V -> record
    // # O -> Python object

    root_typemap.insert(std::make_pair("Char_t",TypeInfo("i1",1)));
    root_typemap.insert(std::make_pair("UChar_t",TypeInfo("u1",1)));

    root_typemap.insert(std::make_pair("Short_t",TypeInfo("i2",2)));
    root_typemap.insert(std::make_pair("UShort_t",TypeInfo("u2",2)));

    root_typemap.insert(std::make_pair("Int_t",TypeInfo("i4",4)));
    root_typemap.insert(std::make_pair("UInt_t",TypeInfo("u4",4)));

    root_typemap.insert(std::make_pair("Float_t",TypeInfo("f4",4)));

    root_typemap.insert(std::make_pair("Double_t",TypeInfo("f8",8)));

    root_typemap.insert(std::make_pair("Long64_t",TypeInfo("i8",8)));
    root_typemap.insert(std::make_pair("ULong64_t",TypeInfo("u8",8)));

    root_typemap.insert(std::make_pair("Bool_t",TypeInfo("bool",1)));
}

//convert roottype string to typeinfo
TypeInfo* convert_roottype(const std::string& t){
    std::map<std::string, TypeInfo>::iterator it = root_typemap.find(t);
    if(it==root_typemap.end()){
        //std::string msg = "Unknown root type: "+t;
        //PyErr_SetString(PyExc_RuntimeError,msg.c_str());
        std::cerr << "Warning: unknown root type: " << t << " skip " << std::endl;
        return NULL;
    }
    return &(it->second);
}

struct LeafInfo{
    std::string name;
    TypeInfo* type;
    std::string root_type;
    char payload[64];//reserve for payload
    LeafInfo():name(),type(){}
    LeafInfo(const std::string& name,const std::string& root_type):name(name),root_type(root_type),type(0){};
    std::string repr(){
        return name + "("+root_type+")";
    }
};

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

//get list of leafinfo from tree
//if branches is not empty, only the branches specified in branches will be used
//otherwise it will automatically list all the branches of the first tree in chain
//caller is responsible to delete each LeafInfo
//return NULL when it fails
int get_leafinfo(TTree& tree,const std::vector<std::string>& branches, std::vector<LeafInfo*>& ret){

    using namespace std;
    vector<string> branchNames;
    //branch is not specified
    if(branches.size()==0) branchNames = get_branchnames(tree);
    else branchNames = vector_unique(branches); //make sure it's unique

    //for each branch figure out the type and construct leafinfo
    for(int i=0;i<branchNames.size();i++){
        TBranch* thisBranch = dynamic_cast<TBranch*>(tree.GetBranch(branchNames[i].c_str()));
        if(thisBranch==0){
            PyErr_SetString(PyExc_ValueError,("Branch "+branchNames[i]+" doesn't exist.").c_str());
            return NULL;
        }
        std::string roottype("Float_t");
        TypeInfo* ti = NULL;
        bool should_add_branch = true;
        if(thisBranch!=0){
            TObjArray* leaves = thisBranch->GetListOfLeaves();
            assert(leaves!=0);
            TLeaf* thisLeaf = dynamic_cast<TLeaf*>(leaves->At(0));
            assert(thisLeaf!=0);
            int ncount=0;
            TLeaf* count_leaf = thisLeaf->GetLeafCounter(ncount);
            if(!(ncount==1 and count_leaf==NULL)){//some sort of array
                std::cerr << "Warning: skipping array branch " << branchNames[i] << std::endl;
                should_add_branch=false;
            }else{
                roottype = thisLeaf->GetTypeName();
                ti = convert_roottype(roottype);
                if(ti==NULL) should_add_branch=false;
            }
        }else{
            std::cerr << "Warning: branch not found in the first tree(assume type of Float_t)" << branchNames[i] << std::endl;
        }
        //need to set branch address at tree level because TChain will fail if branch is set at the first tree
        if(should_add_branch){
            LeafInfo* li = new LeafInfo(thisBranch->GetName(),roottype);
            li->type=ti;
            tree.SetBranchAddress(thisBranch->GetName(),&(li->payload));
            ret.push_back(li);
        }
    }
    return 1;
}
//helper function for building numpy descr
//build == transfer ref ownershp to caller
PyObject* build_numpy_descr(const std::vector<LeafInfo*>& lis){

    PyObject* mylist = PyList_New(0);
    for(int i=0;i<lis.size();++i){
        PyObject* pyname = PyString_FromString(lis[i]->name.c_str());
        PyObject* pytype = lis[i]->type->nptype;

        Py_INCREF(pytype);
        PyObject* nt_tuple = PyTuple_New(2);
        PyTuple_SetItem(nt_tuple,0,pyname);
        PyTuple_SetItem(nt_tuple,1,pytype);

        PyList_Append(mylist,nt_tuple);
    }
    return mylist;
}

//convert all leaf specified in lis to numpy structured array
PyObject* build_array(TTree& chain, std::vector<LeafInfo*>& lis){
    using namespace std;
    int numEntries = chain.GetEntries();
    PyObject* numpy_descr = build_numpy_descr(lis);
    if(numpy_descr==0){return NULL;}

    //build the array
    PyArray_Descr* descr;
    PyArray_DescrConverter(numpy_descr,&descr);
    Py_DECREF(numpy_descr);

    npy_intp dims[1];
    dims[0]=numEntries;

    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromDescr(1,dims,descr);

    //assume numpy array is contiguous
    char* current = (char*)PyArray_DATA(array);

    //now put stuff in array
    for(int iEntry=0;iEntry<numEntries;++iEntry){
        chain.GetEntry(iEntry);
        current = (char*)PyArray_GETPTR1(array, iEntry);
        for(int ileaf=0;ileaf<lis.size();++ileaf){
            int size = lis[ileaf]->type->size;
            // cout << lis[ileaf]->name << " " << lis[ileaf]->root_type << endl;
            //
            // cout << *(int*)(lis[ileaf]->payload) << " " << *(float*)(lis[ileaf]->payload) << endl;
            // cout << lis[ileaf]->name << "("<< iEntry << ")";
            // cout << " current: 0x";
            // cout << std::hex << (long)(current) << std::dec ;
            // cout << " size:" << size << " " ;
            // cout << array->strides[0] << endl;
            memcpy((char*)current,(char*)lis[ileaf]->payload,size);
            current+=size;

        }
    }
    return (PyObject*)array;
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
        //check if it's a patter(chain->Add always return 1 (no idea what's the rationale))
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
    return dynamic_cast<TTree*>(chain);
}

PyObject* root2array_helper(TTree& tree, PyObject* branches_){
    using namespace std;
    vector<string> branches;
    if(!los2vos(branches_,branches)){return NULL;}
    vector<LeafInfo*> lis;
    int flag = get_leafinfo(tree,branches,lis);
    PyObject* array = NULL;
    if(flag!=0){
        array = build_array(tree, lis);
    }
    for(int i=0;i<lis.size();i++){delete lis[i];}
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
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

void cleanup(){
    //do nothing
}

PyObject* list_branches(){

}

PyObject* list_tree(){

}

PyMODINIT_FUNC
init_librootnumpy(void)
{
    import_array();
    init_roottypemap();
    (void) Py_InitModule("_librootnumpy", methods);
    //import_array();
}
