#ifndef __TREESTRUCTURE__H
#define __TREESTRUCTURE__H
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
#include <cstdlib>
#include <cstdio>
#define RNDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << s << std::endl;
#define RNHEXDEBUG(s) std::cout << "DEBUG: " << __FILE__ << "(" <<__LINE__ << ") " << #s << " = " << std::hex << s << std::dec << std::endl;
using namespace std;

struct TypeInfo{
    string nptype;
    int size;//in bytes
    NPY_TYPES npt;
    TypeInfo(const TypeInfo& t):nptype(t.nptype),size(t.size),npt(t.npt){}
    TypeInfo(const std::string& nptype, int size, NPY_TYPES npt):nptype(nptype),size(size),npt(npt){}
    ~TypeInfo(){}
    void print(){
        cout << nptype << ":" << size;
    }
};

static std::map<std::string, TypeInfo> root_typemap;

inline void init_roottypemap(){
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
    
    root_typemap.insert(make_pair("Char_t",TypeInfo("i1",1,NPY_INT8)));
    root_typemap.insert(make_pair("UChar_t",TypeInfo("u1",1,NPY_UINT8)));    

    root_typemap.insert(make_pair("Short_t",TypeInfo("i2",2,NPY_INT16)));
    root_typemap.insert(make_pair("UShort_t",TypeInfo("u2",2,NPY_UINT16)));
        
    root_typemap.insert(make_pair("Int_t",TypeInfo("i4",4,NPY_INT32)));
    root_typemap.insert(make_pair("UInt_t",TypeInfo("u4",4,NPY_UINT32)));

    root_typemap.insert(make_pair("Float_t",TypeInfo("f4",4,NPY_FLOAT32)));
    
    root_typemap.insert(make_pair("Double_t",TypeInfo("f8",8,NPY_FLOAT64)));
    
    root_typemap.insert(make_pair("Long64_t",TypeInfo("i8",8,NPY_INT64)));
    root_typemap.insert(make_pair("ULong64_t",TypeInfo("u8",8,NPY_UINT64)));
    
    root_typemap.insert(make_pair("Bool_t",TypeInfo("bool",1,NPY_BOOL)));
}

//convert root type to numpy type
TypeInfo* rt2npt(const string& rt, bool must_found=false){
    std::map<std::string, TypeInfo>::iterator it;
    TypeInfo* ret=0;//float default
    it = root_typemap.find(rt);
    if(must_found){assert(it!=root_typemap.end());}
    if(it!=root_typemap.end()){
        ret = &(it->second);
    }
    return ret;
}

class ColumnDescr{
public:    
    enum ColType{SINGLE=1,FIXED=2,VARY=3};
    
    static ColumnDescr* build_single(const string& colname, const string& rttype,TTree* tree){
        ColumnDescr* ret = new ColumnDescr();
        ret->coltype = SINGLE;
        ret->colname = colname;
        ret->rttype = rttype;
        ret->lenname = "";
        ret->nptype = rt2npt(rttype,true);
        ret->payload = malloc(ret->nptype->size);
        ret->payloadsize = 1;
        ret->size_ele = ret->nptype->size;
        ret->static_asize = 1;
        ret->asize = &ret->static_asize;
        //need to do this otherwise it won't work with tchain
        //tbranch is reset everytime it change file
        //and get entry would work too
        tree->SetBranchAddress(colname.c_str(),ret->payload,&ret->branch);
        assert(ret->branch!=0);
        //ret->child = xxxx;//do nothing
        return ret;
    }
    
    static ColumnDescr* build_fixed(const string& colname, const string& rttype,int num_ele, TTree* tree){
        ColumnDescr* ret = new ColumnDescr();
        ret->coltype = FIXED;
        ret->colname = colname;
        ret->rttype = rttype;
        ret->lenname = "";
        ret->nptype = rt2npt(rttype,true);
        ret->payload = malloc(ret->nptype->size*num_ele);
        ret->payloadsize = num_ele;
        ret->size_ele = ret->nptype->size;
        ret->static_asize = num_ele;
        ret->asize = &ret->static_asize;
        //need to do this otherwise it won't work with tchain
        //tbranch is reset everytime it change file
        //and get entry would work too
        tree->SetBranchAddress(colname.c_str(),ret->payload,&ret->branch);
        assert(ret->branch!=0);
        //ret->child = xxxx;//do nothing
        return ret;
    }
    
    static ColumnDescr* build_vary(const string& colname, const string& rttype, const string& lenname, TTree* tree){
        int initial_size;
        ColumnDescr* ret = new ColumnDescr();
        ret->coltype = VARY;
        ret->colname = colname;
        ret->rttype = rttype;
        ret->lenname = lenname;
        ret->nptype = rt2npt(rttype,true);
        ret->payload = malloc(ret->nptype->size*initial_size);
        ret->payloadsize = initial_size;
        ret->size_ele = ret->nptype->size;
        ret->static_asize = -1;
        ret->asize = 0;//need to be set later
        //need to do this otherwise it won't work with tchain
        //tbranch is reset everytime it change file
        //and get entry would work too
        tree->SetBranchAddress(colname.c_str(),ret->payload,&ret->branch);
        assert(ret->branch!=0);
        //ret->child = xxxx;//do nothing
        return ret;
    }
    
    ~ColumnDescr(){
        free(payload);
    }
    
    //resize the payload if necessary
    void resize(int size){
        assert(coltype==VARY);//other people shouldn't call this
        if(size > payloadsize){
            payload = realloc(payload,size_ele*size);
            assert(payload!=0);//out of memory!!!
        }
    }
    //copy to numpy element destination
    //and return number of byte written
    int copy_to(void* destination){
        int ret;
        if(coltype==FIXED || coltype==SINGLE){
            ret = (*asize)*size_ele;
            assert(*asize>=0);
            memcpy(destination,payload,ret);
        }else{//variable length array
            //build a numpy array of the length and put pyobject there
            npy_intp dims[1];
            dims[0]=*asize;
            PyArrayObject* newobj = (PyArrayObject*)PyArray_EMPTY(1,dims,nptype->npt,0);
            assert(newobj!=0);
            assert(*asize>=0);//negative array length????
            memcpy(newobj->data,payload,size_ele*(*asize));
            memcpy(destination,&newobj,sizeof(PyArrayObject*));
            ret = sizeof(PyObject*);
        }
        return ret;
    }
    //convert to PyArray_Descr tuple
    PyObject* totuple(){
        //return ('col','f8')
        if(coltype==SINGLE){
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString(nptype->nptype.c_str());
            PyObject* nt_tuple = PyTuple_New(2);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);
            char* tmp = PyString_AsString(pytype);
            return nt_tuple;
        }else if(coltype==FIXED){//return ('col','f8',(10))
            PyObject* pyname = PyString_FromString(colname.c_str());
        
            PyObject* pytype = PyString_FromString(nptype->nptype.c_str());
            
            PyObject* subsize = PyTuple_New(1);
            PyObject* pysubsize = PyInt_FromLong(*asize);
            PyTuple_SetItem(subsize,0,pysubsize);
        
            PyObject* nt_tuple = PyTuple_New(3);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);
            PyTuple_SetItem(nt_tuple,2,subsize);
            
            return nt_tuple;
        }else if(coltype==VARY){//return ('col','object')
            PyObject* pyname = PyString_FromString(colname.c_str());
        
            PyObject* pytype = PyString_FromString("object");
            
            PyObject* nt_tuple = PyTuple_New(2);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);

            return nt_tuple;
        }else{
            assert(false);//shouldn't reach here
        }
        return NULL;
    }
    
    ColType coltype;//single fixed vary?
    string colname;//column name
    string rttype;//name of the roottype
    string lenname;//name of column that defines its length
    TypeInfo* nptype;//name of numpy type 
    void* payload;
    int payloadsize;//size of payload in number of element keeping track in case we need to realloc
    int size_ele;//size of 1 element
    int* asize;//this should point to the payload of the column that define the length of this column
    int static_asize;//in case of fixed length asize points to here

    TBranch* branch;//the branch (for peeking the value)
    vector<ColumnDescr*> child;//columns that the length is this column
    
    string to_str(){
        string ret = "";
        PyObject* descr = totuple();
        //PyObject_Print(descr,stdout,0);
        //PyObject* str = PyObject_Str(descr);
        //RNHEXDEBUG(str);
        char* tmp = PyString_AsString(PyTuple_GetItem(descr,1));
        char* tmp2 = PyString_AsString(PyTuple_GetItem(descr,0));
        ret = ret+ tmp + " " + tmp2 + " --- (" + lenname + ")";
        //Py_DECREF(str);
        Py_DECREF(descr);
        return ret;
    }
    
    void print_payload(ostream& os,int offset=0){
        if(nptype->npt==NPY_INT32){
            os << ((int*)payload)[offset];
        }else if(nptype->npt==NPY_FLOAT32){
            os << ((float*)payload)[offset];
        }else if(nptype->npt==NPY_FLOAT64){
            os << ((double*)payload)[offset];
        }
    }
    
    void print_value(){
        if(coltype==SINGLE){
            print_payload(cout);
        }else if(coltype==FIXED){
            cout << *asize << "[ ";
            for(int i=0;i<*asize;i++){
                print_payload(cout,i);
                cout << ", ";
            }
            cout << "]";
        }else if(coltype==VARY){
            cout << *asize << "v[ ";
            for(int i=0;i<*asize;i++){
                print_payload(cout,i);
                cout << ", ";
            }
            cout << "]";
        }
    }
};

vector<string> branch_names(TTree* tree){
     //first get list of branches
    vector<string> ret;    
    TObjArray* branches = tree->GetListOfBranches();
    int numbranches = branches->GetEntries();
    for(int ib=0;ib<numbranches;++ib){
        TBranch* branch = dynamic_cast<TBranch*>(branches->At(ib));
        const char* bname = branch->GetName();
        ret.push_back(bname);
    }
    return ret;
}

class TreeStructure{
public: 
    vector<ColumnDescr*> lencols;
    vector<ColumnDescr*> cols;
    
    TTree* tree;
    bool good;
    vector<string> bnames;
    int prepareLength();//prepare the payload for the array column
    TreeStructure(TTree*tree,const vector<string>& bnames):tree(tree),bnames(bnames){
        good=false;
        init();
    }
    
    ~TreeStructure(){
        for(int icol=0;icol<cols.size();icol++){
            cols[icol]->branch->ResetAddress();
            free(cols[icol]);
        }
    }
    
    void init(){
        map<string,ColumnDescr*> colmap;
        //map of name of len column and all the column that has length defined by the key
        map<string,vector<ColumnDescr*> > collenmap;
        for(int i=0;i<bnames.size();i++){
            string bname = bnames[i];
            string colname = bname;
            TBranch* branch = tree->FindBranch(bname.c_str());
            if(branch==0){
                good=false;
                PyErr_SetString(PyExc_IOError,("Unable to get find branch "+bname).c_str());
                return;
            }
            //now get the leaf the type info
            TObjArray* leaves = branch->GetListOfLeaves();
            TLeaf* leaf = dynamic_cast<TLeaf*>(leaves->At(0));
            if(leaf==0){
                good=false;
                PyErr_SetString(PyExc_IOError,("Unable to get first leaf of branch "+bname).c_str());
                return;
            }
            string rttype(leaf->GetTypeName());
            std::map<std::string, TypeInfo>::const_iterator ti = root_typemap.find(rttype);
            if(ti==root_typemap.end()){//no idea how to convert this
                cerr << "Warning: unable to convert " << rttype << " for branch " << bname << ". Skip." << endl;
                continue;
            }
            
            int countval;
            //now check whether it's array if so of which type
            TLeaf* len_leaf = leaf->GetLeafCounter(countval);
            string lenname;
            ColumnDescr::ColType coltype;
            if(countval==1){
                if(len_leaf==0){//single element
                    coltype = ColumnDescr::SINGLE;
                }
                else{//variable length
                    coltype = ColumnDescr::VARY;
                    lenname = len_leaf->GetBranch()->GetName();
                }
            }else if(countval>0){
                //fixed multiple array
                coltype = ColumnDescr::FIXED;
            }else{//negative
                cerr << "Warning: unable to understand the structure of branch " << bname << ". Skip." << endl;
                continue;
            }
            
            //now we have all the information to build this column
            //put in column map and column
            if(coltype==ColumnDescr::SINGLE){
                //build column add it to cols and col map
                ColumnDescr* thiscol =  ColumnDescr::build_single(colname, rttype,tree);
                cols.push_back(thiscol);
                colmap.insert(make_pair(colname,thiscol));
            }else if(coltype==ColumnDescr::FIXED){
                //build column add it to cols and col map
                ColumnDescr* thiscol = ColumnDescr::build_fixed(colname, rttype, countval, tree);
                cols.push_back(thiscol);
                colmap.insert(make_pair(colname,thiscol));
            }else if(coltype){
                //build column add it to cols and col map then update collenmap
                ColumnDescr* thiscol = ColumnDescr::build_vary(colname, rttype, lenname, tree);
                cols.push_back(thiscol);
                colmap.insert(make_pair(colname,thiscol));
                map<string,vector<ColumnDescr*> >::iterator it;
                it = collenmap.find(lenname);
                if(it==collenmap.end()){
                    vector<ColumnDescr*> vcd;
                    vcd.push_back(thiscol);
                    collenmap.insert(make_pair(lenname,vcd));
                }else{
                    it->second.push_back(thiscol);
                }
            }
        }
        
        //now colmap collenmap and cols is build
        //time to cache lencols and update the asize and child fields in cols
        map<string,vector<ColumnDescr*> >::iterator it;
        for(it = collenmap.begin();it!=collenmap.end();++it){
            string colname = it->first;
            ColumnDescr* this_lencol = colmap[colname];
            assert(this_lencol->coltype == ColumnDescr::SINGLE);//realloc will break this
            int *payload = (int*)(this_lencol->payload);
            vector<ColumnDescr*>& childs = it->second;
            //for each child update the asize
            for(int ichild=0;ichild<childs.size();++ichild){
                ColumnDescr* thisChild = childs[ichild];
                thisChild->asize = payload;
            }
            //update childfiled for this_lencol
            this_lencol->child = childs;
        }
        //done!!!!!
        good=true;
    }
    
    PyObject* to_nptype_list(){
        PyObject* mylist = PyList_New(0);
        for(int icols=0;icols<cols.size();++icols){
            ColumnDescr* thiscol = cols[icols];
            PyList_Append(mylist,thiscol->totuple());
        }
        return mylist;
    }
    
    //look ahead for length and prepare the payload accordingly
    void peek(int i){
        for(int ilc=0;ilc<lencols.size();ilc++){
            ColumnDescr* lencol = lencols[ilc];
            lencol->branch->GetEntry(i);//peek only the length column
            vector<ColumnDescr*>& child = lencol->child;
            int newlen = *((int*)(lencol->payload));
            for(int ichild=0;ichild<child.size();ichild++){
                ColumnDescr* thischild = child[i];
                thischild->resize(newlen);
            }
        }
    }
    
    void print_current_value(){
        for(int i=0;i<cols.size();i++){
            cout << cols[i]->colname << ":";
            cols[i]->print_value();
            cout << " ";
        }
    }
    //return list of tuple
    //[('col','f8'),('kkk','i4',(10)),('bbb','object')]
    PyObject* to_descr_list(){
        PyObject* mylist = PyList_New(0);
        for(int i=0;i<cols.size();++i){
            PyList_Append(mylist,cols[i]->totuple());
       }
       return mylist;
    }
    
    int copy_to(void* destination){
        char* current = (char*)destination;
        int total;
        for(int i=0;i<cols.size();++i){
            ColumnDescr* thiscol = cols[i];
            int nbytes = thiscol->copy_to((void*)current);
            current += nbytes;
            total += nbytes;
        }
        return total;
    }
    
    string to_str(){
        string tmp;
        for(int i=0;i<cols.size();i++){
            tmp+=cols[i]->to_str()+"\n";
        }
        return tmp;
    }
};
#endif
