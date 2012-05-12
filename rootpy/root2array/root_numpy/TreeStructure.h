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
#include <TObject.h>
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

bool convertible(const string& rt){
    std::map<std::string, TypeInfo>::iterator it;
    it = root_typemap.find(rt);
    return it!=root_typemap.end();
}

//missing string printf
//this is safe and convenient but not exactly efficient
inline std::string format(const char* fmt, ...){
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl,fmt);
    int nsize = vsnprintf(buffer,size,fmt,vl);
    if(size<=nsize){//fail delete buffer and try again
        delete buffer; buffer = 0;
        buffer = new char[nsize+1];//+1 for /0
        nsize = vsnprintf(buffer,size,fmt,vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete buffer;
    return ret;
}

class Column{
public:
    enum ColType{SINGLE=1,FIXED=2,VARY=3};
    TLeaf* leaf;
    bool skipped;
    ColType coltype;//single fixed vary?
    string colname;//column name
    int countval; //useful in case of fixed element
    string rttype;//name of the roottype
    TypeInfo* tinfo;//name of numpy type

    static Column* build(TLeaf* leaf,const string& colname){
        Column* ret = new Column();
        ret->leaf = leaf;
        ret->colname = colname;
        ret->skipped = false;
        int ok = find_coltype(leaf,ret->coltype,ret->countval);
        if(!ok){
            delete ret;
            return NULL;
        }
        ret->rttype = leaf->GetTypeName();
        ret->tinfo = rt2npt(ret->rttype,true);
        return ret;
    }
    
    void SetLeaf(TLeaf* newleaf, bool paranoidmode=false){
        leaf = newleaf;
        if(paranoidmode){
            assert(leaf->GetTypeName() == rttype);
            int cv;
            ColType ct;
            int ok = find_coltype(leaf,ct,cv);
            assert(ok!=NULL);
            assert(ct==coltype);
            //if(ct==FIXED){assert(cv==countval);}
        }
    }
    
    static int find_coltype(TLeaf* leaf, Column::ColType& coltype, int& countval ){
        //now check whether it's array if so of which type
        TLeaf* len_leaf = leaf->GetLeafCounter(countval);
        if(countval==1){
            if(len_leaf==0){//single element
                coltype = Column::SINGLE;
            }
            else{//variable length          
                coltype = Column::VARY;
            }
        }else if(countval>0){
            //fixed multiple array
            coltype = Column::FIXED;
        }else{//negative
            string msg("Unable to understand the structure of leaf ");
            msg += leaf->GetName();
            PyErr_SetString(PyExc_IOError,msg.c_str());
            return NULL;
        }
        return 1;
    }
    
    //copy to numpy element destination
    //and return number of byte written
    int copy_to(void* destination){
        if(skipped){
            if(coltype==FIXED){
                return countval*tinfo->size;
            }else if(coltype==SINGLE){
                return tinfo->size;
            }else if(coltype==VARY){
                //make empty array
                npy_intp dims[1];
                dims[0]=0;
                PyArrayObject* newobj = (PyArrayObject*)PyArray_EMPTY(1,dims,tinfo->npt,0);
                assert(newobj!=0);
                memcpy(destination,&newobj,sizeof(PyArrayObject*));
                return sizeof(PyObject*);
            }else{
                assert(false);//shouldn't reach here
                return 0;
            }
        }
        else{
            int ret;
            if(coltype==FIXED || coltype==SINGLE){
                assert(leaf!=NULL);
                void* src = leaf->GetValuePointer();
                assert(src!=NULL);
                ret = leaf->GetLenType()*leaf->GetLen();
                assert(ret>=0);
                memcpy(destination,src,ret);
            }else{//variable length array
                //build a numpy array of the length and put pyobject there
                void* src = leaf->GetValuePointer();
                int sizetocopy = leaf->GetLenType()*leaf->GetLen();
                npy_intp dims[1];
                dims[0]=leaf->GetLen();
                PyArrayObject* newobj = (PyArrayObject*)PyArray_EMPTY(1,dims,tinfo->npt,0);
                assert(newobj!=0);
                memcpy(newobj->data,src,sizetocopy);
                memcpy(destination,&newobj,sizeof(PyArrayObject*));
                ret = sizeof(PyObject*);
            }
            return ret;
        }
        assert(false);//shoudln't reach here
        return 0;
    }
    //convert to PyArray_Descr tuple
    PyObject* totuple(){
        //return ('col','f8')
        if(coltype==SINGLE){
            
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString(tinfo->nptype.c_str());
            PyObject* nt_tuple = PyTuple_New(2);
            PyTuple_SetItem(nt_tuple,0,pyname);
            PyTuple_SetItem(nt_tuple,1,pytype);
            char* tmp = PyString_AsString(pytype);
            return nt_tuple;
        }else if(coltype==FIXED){//return ('col','f8',(10))
            PyObject* pyname = PyString_FromString(colname.c_str());

            PyObject* pytype = PyString_FromString(tinfo->nptype.c_str());

            PyObject* subsize = PyTuple_New(1);
            PyObject* pysubsize = PyInt_FromLong(countval);
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

};

//correct TChain implementation with cache TLeaf*
class BetterChain{
public:
    class MiniNotify:public TObject{
    public:
        bool notified;
        TObject* oldnotify;
        MiniNotify(TObject* oldnotify):TObject(),notified(false),oldnotify(oldnotify){}
        virtual Bool_t Notify(){
            notified=true;
            if(oldnotify) oldnotify->Notify();
            return true;
        }
    };
    
    TTree* fChain;
    int fCurrent;
    MiniNotify* notifier;
    BetterChain(TTree* fChain):fChain(fChain){
        fCurrent = -1;
        notifier = new MiniNotify(fChain->GetNotify());
        fChain->SetNotify(notifier);
        LoadTree(0);
        fChain->SetBranchStatus("*",0);//disable all branches
        //fChain->SetCacheSize(10000000);
    }
    ~BetterChain(){
        // if (!fChain) return;//some how i need this(copy from make class)
        //         delete fChain->GetCurrentFile();
        LeafCache::iterator it;
        for(it=leafcache.begin();it!=leafcache.end();++it){
            delete it->second;
        }
        fChain->SetNotify(notifier->oldnotify);
        delete notifier;
    }
    typedef pair<string,string> BL;
    typedef map<BL,Column*> LeafCache;
    LeafCache leafcache;

    int LoadTree(int entry){
        if (!fChain) return -5;
        //RNHEXDEBUG(fChain->FindBranch("mcLen")->FindLeaf("mcLen"));
        //some how load tree chnage the leaf even when 
        Long64_t centry = fChain->LoadTree(entry);
        //RNHEXDEBUG(fChain->FindBranch("mcLen")->FindLeaf("mcLen"));
        if (centry < 0) return centry;
        if (fChain->GetTreeNumber() != fCurrent) {
           fCurrent = fChain->GetTreeNumber();
        }
        if(notifier->notified){
            Notify();
            notifier->notified=false;
        }
        return centry;
    }
    
    int GetEntry(int entry){
        // Read contents of entry.
        if (!fChain) return 0;
        LoadTree(entry);
        return fChain->GetEntry(entry);
    }
    
    void Notify(){
        //taking care of all the leaves
        //RNDEBUG("NOTIFY");
        LeafCache::iterator it;
        for(it=leafcache.begin();it!=leafcache.end();++it){
            string bname = it->first.first;
            string lname = it->first.second;
            TBranch* branch = fChain->FindBranch(bname.c_str());
            if(branch==0){
                cerr << "Warning cannot find branch " << bname << endl;
                it->second->skipped = true;
                continue;
            }
            TLeaf* leaf = branch->FindLeaf(lname.c_str());
            if(leaf==0){
                cerr << "Warning cannot find leaf " << lname << " for branch " << bname << endl;
                it->second->skipped = true;
                continue;
            }
            it->second->SetLeaf(leaf,true); 
            it->second->skipped = false;
        }
    }
    
    int GetEntries(){
        int ret = fChain->GetEntries();
        return ret;
    }
    
    TBranch* FindBranch(const char* bname){
        return fChain->FindBranch(bname);
    }
    
    Column* MakeColumn(const string& bname, const string& lname, const string& colname){
        //as bonus set branch status on all the active branch including the branch that define the length
        LoadTree(0);

        TBranch* branch = fChain->FindBranch(bname.c_str());
        if(branch==0){
            PyErr_SetString(PyExc_IOError,format("Cannot find branch %s",bname.c_str()).c_str());
            return 0;
        }
        
        TLeaf* leaf = fChain->FindLeaf(lname.c_str());
        if(leaf==0){
            PyErr_SetString(PyExc_IOError,format("Cannot find leaf %s for branch %s",lname.c_str(),bname.c_str()).c_str());
            return 0;
        }
        
        //make sure we know how to convert this
        const char* rt = leaf->GetTypeName();
        assert(convertible(rt)); //we already check this
        
        //make the branch active
        //and cache it
        fChain->SetBranchStatus(bname.c_str(),1);
        fChain->AddBranchToCache(branch,kTRUE);
        //and the length leaf as well
        TLeaf* leafCount = leaf->GetLeafCount();
        if(leafCount != 0){
            fChain->SetBranchStatus(leafCount->GetBranch()->GetName(),1);
            fChain->AddBranchToCache(leafCount->GetBranch(),kTRUE);
        }
        
        BL bl = make_pair(bname,lname);
        Column* ret = Column::build(leaf,colname);
        if(ret==0){return 0;}
        leafcache.insert(make_pair(bl,ret));
        return ret;
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
    vector<Column*> cols;//i don't own this
    BetterChain bc;
    bool good;
    vector<string> bnames;
    
    TreeStructure(TTree*tree,const vector<string>& bnames):bc(tree),bnames(bnames){
        good=false;
        init();
    }
    
    void init(){
        //TODO: refractor this
        //goal here is to fil cols array
        //map of name of len column and all the column that has length defined by the key
        for(int i=0;i<bnames.size();i++){
            string bname = bnames[i];
            TBranch* branch = bc.FindBranch(bname.c_str());
            if(branch==0){
                good=false;
                PyErr_SetString(PyExc_IOError,("Unable to get branch "+bname).c_str());
                return;
            }
            //now get the leaf the type info
            TObjArray* leaves = branch->GetListOfLeaves();
            int numleaves = leaves->GetEntries();
            bool shortname = numleaves==1;
            
            for(int ileaves=0;ileaves<numleaves;ileaves++){
                TLeaf* leaf = dynamic_cast<TLeaf*>(leaves->At(ileaves));
                if(leaf==0){
                    good=false;
                    PyErr_SetString(PyExc_IOError,format("Unable to get leaf %s for branch %s",leaf->GetName(),branch->GetName()).c_str());
                    return;
                }

                string rttype(leaf->GetTypeName());
                if(!convertible(rttype)){//no idea how to convert this
                    cerr << "Warning: unable to convert " << rttype << " for branch " << bname << ". Skip." << endl;
                    continue;
                }

                //figure out column name
                string colname;
                if(shortname){colname=bname;}
                else{colname=format("%s_%s",bname.c_str(),leaf->GetName());}
                
                Column* thisCol = bc.MakeColumn(bname,leaf->GetName(),colname);
                if(thisCol==0){return;}
                cols.push_back(thisCol);
            }//end for each laves
        }//end for each branch
        
        good=true;
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
            Column* thiscol = cols[i];
            int nbytes = thiscol->copy_to((void*)current);
            current += nbytes;
            total += nbytes;
        }
        return total;
    }

    //convert all leaf specified in lis to numpy structured array
    PyObject* build_array(){
        using namespace std;
        int numEntries = bc.GetEntries();
        PyObject* numpy_descr = to_descr_list();
        if(numpy_descr==0){return NULL;}
        //build the array

        PyArray_Descr* descr;
        int kkk = PyArray_DescrConverter(numpy_descr,&descr);
        Py_DECREF(numpy_descr);

        npy_intp dims[1];
        dims[0]=numEntries;

        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromDescr(1,dims,descr);

        //assume numpy array is contiguous
        char* current = NULL;
        //now put stuff in array
        for(int iEntry=0;iEntry<numEntries;++iEntry){
            int ilocal = bc.LoadTree(iEntry);
            bc.GetEntry(iEntry);
            current = (char*)PyArray_GETPTR1(array, iEntry);
            int nbytes = copy_to((void*)current);
            current+=nbytes;
        }
        return (PyObject*)array;
    }
};
#endif
