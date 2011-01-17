import xml.parsers.expat

class XMLParser:

    def __init__(self, xml_file):
            
        assert(xml_file != "")
        if type(xml_file) is file:
            self.xml_file = xml_file
        else:
            self.xml_file = open(xml_file, "r")
        
        self.Parser = xml.parsers.expat.ParserCreate()
        self.Parser.CharacterDataHandler = self.handleCharData
        self.Parser.StartElementHandler = self.handleStartElement
        self.Parser.EndElementHandler = self.handleEndElement
  
    def parse(self):
        
        self.Parser.ParseFile(self.xml_file)

    def handleCharData(self, data):
        pass
    def handleStartElement(self, name, attrs):
        pass
    def handleEndElement(self, name):
        pass

import glob
import sys
import os
import re
import commands
import subprocess

class GridXMLParser(XMLParser):
    
    options = None
    
    def __init__(self, file, datadir, testarea, opts):
        
        global options
        XMLParser.__init__(self, file)
        options = opts
        self.datadir = datadir
        self.testarea = testarea
        self.type = None
        self.user = None
        self.lib = None
        self.site = None
        self.cloud = None
        self.release = None
    
    def handleStartElement(self, name, attrs):
        
        if name == "jobs":
            self.type = attrs["type"]
            self.user = attrs["user"]
            self.lib = attrs["lib"]
            self.site = attrs["site"]
            self.cloud = attrs["cloud"]
            self.release = attrs["release"]
        elif name == "job":

            flagged = attrs["flag"] == "1"
            isTest = attrs["test"] == "1"
            if not isTest and options.test:
                return
            if not flagged and options.flagged:
                return
            execString = attrs["exec"]
            inDS = attrs["inDS"]
            output = attrs["outDS"]
            outDS = ".".join([self.user,output])
            print "<<<<< "+inDS+" --> "+outDS+" >>>>>"
           
            extFiles = None
            if attrs.has_key("extFile"):
                extFiles = attrs["extFile"]

            pattern = re.compile('^'+outDS+'.([0-9]+)[/]?$')
            
            localVersion = -1
            # Check for local versions of dataset
            files = glob.glob(self.datadir+'/'+self.type+'/'+outDS+'.*')
            if len(files) == 0:
                print "No local version of this output dataset exists."
            elif len(files) == 1:
                localVersion = int(files[0].split(".")[-1])
                print "Local version of output dataset is %i"% localVersion
            else:
                print "More than one local version of output dataset "+outDS
                print "Please remove old versions"
                sys.exit(0)
            
            #Get latest version on grid
            latest = 0
            files = commands.getoutput('dq2-ls '+outDS+'.\*')
            if files != '':
                files = files.split('\n')
                for file in files:
                    match = re.match(pattern,file)
                    if match:
                        version = int(match.group(1))
                        if version > latest:
                            latest = version
                print "Latest version of output dataset found on grid is %i."% latest
                currSuffix = "%i"% (latest)
            else:
                print "No version of this output dataset found on grid."
                currSuffix = None
            
            if options.get:
                if not currSuffix:
                    print "Skipping..."
                else:
                    print "Attempting to get version "+currSuffix
                    outDS = ".".join([outDS, currSuffix])
                    if os.path.isdir(self.datadir+'/'+self.type+'/'+outDS):
                        print "Local output dataset already at latest version. Skipping..."
                    else:
                        cmd = 'cd '+self.datadir+'/'+self.type+'; dq2-get '+outDS+'/'
                        print cmd
                        if not options.dry:
                            child = subprocess.Popen(args=cmd, shell=True)
                            child.wait()
                            #print commands.getoutput(cmd)
            elif options.submit or options.resubmit:
                skip = False
                if options.onlynew:
                    skip = currSuffix or localVersion != -1
                if not skip:
                    if options.resubmit:
                        newSuffix = "%i"% latest
                    else:
                        if localVersion > latest:
                            print "Version of local output dataset is newer than new version to submit."
                            newSuffix = "%i"% (localVersion+1)
                        else:
                            newSuffix = "%i"% (latest+1)
                    
                    print "Submitting job for "+newSuffix
                    
                    if "*" in inDS:
                        inDSs = commands.getoutput("dq2-ls %s"% inDS).split('\n')
                        if len(inDSs) > 1:
                            print "Warning: more than one possible inDS matches pattern. Using first."
                        inDS = inDSs[0]
                    
                    outDS = ".".join([outDS, newSuffix])
                    #--cloud='+self.cloud+'
                    #--site='+self.site+' 
                    cmd = 'cd '+self.testarea+'; prun --exec "'+execString+'" --outDS '+outDS+' --inDS '+\
                        inDS+' --outputs '+output+'.root --useAthenaPackages'#--athenaTag='+self.release
                    if options.dry:
                        cmd += ' --noSubmit'
                    if options.test:
                        cmd += ' --nFiles 1'
                    elif options.uselib:
                        cmd += ' --libDS '+self.lib
                    if (not options.site) and (self.site != ''):
                        cmd += ' --site '+self.site
                    elif options.site != "auto":
                        cmd += ' --site '+options.site
                    if extFiles != None:
                        cmd += ' --extFile '+extFiles
                    print cmd
                    
                    child = subprocess.Popen(args=cmd, shell=True)
                    child.wait()
                    #print commands.getoutput(cmd)
                else:
                    print "Skipping..."
            print ""
