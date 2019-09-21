#!/usr/bin/python

# Code for generating and reading graphrat mazes
# Parameters:
# k: Grid size will be k * k nodes
# Type:
# u : uniform
# t : tiled
# v : vertical slices
# h : horizontal slices
# p : parquet
# i : irregular


import getopt
import sys
import math
import string
import datetime

import rutil

def usage(name):
    print "Usage: %s [-z] [-h] [-k K] [-t (u|t|v|h|p|i)] [-l L:H] [-o OUT] [-s SEED]"
    print "\t-h     Print this message"
    print "\t-z     Include zones in graph file"
    print "\t-k K   Base graph as k x k grid"
    print "\t-t     Specify graph type:"
    print "\t       u : uniform"
    print "\t       t : tiled"
    print "\t       v : vertical slices"
    print "\t       h : horizontal slices"
    print "\t       p : parquet"
    print "\t       i : irregular"
    print "\t-l L:H Specify range of ideal load factors"
    print "\t-o OUT Specify output file"
    sys.exit(0)

def trim(s):
    while len(s) > 0 and s[-1] in '\n\r':
        s = s[:-1]
    return s


class RatMode:
    # Different options for specifying initial rat state
    (uniform, diagonal, upleft, lowright) =  range(4)
    modeNames = ["uniform", "diagonal", "upper-left", "lower-right"]

class GraphType:
    # Different graph types
    (uniform, tiled, vertical, horizontal, parquet, irregular) = range(6)
    modeNames = ["uniform", " tiled", " vertical", " horizontal", " parquet", " irregular"]
    modeTags = ['u', 't', 'v', 'h', 'p', 'i']
    
    def __init__(self):
        pass

    def getType(self, tag):
        found = False
        for gtype in range(len(self.modeTags)):
            if tag == self.modeTags[gtype]:
                found = True
                break
        return gtype if found else -1
            
    def feasible(self, gtype, k):
        cells = 1
        if gtype in [self.vertical, self.horizontal, self.irregular]:
            cells = 12
        if gtype in [self.tiled, self.parquet]:
            cells = 6
        return k % cells == 0

class Graph:
    k = 0
    nodeCount = 0
    edges = {}  # Maps edges to True.  Include both directions
    commentList = []  # Documentation about how generated
    nodeList = [] # Node ideal load factors
    ilfRange = (1.2,1.8) # Range of ideal load factors
    doZone = False
    zoneList = []
    rng = None

    def __init__(self, k = 0, gtype = GraphType.uniform, ilf = None, doZone = False):
        self.generate(k, gtype = gtype, ilf = ilf, doZone = doZone)

    def generate(self, k = 12, gtype = GraphType.uniform, ilf = None, seed = rutil.DEFAULTSEED, doZone = False):
        gt = GraphType()
        if not gt.feasible(gtype, k):
            print "Cannot generate graph of type %s for k = %d" % (gt.modeNames[gtype], k)
            return
        self.rng = rutil.RNG([seed])
        if ilf is not None:
            self.ilfRange = ilf
        self.commentList = []
        tgen = datetime.datetime.now()
        self.commentList.append("# Generated %s" % tgen.ctime())
        self.commentList.append("# Parameters: k = %d, type = %s, ilf = (%.2f,%.2f)" % (k, gt.modeNames[gtype], self.ilfRange[0], self.ilfRange[1]))
        self.k = k
        self.nodeCount = k * k
        self.nodeList = [self.assignIlf(i) for i in range(self.nodeCount)]
        self.edges = {}
        self.doZone = doZone
        self.zoneList = []
        # Generate grid edges
        for r in range(k):
            for c in range(k):
                own = self.id(r, c)
                north = self.id(r-1, c)
                if north >= 0:
                    self.addEdge(own, north)
                south = self.id(r+1, c)
                if south >= 0:
                    self.addEdge(own, south)
                west = self.id(r, c-1)
                if west >= 0:
                    self.addEdge(own, west)
                east = self.id(r, c+1)
                if east >= 0:
                    self.addEdge(own, east)
        if gtype in [gt.tiled, gt.vertical, gt.horizontal]:
            cells = 6 if gtype == gt.tiled else 12
            unit = self.k/cells
            tileX = unit if gtype in [gt.tiled, gt.vertical] else self.k
            tileY = unit if gtype in [gt.tiled, gt.horizontal] else self.k
            self.tile(tileX, tileY)

        elif gtype == gt.parquet:
            self.parquet()
        elif gtype == gt.irregular:
            self.irregular()
        elif gtype == gt.uniform and self.doZone:
            for r in range(k):
                for c in range(k):
                    self.zoneList.append((c, r, 1, 1))
        elif gtype != gt.uniform:
            print "Unknown graph type %d" % gtype


    def tile(self, tileX, tileY):
        if tileY == 0:
            tileY = tileX
        if tileX == 0:
            tileX = tileY
        for x in range(0, self.k, tileX):
            w = min(tileX, self.k - x)
            for y in range(0, self.k-tileY+1, tileY):
                h = min(tileY, self.k - y)
                self.makeHubs(x, y, w, h)
        if self.doZone:
            if tileX == self.k/6 and tileY == self.k/6:
                # Create different rectangular zones
                w, h = self.k/2, self.k/6
            else:
                w, h = tileX, tileY
            for x in range(0, self.k, w):
                for y in range(0, self.k, h):
                    self.zoneList.append((x, y, w, h))

    def parquet(self):
        cells = 6
        unit = self.k/cells
        # Upper Left
        x = 0
        w = unit * 3
        h = unit
        ystart = 0
        yend = ystart + 3 * h
        for y in range(ystart, yend, h):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))
        # Upper Right
        w = unit
        y = 0
        h = unit * 3
        xstart = unit * 3
        xend = xstart + 3 * w
        for x in range(xstart, xend, w):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))
        # Lower Left
        w = unit
        y = unit * 3
        h = unit * 3
        xstart = 0
        xend = xstart + 3 * w
        for x in range(xstart, xend, w):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))
        # Lower Right
        x = unit * 3
        w = unit * 3
        h = unit
        ystart = unit * 3
        yend = ystart + 3 * h
        for y in range(ystart, yend, h):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))

    def irregular(self):
        cells = 12
        unit = self.k/cells
        # Upper Left
        x = 0
        w = unit * 4
        y = 0
        h = unit * 6
        self.makeHubs(x, y, w, h)
        if self.doZone:
            self.zoneList.append((x, y, w, h))

        # Upper Right
        x = unit * 4
        w = unit * 8
        h = unit * 2
        ystart = 0
        yend = ystart + 3 * h
        for y in range(ystart, yend, h):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))
        # Lower Left
        x = 0
        w = unit * 6
        h = unit * 3
        ystart = unit * 6
        yend = ystart + 2 * h
        for y in range(ystart, yend, h):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))
        # Lower Right
        w = unit * 3
        y = unit * 6
        h = unit * 6
        xstart = unit * 6
        xend = xstart + 2 * w
        for x in range(xstart, xend, w):
            self.makeHubs(x, y, w, h)
            if self.doZone:
                self.zoneList.append((x, y, w, h))

    def makeHubs(self, x, y, w, h, xcount=1, ycount=1):
        if w <= 2*xcount:
            wsep = w/xcount
        else:
            wsep = w/(xcount + 1)

        hsep = h/(ycount + 1)
        
        if w <= xcount:
            cxList = [x + wsep * i for i in range(xcount)]
        elif w <= 2*xcount:
            cxList = [1 + x + wsep * i for i in range(xcount)]
        else:
            cxList = [x + wsep * (i + 1) for i in range(xcount)]
        cyList = [y + hsep * (i + 1) for i in range(ycount)]
        for cx in cxList:
            for cy in cyList:
                cid = self.id(cy, cx)
                for j in range(w):
                    for i in range(h):
                        id = self.id(y+i, x+j)
                        self.addEdge(cid, id)
        
    # Check whether string is a comment
    def isComment(self, s):
        # Strip off leading whitespace
        while len(s) > 0 and s[0] in string.whitespace:
            s = s[1:]
        return len(s) == 0 or s[0] == '#'

    # Load graph from file
    def load(self, fname = ""):
        self.k = 0
        self.nodeList = []
        self.edges = {}
        if fname == "":
            f = sys.stdin
        else:
            try:
                f = open(fname, "r")
            except:
                sys.stderr.write("Could not open file '%s'\n" % fname)
                return False
        expectedEgeCount = 0
        realEdgeCount = 0
        realNodeCount = 0
        for line in f:
            if self.isComment(line):
                continue
            args = line.split()
            if len(args) == 0:
                continue
            cmd = args[0]
            # Header information
            if self.k == 0:
                self.nodeCount = int(args[0])
                self.nodeList = [1.5 for i in range(self.nodeCount)]
                self.k = int(math.sqrt(self.nodeCount))
                expectedEdgeCount = int(args[1])
            elif cmd == 'n':
                ilf = float(args[1])
                self.nodeList[realNodeCount] = ilf
                realNodeCount += 1
            elif cmd == 'e':
                i = int(args[1])
                j = int(args[2])
                if self.addEdge(i,j):
                    # Since addEdge puts both (i,j) and (j,i) into set, only half of the
                    # edges will return True from addEdge
                    realEdgeCount += 2 
            elif cmd == 'z':
                # Ignore zones
                pass
            else:
                sys.stderr.write("Error reading graph file '%s'.  Invalid line: '%'" % fname, trim(line))
                return False
        if fname != "":
            f.close()
        if realNodeCount != self.nodeCount:
            sys.stderr.write("Error reading graph file '%s'.  Expected %d nodes.  Found %d\n" % (fname, self.nodeCount, realNodeCount))
            return False
        if realEdgeCount != expectedEdgeCount:
            sys.stderr.write("Error reading graph file '%s'.  Expected %d edges.  Found %d\n" % (fname, expectedEdgeCount, realEdgeCount))
            return False
        else:
            sys.stderr.write("Read graph with %d nodes and %d edges\n" % (self.nodeCount, realEdgeCount))
            return True
 
    def id(self, r, c):
        if r < 0 or r >= self.k:
            return -1
        if c < 0 or c >= self.k:
            return -1
        return r * self.k + c

    def rowColumn(self, id):
        r = id/self.k
        c = id - r*self.k
        return (r, c)

    # Set ideal load factors to be maximum on right and minimum on left
    def assignIlf(self, id):
        r, c = self.rowColumn(id)
        delta = self.ilfRange[1] - self.ilfRange[0]
        return self.ilfRange[0] + self.rng.randFloat(delta)

    def addEdge(self, i, j):
        nodeCount = len(self.nodeList)
        if i < 0 or i >= nodeCount:
            sys.stderr.write("Error: Invalid from node id %d\n" % i)
        if j < 0 or j >= nodeCount:
            sys.stderr.write("Error: Invalid to node id %d\n" % j)
        if i != j and (i,j) not in self.edges:
            self.edges[(i,j)] = True
            self.edges[(j,i)] = True
            return True
        return False
            
    def edgeList(self):
        elist = [e for e in self.edges]
        elist.sort()
        return elist

    # Generate list with entry with each node, giving its degree (including self)
    def degreeList(self):
        result = [1] * len(self.nodeList)
        for e in self.edges:
            idx = e[0]
            result[idx] += 1
        return result

    # Store graph
    def store(self, fname = ""):
        if fname == "":
            f = sys.stdout
        else:
            try:
                f = open(fname, "w")
            except:
                sys.stderr.write("Error.  Couldn't open file '%s' for writing\n" % (fname))
                return False
        elist = self.edgeList()
        if self.doZone:
            f.write("%d %d %d\n" % (len(self.nodeList), len(elist), len(self.zoneList)))
        else:
            f.write("%d %d\n" % (len(self.nodeList), len(elist)))
        for c in self.commentList:
            f.write(c + '\n')
        for i in range(len(self.nodeList)):
            f.write("n %.5f\n" % self.nodeList[i])
        for e in elist:
            f.write("e %d %d\n" % e)
        for z in self.zoneList:
            f.write("z %d %d %d %d\n" % z)
        if fname != "":
            f.close()
        return True

    # Generate rats for graph and write to file
    def makeRats(self, fname = "", mode = RatMode.uniform, load = 1, seed = rutil.DEFAULTSEED):
        nodeCount = len(self.nodeList)
        clist = []
        tgen = datetime.datetime.now()
        clist.append("# Generated %s" % tgen.ctime())
        clist.append("# Parameters: load = %d, mode = %s, seed = %d" % (load, RatMode.modeNames[mode], seed))
        rng = rutil.RNG([seed])
        if fname == "":
            f = sys.stdout
        else:
            try:
                f = open(fname, "w")
            except:
                "Couldn't open output file '%s'"
                return False
        rlist = []
        if mode == RatMode.uniform:
            rlist = range(nodeCount)
        elif mode == RatMode.diagonal:
            rlist = [(self.k+1)*i for i in range(self.k)]
        elif mode == RatMode.upleft:
            rlist = [0]
        elif mode == RatMode.lowright:
            rlist = [nodeCount-1]
        else:
            sys.stderr.write("ERROR: Invalid rat mode\n")
            return False
        factor = nodeCount * load / len(rlist)
        fullRlist = rlist * factor
        if len(rlist) > 0:
            fullRlist = rng.permute(fullRlist)
        # Print it out
        f.write("%d %d\n" % (nodeCount, len(fullRlist)))
        for c in clist:
            f.write(c + '\n')
        for id in fullRlist:
            f.write("%d\n" % id)
        if fname != "":
            f.close()
        return True

# Runtime code for graph generation
def run(name, args):
    k = 10
    gtype = GraphType.uniform
    fname = ""
    ilf = None
    doZone = False
    optlist, args = getopt.getopt(args, "hzk:t:l:o:")
    for (opt, val) in optlist:
        if opt == '-h':
            usage(name)
        if opt == '-z':
            doZone = True
        if opt == '-k':
            k = int(val)
        if opt == '-t':
            gt = GraphType()
            gtype = gt.getType(val)
            if gtype < 0:
                print "Uknown graph type '%s'" % val
                usage(name)
        if opt == '-l':
            fields = val.split(':')
            if fields != 2:
                print "Ideal load factor requires two parameters"
                usage(name)
            try:
                lval = float(fields[0])
                hval = float(fields[1])
            except:
                print "Ideal load factor requires two numeric parameters"
                usage(name)
            ilf = (lval, hval)
        if opt == '-o':
            fname = val
    g = Graph(k = k, gtype = gtype, ilf = ilf, doZone = doZone)
    g.store(fname = fname)

if __name__ == "__main__":
    run(sys.argv[0], sys.argv[1:])


        
    
